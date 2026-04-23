using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

namespace LlamaCpp.Bindings;

/// <summary>
/// Internals of <see cref="JsonSchemaToGbnf"/>. Split out so the public
/// facade stays readable; this file holds the schema walker, the rule
/// table maintenance, the integer-range builder, and the regex-to-GBNF
/// compiler. Structure mirrors
/// <c>llama.cpp/common/json-schema-to-grammar.cpp</c>'s
/// <c>common_schema_converter</c> class.
/// </summary>
public static partial class JsonSchemaToGbnf
{
    internal sealed class SchemaConverter
    {
        private readonly bool _dotall;
        // SortedDictionary so format_grammar output is stable across runs.
        private readonly SortedDictionary<string, string> _rules = new(StringComparer.Ordinal);
        private readonly Dictionary<string, object?> _refs = new(StringComparer.Ordinal);
        private readonly HashSet<string> _refsBeingResolved = new(StringComparer.Ordinal);
        private readonly List<string> _errors = new();
        private readonly List<string> _warnings = new();

        public SchemaConverter(bool dotall)
        {
            _dotall = dotall;
            _rules["space"] = SpaceRule;
        }

        public void CheckErrors()
        {
            if (_errors.Count > 0)
            {
                throw new JsonSchemaConversionException(
                    "JSON schema conversion failed:\n" + string.Join("\n", _errors));
            }
        }

        public string FormatGrammar()
        {
            var sb = new StringBuilder();
            foreach (var kv in _rules)
            {
                sb.Append(kv.Key).Append(" ::= ").Append(kv.Value).Append('\n');
            }
            return sb.ToString();
        }

        /// <summary>
        /// Register a rule. If a rule with the same escaped name already
        /// exists with different content, disambiguate by appending an
        /// index until we find a free slot. Returns the final rule name
        /// the caller should reference.
        /// </summary>
        public string AddRule(string name, string rule)
        {
            var escName = InvalidRuleChars.Replace(name, "-");
            if (!_rules.TryGetValue(escName, out var existing) || existing == rule)
            {
                _rules[escName] = rule;
                return escName;
            }
            int i = 0;
            while (_rules.TryGetValue(escName + i, out var existing2) && existing2 != rule)
            {
                i++;
            }
            var key = escName + i;
            _rules[key] = rule;
            return key;
        }

        /// <summary>
        /// Emit one of the built-in primitive / string-format rules and
        /// every transitively required dependency.
        /// </summary>
        public string AddPrimitive(string name, BuiltinRule rule)
        {
            var n = AddRule(name, rule.Content);
            foreach (var dep in rule.Deps)
            {
                if (_rules.ContainsKey(dep)) continue;
                if (PrimitiveRules.TryGetValue(dep, out var d1)) AddPrimitive(dep, d1);
                else if (StringFormatRules.TryGetValue(dep, out var d2)) AddPrimitive(dep, d2);
                else _errors.Add($"Rule {dep} not known");
            }
            return n;
        }

        /// <summary>
        /// Generate a <c>a | b | c</c> rule body where each alternative is
        /// the result of visiting a sub-schema.
        /// </summary>
        private string GenerateUnionRule(string name, IReadOnlyList<object?> altSchemas)
        {
            var rules = new List<string>(altSchemas.Count);
            for (int i = 0; i < altSchemas.Count; i++)
            {
                var altName = name + (name.Length == 0 ? "alternative-" : "-") + i;
                rules.Add(Visit(altSchemas[i], altName));
            }
            return string.Join(" | ", rules);
        }

        private static string GenerateConstantRule(object? value) =>
            FormatLiteral(JsonDump(value));

        /// <summary>
        /// Resolve every <c>$ref</c> in <paramref name="schema"/>,
        /// recording the referenced target in <see cref="_refs"/> so
        /// <see cref="ResolveRef"/> can materialise it on demand. Local
        /// refs ("#/defs/X") are supported; remote (https://) are not.
        /// </summary>
        public void ResolveRefs(object? schema, string url)
        {
            void Visit(object? n)
            {
                if (n is List<object?> arr)
                {
                    foreach (var x in arr) Visit(x);
                }
                else if (n is OrderedJsonObject obj)
                {
                    if (obj.TryGetValue("$ref", out var refVal) && refVal is string refStr)
                    {
                        if (_refs.ContainsKey(refStr)) return;
                        object? target;
                        string finalRef;

                        if (refStr.StartsWith("https://", StringComparison.Ordinal))
                        {
                            _errors.Add($"Unsupported remote ref: {refStr}");
                            return;
                        }
                        if (refStr.StartsWith("#/", StringComparison.Ordinal))
                        {
                            target = schema;
                            finalRef = url + refStr;
                            obj["$ref"] = finalRef;
                        }
                        else
                        {
                            _errors.Add($"Unsupported ref: {refStr}");
                            return;
                        }

                        var hashIdx = finalRef.IndexOf('#');
                        var pointer = hashIdx < 0 ? string.Empty : finalRef.Substring(hashIdx + 1);
                        var tokens = pointer.Split('/');
                        for (int i = 1; i < tokens.Length; i++)
                        {
                            var sel = tokens[i];
                            if (target is OrderedJsonObject tObj && tObj.ContainsKey(sel))
                            {
                                target = tObj[sel];
                            }
                            else if (target is List<object?> tArr)
                            {
                                if (int.TryParse(sel, out var idx) && idx < tArr.Count)
                                {
                                    target = tArr[idx];
                                }
                                else
                                {
                                    _errors.Add($"Error resolving ref {finalRef}: {sel} not in array");
                                    return;
                                }
                            }
                            else
                            {
                                _errors.Add($"Error resolving ref {finalRef}: {sel} not found");
                                return;
                            }
                        }
                        _refs[finalRef] = target;
                    }
                    else
                    {
                        foreach (var kv in obj.Items()) Visit(kv.Value);
                    }
                }
            }
            Visit(schema);
        }

        private string ResolveRef(string refStr)
        {
            var hashIdx = refStr.IndexOf('#');
            var fragment = hashIdx < 0 ? refStr : refStr.Substring(hashIdx + 1);
            var refName = "ref" + Regex.Replace(fragment, "[^a-zA-Z0-9-]+", "-");
            if (!_rules.ContainsKey(refName) && !_refsBeingResolved.Contains(refStr))
            {
                _refsBeingResolved.Add(refStr);
                if (!_refs.TryGetValue(refStr, out var resolved))
                {
                    _errors.Add($"Unresolved ref: {refStr}");
                    _refsBeingResolved.Remove(refStr);
                    return refName;
                }
                refName = Visit(resolved, refName);
                _refsBeingResolved.Remove(refStr);
            }
            return refName;
        }

        // ========================================================
        // visit — the schema dispatcher. Mirrors the cascade of
        // conditions in common_schema_converter::visit.
        // ========================================================

        public string Visit(object? schemaObj, string name)
        {
            var schema = schemaObj as OrderedJsonObject;
            if (schema is null)
            {
                // Non-object schema — treat as empty (accept any value).
                return AddRule(name.Length == 0 ? "root" : name,
                    AddPrimitive("value", PrimitiveRules["value"]));
            }

            var schemaType = schema.TryGetValue("type", out var t) ? t : null;
            var schemaFormat = schema.TryGetValue("format", out var f) && f is string fs ? fs : string.Empty;
            var ruleName = IsReservedName(name) ? name + "-" : (name.Length == 0 ? "root" : name);

            // $ref
            if (schema.TryGetValue("$ref", out var refVal) && refVal is string refStr)
            {
                return AddRule(ruleName, ResolveRef(refStr));
            }

            // oneOf / anyOf
            if (schema.ContainsKey("oneOf") || schema.ContainsKey("anyOf"))
            {
                var alt = (schema["oneOf"] ?? schema["anyOf"]) as List<object?>
                          ?? new List<object?>();
                return AddRule(ruleName, GenerateUnionRule(name, alt));
            }

            // type: ["string", "null"] — treat each as its own schema then union.
            if (schemaType is List<object?> typeList)
            {
                var schemaCopies = new List<object?>(typeList.Count);
                foreach (var ty in typeList)
                {
                    var copy = new OrderedJsonObject();
                    foreach (var kv in schema.Items()) copy[kv.Key] = kv.Value;
                    copy["type"] = ty;
                    schemaCopies.Add(copy);
                }
                return AddRule(ruleName, GenerateUnionRule(name, schemaCopies));
            }

            // const
            if (schema.TryGetValue("const", out var constVal))
            {
                return AddRule(ruleName, GenerateConstantRule(constVal) + " space");
            }

            // enum
            if (schema.TryGetValue("enum", out var enumObj) && enumObj is List<object?> enumList)
            {
                var enumRules = new List<string>(enumList.Count);
                foreach (var v in enumList) enumRules.Add(GenerateConstantRule(v));
                return AddRule(ruleName, "(" + string.Join(" | ", enumRules) + ") space");
            }

            // object with explicit structure
            bool typeIsNullOrObject = schemaType is null || (schemaType is string ts1 && ts1 == "object");
            if (typeIsNullOrObject
                && (schema.ContainsKey("properties")
                    || (schema.TryGetValue("additionalProperties", out var addPropObj)
                        && !(addPropObj is bool b1 && b1))))
            {
                var required = new HashSet<string>(StringComparer.Ordinal);
                if (schema.TryGetValue("required", out var reqObj) && reqObj is List<object?> reqList)
                {
                    foreach (var r in reqList) if (r is string rs) required.Add(rs);
                }
                var properties = new List<KeyValuePair<string, object?>>();
                if (schema.TryGetValue("properties", out var propsObj) && propsObj is OrderedJsonObject propsMap)
                {
                    foreach (var kv in propsMap.Items()) properties.Add(kv);
                }
                var additional = schema.TryGetValue("additionalProperties", out var addProp) ? addProp : null;
                return AddRule(ruleName, BuildObjectRule(properties, required, name, additional));
            }

            // allOf
            bool typeIsNullObjectOrString =
                schemaType is null
                || (schemaType is string ts2 && (ts2 == "object" || ts2 == "string"));
            if (typeIsNullObjectOrString && schema.TryGetValue("allOf", out var allOfObj)
                && allOfObj is List<object?> allOfList)
            {
                var required = new HashSet<string>(StringComparer.Ordinal);
                var properties = new List<KeyValuePair<string, object?>>();
                var enumValues = new Dictionary<string, int>(StringComparer.Ordinal);

                void AddComponent(object? comp, bool isRequired)
                {
                    if (comp is not OrderedJsonObject co) return;
                    if (co.TryGetValue("$ref", out var cref) && cref is string crefS
                        && _refs.TryGetValue(crefS, out var resolved))
                    {
                        AddComponent(resolved, isRequired);
                    }
                    else if (co.TryGetValue("properties", out var cp) && cp is OrderedJsonObject cpObj)
                    {
                        foreach (var kv in cpObj.Items())
                        {
                            properties.Add(kv);
                            if (isRequired) required.Add(kv.Key);
                        }
                    }
                    else if (co.TryGetValue("enum", out var ce) && ce is List<object?> ceList)
                    {
                        foreach (var v in ceList)
                        {
                            var rule = GenerateConstantRule(v);
                            enumValues.TryGetValue(rule, out var cnt);
                            enumValues[rule] = cnt + 1;
                        }
                    }
                }

                foreach (var component in allOfList)
                {
                    if (component is OrderedJsonObject compObj
                        && compObj.TryGetValue("anyOf", out var ao) && ao is List<object?> aoList)
                    {
                        foreach (var inner in aoList) AddComponent(inner, false);
                    }
                    else
                    {
                        AddComponent(component, true);
                    }
                }

                if (enumValues.Count > 0)
                {
                    var intersection = new List<string>();
                    foreach (var p in enumValues)
                    {
                        if (p.Value == allOfList.Count) intersection.Add(p.Key);
                    }
                    if (intersection.Count > 0)
                    {
                        return AddRule(ruleName, "(" + string.Join(" | ", intersection) + ") space");
                    }
                }
                return AddRule(ruleName, BuildObjectRule(properties, required, name, null));
            }

            // array
            bool typeIsNullOrArray = schemaType is null || (schemaType is string ts3 && ts3 == "array");
            if (typeIsNullOrArray
                && (schema.ContainsKey("items") || schema.ContainsKey("prefixItems")))
            {
                var items = schema.TryGetValue("items", out var it) ? it : schema["prefixItems"];
                if (items is List<object?> itemsList)
                {
                    // Tuple form — fixed sequence.
                    var rule = new StringBuilder("\"[\" space ");
                    for (int i = 0; i < itemsList.Count; i++)
                    {
                        if (i > 0) rule.Append(" \",\" space ");
                        var tupleName = name + (name.Length == 0 ? "" : "-") + "tuple-" + i;
                        rule.Append(Visit(itemsList[i], tupleName));
                    }
                    rule.Append(" \"]\" space");
                    return AddRule(ruleName, rule.ToString());
                }
                var itemRuleName = Visit(items, name + (name.Length == 0 ? "" : "-") + "item");
                int minItems = schema.TryGetValue("minItems", out var minI) && minI is long ml ? (int)ml : 0;
                int maxItems = int.MaxValue;
                if (schema.TryGetValue("maxItems", out var maxI) && maxI is long mxl) maxItems = (int)mxl;
                return AddRule(ruleName,
                    "\"[\" space " + BuildRepetition(itemRuleName, minItems, maxItems, "\",\" space") + " \"]\" space");
            }

            // string with pattern
            bool typeIsNullOrString = schemaType is null || (schemaType is string ts4 && ts4 == "string");
            if (typeIsNullOrString && schema.TryGetValue("pattern", out var pat) && pat is string patStr)
            {
                return VisitPattern(patStr, ruleName);
            }

            // string with uuid-family format
            if (typeIsNullOrString && Regex.IsMatch(schemaFormat, "^uuid[1-5]?$"))
            {
                return AddPrimitive(ruleName == "root" ? "root" : schemaFormat, PrimitiveRules["uuid"]);
            }

            // string with date/time/date-time format
            if (typeIsNullOrString && StringFormatRules.ContainsKey(schemaFormat + "-string"))
            {
                var primName = schemaFormat + "-string";
                return AddRule(ruleName, AddPrimitive(primName, StringFormatRules[primName]));
            }

            // string with minLength / maxLength
            if (schemaType is string ts5 && ts5 == "string"
                && (schema.ContainsKey("minLength") || schema.ContainsKey("maxLength")))
            {
                var charRule = AddPrimitive("char", PrimitiveRules["char"]);
                int minLen = schema.TryGetValue("minLength", out var ml1) && ml1 is long mlv ? (int)mlv : 0;
                int maxLen = int.MaxValue;
                if (schema.TryGetValue("maxLength", out var mx1) && mx1 is long mxv) maxLen = (int)mxv;
                return AddRule(ruleName,
                    "\"\\\"\" " + BuildRepetition(charRule, minLen, maxLen) + " \"\\\"\" space");
            }

            // integer with min/max
            if (schemaType is string ts6 && ts6 == "integer"
                && (schema.ContainsKey("minimum") || schema.ContainsKey("exclusiveMinimum")
                    || schema.ContainsKey("maximum") || schema.ContainsKey("exclusiveMaximum")))
            {
                long minValue = long.MinValue;
                long maxValue = long.MaxValue;
                if (schema.TryGetValue("minimum", out var minO) && minO is long minL) minValue = minL;
                else if (schema.TryGetValue("exclusiveMinimum", out var exMinO) && exMinO is long exMinL)
                    minValue = exMinL + 1;
                if (schema.TryGetValue("maximum", out var maxO) && maxO is long maxL) maxValue = maxL;
                else if (schema.TryGetValue("exclusiveMaximum", out var exMaxO) && exMaxO is long exMaxL)
                    maxValue = exMaxL - 1;
                var sb = new StringBuilder("(");
                BuildMinMaxInt(minValue, maxValue, sb);
                sb.Append(") space");
                return AddRule(ruleName, sb.ToString());
            }

            // empty schema / bare object
            if (schema.Count == 0 || (schemaType is string ts7 && ts7 == "object"))
            {
                return AddRule(ruleName, AddPrimitive("object", PrimitiveRules["object"]));
            }

            // No type constraint and no recognized structural keywords —
            // {"description": "..."} etc. — accept any JSON value.
            if (schemaType is null)
            {
                return AddRule(ruleName, AddPrimitive("value", PrimitiveRules["value"]));
            }

            // Primitive type match
            if (schemaType is string tsPrim && PrimitiveRules.TryGetValue(tsPrim, out var primRule))
            {
                return AddPrimitive(ruleName == "root" ? "root" : tsPrim, primRule);
            }

            _errors.Add("Unrecognized schema: " + JsonDump(schema));
            return string.Empty;
        }

        // ========================================================
        // _build_object_rule — properties with required/optional/
        // additional, including the optional-recursive-refs lambda.
        // ========================================================

        private string BuildObjectRule(
            IReadOnlyList<KeyValuePair<string, object?>> properties,
            HashSet<string> required,
            string name,
            object? additionalProperties)
        {
            var requiredProps = new List<string>();
            var optionalProps = new List<string>();
            var propKvRuleNames = new Dictionary<string, string>(StringComparer.Ordinal);
            var propNames = new List<string>();

            foreach (var kv in properties)
            {
                var propName = kv.Key;
                var propSchema = kv.Value;
                var propRuleName = Visit(propSchema, name + (name.Length == 0 ? "" : "-") + propName);
                var kvRuleName = AddRule(
                    name + (name.Length == 0 ? "" : "-") + propName + "-kv",
                    FormatLiteral(JsonDump(propName)) + " space \":\" space " + propRuleName);
                propKvRuleNames[propName] = kvRuleName;
                if (required.Contains(propName)) requiredProps.Add(propName);
                else optionalProps.Add(propName);
                propNames.Add(propName);
            }

            // additionalProperties: true or a sub-schema
            bool addAllowed =
                (additionalProperties is bool apBool && apBool)
                || additionalProperties is OrderedJsonObject;
            if (addAllowed)
            {
                var subName = name + (name.Length == 0 ? "" : "-") + "additional";
                var valueRule = additionalProperties is OrderedJsonObject apSchema
                    ? Visit(apSchema, subName + "-value")
                    : AddPrimitive("value", PrimitiveRules["value"]);
                var keyRule = propNames.Count == 0
                    ? AddPrimitive("string", PrimitiveRules["string"])
                    : AddRule(subName + "-k", NotStrings(propNames));
                var kvRule = AddRule(subName + "-kv", keyRule + " \":\" space " + valueRule);
                propKvRuleNames["*"] = kvRule;
                optionalProps.Add("*");
            }

            var sb = new StringBuilder("\"{\" space ");
            for (int i = 0; i < requiredProps.Count; i++)
            {
                if (i > 0) sb.Append(" \",\" space ");
                sb.Append(propKvRuleNames[requiredProps[i]]);
            }

            if (optionalProps.Count > 0)
            {
                sb.Append(" (");
                if (requiredProps.Count > 0) sb.Append(" \",\" space ( ");

                // Mirror the C++ recursive "optional trailing refs" lambda.
                string GetRecursiveRefs(IReadOnlyList<string> ks, bool firstIsOptional)
                {
                    if (ks.Count == 0) return string.Empty;
                    var k = ks[0];
                    var kvRuleName = propKvRuleNames[k];
                    var commaRef = "( \",\" space " + kvRuleName + " )";
                    string res = firstIsOptional
                        ? commaRef + (k == "*" ? "*" : "?")
                        : kvRuleName + (k == "*" ? " " + commaRef + "*" : string.Empty);
                    if (ks.Count > 1)
                    {
                        res += " " + AddRule(
                            name + (name.Length == 0 ? "" : "-") + k + "-rest",
                            GetRecursiveRefs(Slice(ks, 1), true));
                    }
                    return res;
                }

                for (int i = 0; i < optionalProps.Count; i++)
                {
                    if (i > 0) sb.Append(" | ");
                    sb.Append(GetRecursiveRefs(Slice(optionalProps, i), false));
                }
                if (requiredProps.Count > 0) sb.Append(" )");
                sb.Append(" )?");
            }

            sb.Append(" \"}\" space");
            return sb.ToString();
        }

        private static IReadOnlyList<string> Slice(IReadOnlyList<string> list, int from)
        {
            var r = new List<string>(list.Count - from);
            for (int i = from; i < list.Count; i++) r.Add(list[i]);
            return r;
        }

        // ========================================================
        // _not_strings — trie-based "reject these exact strings" rule.
        // Used when additionalProperties has a declared key-exclusion set.
        // ========================================================

        private string NotStrings(IReadOnlyList<string> strings)
        {
            var trie = new TrieNode();
            foreach (var s in strings) trie.Insert(s);
            var charRule = AddPrimitive("char", PrimitiveRules["char"]);
            var sb = new StringBuilder("[\"] ( ");
            BuildNotStrings(trie, charRule, sb);
            sb.Append(" )");
            if (!trie.IsEndOfString) sb.Append("?");
            sb.Append(" [\"] space");
            return sb.ToString();
        }

        private static void BuildNotStrings(TrieNode node, string charRule, StringBuilder sb)
        {
            var rejects = new StringBuilder();
            bool first = true;
            foreach (var kv in node.Children)
            {
                rejects.Append(kv.Key);
                if (first) first = false;
                else sb.Append(" | ");
                sb.Append('[').Append(kv.Key).Append(']');
                if (kv.Value.Children.Count > 0)
                {
                    sb.Append(" (");
                    BuildNotStrings(kv.Value, charRule, sb);
                    sb.Append(")");
                }
                else if (kv.Value.IsEndOfString)
                {
                    sb.Append(' ').Append(charRule).Append('+');
                }
            }
            if (node.Children.Count > 0)
            {
                if (!first) sb.Append(" | ");
                sb.Append("[^\"").Append(rejects).Append("] ").Append(charRule).Append('*');
            }
        }

        private sealed class TrieNode
        {
            public SortedDictionary<char, TrieNode> Children { get; } = new();
            public bool IsEndOfString { get; set; }

            public void Insert(string s)
            {
                var node = this;
                foreach (var c in s)
                {
                    if (!node.Children.TryGetValue(c, out var child))
                    {
                        child = new TrieNode();
                        node.Children[c] = child;
                    }
                    node = child;
                }
                node.IsEndOfString = true;
            }
        }

        // ========================================================
        // _visit_pattern — regex → GBNF compiler.
        // Matches only anchored patterns (^...$).
        // ========================================================

        private string VisitPattern(string pattern, string name)
        {
            if (!(pattern.Length >= 2 && pattern[0] == '^' && pattern[^1] == '$'))
            {
                _errors.Add("Pattern must start with '^' and end with '$'");
                return string.Empty;
            }
            var subPattern = pattern.Substring(1, pattern.Length - 2);
            var subRuleIds = new Dictionary<string, string>(StringComparer.Ordinal);
            int i = 0;
            int length = subPattern.Length;

            string Transform()
            {
                int start = i;
                var seq = new List<(string Text, bool IsLiteral)>();

                string GetDot() => AddRule("dot",
                    _dotall ? "[\\U00000000-\\U0010FFFF]" : "[^\\x0A\\x0D]");

                string ToRule((string Text, bool IsLiteral) ls)
                    => ls.IsLiteral ? "\"" + ls.Text + "\"" : ls.Text;

                string JoinSeq()
                {
                    var ret = new List<(string Text, bool IsLiteral)>();
                    var literal = new StringBuilder();
                    void FlushLiteral()
                    {
                        if (literal.Length == 0) return;
                        ret.Add((literal.ToString(), true));
                        literal.Clear();
                    }
                    foreach (var item in seq)
                    {
                        if (item.IsLiteral) literal.Append(item.Text);
                        else { FlushLiteral(); ret.Add(item); }
                    }
                    FlushLiteral();
                    var results = new List<string>(ret.Count);
                    foreach (var item in ret) results.Add(ToRule(item));
                    return string.Join(" ", results);
                }

                while (i < length)
                {
                    char c = subPattern[i];
                    if (c == '.')
                    {
                        seq.Add((GetDot(), false));
                        i++;
                    }
                    else if (c == '(')
                    {
                        i++;
                        if (i < length && subPattern[i] == '?')
                        {
                            if (i + 1 < length && subPattern[i + 1] == ':')
                            {
                                i += 2; // non-capturing group — treat as normal
                            }
                            else
                            {
                                // lookaround not supported — skip to matching ')'
                                _warnings.Add("Unsupported pattern syntax");
                                int depth = 1;
                                while (i < length && depth > 0)
                                {
                                    if (subPattern[i] == '\\' && i + 1 < length)
                                    {
                                        i += 2;
                                    }
                                    else
                                    {
                                        if (subPattern[i] == '(') depth++;
                                        else if (subPattern[i] == ')') depth--;
                                        i++;
                                    }
                                }
                                continue;
                            }
                        }
                        var inner = Transform();
                        seq.Add(("(" + inner + ")", false));
                    }
                    else if (c == ')')
                    {
                        i++;
                        if (start > 0 && subPattern[start - 1] != '(' &&
                            (start < 2 || subPattern[start - 2] != '?' || subPattern[start - 1] != ':'))
                        {
                            _errors.Add("Unbalanced parentheses");
                        }
                        return JoinSeq();
                    }
                    else if (c == '[')
                    {
                        var sb2 = new StringBuilder();
                        sb2.Append(c);
                        i++;
                        while (i < length && subPattern[i] != ']')
                        {
                            if (subPattern[i] == '\\')
                            {
                                if (i + 1 < length) sb2.Append(subPattern, i, 2);
                                else sb2.Append(subPattern[i]);
                                i += 2;
                            }
                            else
                            {
                                sb2.Append(subPattern[i]);
                                i++;
                            }
                        }
                        if (i >= length) _errors.Add("Unbalanced square brackets");
                        sb2.Append(']');
                        i++;
                        seq.Add((sb2.ToString(), false));
                    }
                    else if (c == '|')
                    {
                        seq.Add(("|", false));
                        i++;
                    }
                    else if (c == '*' || c == '+' || c == '?')
                    {
                        if (seq.Count == 0) { i++; continue; }
                        var last = seq[^1];
                        seq[^1] = (ToRule(last) + c, false);
                        i++;
                    }
                    else if (c == '{')
                    {
                        var cb = new StringBuilder();
                        cb.Append(c);
                        i++;
                        while (i < length && subPattern[i] != '}')
                        {
                            cb.Append(subPattern[i]);
                            i++;
                        }
                        if (i >= length) _errors.Add("Unbalanced curly brackets");
                        cb.Append('}');
                        i++;
                        var nums = cb.ToString(1, cb.Length - 2).Split(',');
                        int minTimes = 0;
                        int maxTimes = int.MaxValue;
                        try
                        {
                            if (nums.Length == 1)
                            {
                                minTimes = maxTimes = int.Parse(nums[0]);
                            }
                            else if (nums.Length != 2)
                            {
                                _errors.Add("Wrong number of values in curly brackets");
                            }
                            else
                            {
                                if (nums[0].Length > 0) minTimes = int.Parse(nums[0]);
                                if (nums[1].Length > 0) maxTimes = int.Parse(nums[1]);
                            }
                        }
                        catch (FormatException)
                        {
                            _errors.Add("Invalid number in curly brackets");
                            return string.Empty;
                        }
                        if (seq.Count == 0) continue;
                        var last2 = seq[^1];
                        var sub = last2.Text;
                        var subIsLiteral = last2.IsLiteral;
                        if (!subIsLiteral)
                        {
                            if (!subRuleIds.TryGetValue(sub, out var subId))
                            {
                                subId = AddRule(name + "-" + subRuleIds.Count, sub);
                                subRuleIds[sub] = subId;
                            }
                            sub = subId;
                        }
                        seq[^1] = (BuildRepetition(
                            subIsLiteral ? "\"" + sub + "\"" : sub,
                            minTimes, maxTimes), false);
                    }
                    else
                    {
                        var literal = new StringBuilder();
                        bool IsNonLiteral(char ch) => NonLiteralSet.Contains(ch);
                        while (i < length)
                        {
                            if (subPattern[i] == '\\' && i < length - 1)
                            {
                                var next = subPattern[i + 1];
                                if (EscapedInRegexpsButNotInLiterals.Contains(next))
                                {
                                    i++;
                                    literal.Append(subPattern[i]);
                                    i++;
                                }
                                else
                                {
                                    literal.Append(subPattern, i, 2);
                                    i += 2;
                                }
                            }
                            else if (subPattern[i] == '"')
                            {
                                literal.Append("\\\"");
                                i++;
                            }
                            else if (!IsNonLiteral(subPattern[i]) &&
                                (i == length - 1
                                 || literal.Length == 0
                                 || subPattern[i + 1] == '.'
                                 || !IsNonLiteral(subPattern[i + 1])))
                            {
                                literal.Append(subPattern[i]);
                                i++;
                            }
                            else break;
                        }
                        if (literal.Length > 0) seq.Add((literal.ToString(), true));
                    }
                }
                return JoinSeq();
            }

            var body = Transform();
            return AddRule(name, "\"\\\"\" (" + body + ") \"\\\"\" space");
        }

        // ========================================================
        // build_min_max_int — translate an integer range into a GBNF
        // expression. The recursive uniform_range lambda + digit-range
        // helpers are ported directly. Decimals_left budgets the
        // recursion so unconstrained-max arms don't run forever.
        // ========================================================

        private static void BuildMinMaxInt(long minValue, long maxValue, StringBuilder sb,
                                           int decimalsLeft = 16, bool topLevel = true)
        {
            bool hasMin = minValue != long.MinValue;
            bool hasMax = maxValue != long.MaxValue;

            void DigitRange(char from, char to)
            {
                sb.Append('[');
                if (from == to) sb.Append(from);
                else { sb.Append(from).Append('-').Append(to); }
                sb.Append(']');
            }

            void MoreDigits(int minDigits, int maxDigits)
            {
                sb.Append("[0-9]");
                if (minDigits == maxDigits && minDigits == 1) return;
                sb.Append('{').Append(minDigits);
                if (maxDigits != minDigits)
                {
                    sb.Append(',');
                    if (maxDigits != int.MaxValue) sb.Append(maxDigits);
                }
                sb.Append('}');
            }

            void UniformRange(string from, string to)
            {
                int i2 = 0;
                while (i2 < from.Length && i2 < to.Length && from[i2] == to[i2]) i2++;
                if (i2 > 0) sb.Append('"').Append(from.AsSpan(0, i2)).Append('"');
                if (i2 < from.Length && i2 < to.Length)
                {
                    if (i2 > 0) sb.Append(' ');
                    int subLen = from.Length - i2 - 1;
                    if (subLen > 0)
                    {
                        var fromSub = from.Substring(i2 + 1);
                        var toSub = to.Substring(i2 + 1);
                        var subZeros = new string('0', subLen);
                        var subNines = new string('9', subLen);
                        bool toReached = false;
                        sb.Append('(');
                        if (fromSub == subZeros)
                        {
                            DigitRange(from[i2], (char)(to[i2] - 1));
                            sb.Append(' ');
                            MoreDigits(subLen, subLen);
                        }
                        else
                        {
                            sb.Append('[').Append(from[i2]).Append("] ");
                            sb.Append('(');
                            UniformRange(fromSub, subNines);
                            sb.Append(')');
                            if (from[i2] < to[i2] - 1)
                            {
                                sb.Append(" | ");
                                if (toSub == subNines)
                                {
                                    DigitRange((char)(from[i2] + 1), to[i2]);
                                    toReached = true;
                                }
                                else
                                {
                                    DigitRange((char)(from[i2] + 1), (char)(to[i2] - 1));
                                }
                                sb.Append(' ');
                                MoreDigits(subLen, subLen);
                            }
                        }
                        if (!toReached)
                        {
                            sb.Append(" | ");
                            DigitRange(to[i2], to[i2]);
                            sb.Append(' ');
                            UniformRange(subZeros, toSub);
                        }
                        sb.Append(')');
                    }
                    else
                    {
                        sb.Append('[').Append(from[i2]).Append('-').Append(to[i2]).Append(']');
                    }
                }
            }

            if (hasMin && hasMax)
            {
                if (minValue < 0 && maxValue < 0)
                {
                    sb.Append("\"-\" (");
                    BuildMinMaxInt(-maxValue, -minValue, sb, decimalsLeft, true);
                    sb.Append(")");
                    return;
                }
                if (minValue < 0)
                {
                    sb.Append("\"-\" (");
                    BuildMinMaxInt(0, -minValue, sb, decimalsLeft, true);
                    sb.Append(") | ");
                    minValue = 0;
                }
                var minS = minValue.ToString();
                var maxS = maxValue.ToString();
                int minDigits = minS.Length;
                int maxDigits = maxS.Length;
                for (int digits = minDigits; digits < maxDigits; digits++)
                {
                    UniformRange(minS, new string('9', digits));
                    minS = "1" + new string('0', digits);
                    sb.Append(" | ");
                }
                UniformRange(minS, maxS);
                return;
            }

            int lessDecimals = Math.Max(decimalsLeft - 1, 1);

            if (hasMin)
            {
                if (minValue < 0)
                {
                    sb.Append("\"-\" (");
                    BuildMinMaxInt(long.MinValue, -minValue, sb, decimalsLeft, false);
                    sb.Append(") | [0] | [1-9] ");
                    MoreDigits(0, decimalsLeft - 1);
                }
                else if (minValue == 0)
                {
                    if (topLevel) { sb.Append("[0] | [1-9] "); MoreDigits(0, lessDecimals); }
                    else MoreDigits(1, decimalsLeft);
                }
                else if (minValue <= 9)
                {
                    char c = (char)('0' + minValue);
                    char rangeStart = topLevel ? '1' : '0';
                    if (c > rangeStart)
                    {
                        DigitRange(rangeStart, (char)(c - 1));
                        sb.Append(' ');
                        MoreDigits(1, lessDecimals);
                        sb.Append(" | ");
                    }
                    DigitRange(c, '9');
                    sb.Append(' ');
                    MoreDigits(0, lessDecimals);
                }
                else
                {
                    var minS = minValue.ToString();
                    int len = minS.Length;
                    char c = minS[0];
                    if (c > '1')
                    {
                        DigitRange(topLevel ? '1' : '0', (char)(c - 1));
                        sb.Append(' ');
                        MoreDigits(len, lessDecimals);
                        sb.Append(" | ");
                    }
                    DigitRange(c, c);
                    sb.Append(" (");
                    BuildMinMaxInt(long.Parse(minS.Substring(1)), long.MaxValue, sb, lessDecimals, false);
                    sb.Append(")");
                    if (c < '9')
                    {
                        sb.Append(" | ");
                        DigitRange((char)(c + 1), '9');
                        sb.Append(' ');
                        MoreDigits(len - 1, lessDecimals);
                    }
                }
                return;
            }

            if (hasMax)
            {
                if (maxValue >= 0)
                {
                    if (topLevel)
                    {
                        sb.Append("\"-\" [1-9] ");
                        MoreDigits(0, lessDecimals);
                        sb.Append(" | ");
                    }
                    BuildMinMaxInt(0, maxValue, sb, decimalsLeft, true);
                }
                else
                {
                    sb.Append("\"-\" (");
                    BuildMinMaxInt(-maxValue, long.MaxValue, sb, decimalsLeft, false);
                    sb.Append(")");
                }
                return;
            }

            throw new ArgumentException("At least one of minValue or maxValue must be set.");
        }

        /// <summary>
        /// Serialise a <see cref="object"/>-tree value to a JSON string.
        /// Matches nlohmann's <c>dump()</c> for the atoms we generate —
        /// strings get double-quoted + escaped, numbers render literal.
        /// </summary>
        internal static string JsonDump(object? value)
        {
            switch (value)
            {
                case null: return "null";
                case bool b: return b ? "true" : "false";
                case string s: return System.Text.Json.JsonSerializer.Serialize(s);
                case long l: return l.ToString(System.Globalization.CultureInfo.InvariantCulture);
                case int i: return i.ToString(System.Globalization.CultureInfo.InvariantCulture);
                case double d: return d.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
                case List<object?> arr:
                    {
                        var sb2 = new StringBuilder("[");
                        for (int k = 0; k < arr.Count; k++)
                        {
                            if (k > 0) sb2.Append(',');
                            sb2.Append(JsonDump(arr[k]));
                        }
                        sb2.Append(']');
                        return sb2.ToString();
                    }
                case OrderedJsonObject obj:
                    {
                        var sb2 = new StringBuilder("{");
                        bool first = true;
                        foreach (var kv in obj.Items())
                        {
                            if (!first) sb2.Append(',');
                            first = false;
                            sb2.Append(System.Text.Json.JsonSerializer.Serialize(kv.Key));
                            sb2.Append(':');
                            sb2.Append(JsonDump(kv.Value));
                        }
                        sb2.Append('}');
                        return sb2.ToString();
                    }
                default: return System.Text.Json.JsonSerializer.Serialize(value);
            }
        }
    }
}
