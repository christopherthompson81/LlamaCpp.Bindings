namespace LlamaCpp.Bindings.Tests;

/// <summary>
/// Smoke tests for <see cref="JsonSchemaToGbnf"/> — the pure-C# port of
/// llama.cpp's json-schema-to-grammar. These don't need a loaded model
/// (the converter has no native dependencies), so they run on every
/// build regardless of the test fixture's GGUF availability.
/// </summary>
public class JsonSchemaToGbnfTests
{
    [Fact]
    public void Empty_Schema_Compiles_To_Any_Value()
    {
        // {} matches any JSON value — the spec's "true" schema equivalent.
        var g = JsonSchemaToGbnf.Convert("{}");
        Assert.Contains("root ::= ", g);
        Assert.Contains("value", g);
    }

    [Fact]
    public void Primitive_Types_Use_Builtin_Rules()
    {
        foreach (var type in new[] { "string", "number", "integer", "boolean", "null" })
        {
            var g = JsonSchemaToGbnf.Convert($"{{\"type\":\"{type}\"}}");
            Assert.Contains("root ::= ", g);
        }
    }

    [Fact]
    public void Object_With_Required_And_Optional_Properties()
    {
        var schema = @"{
            ""type"": ""object"",
            ""properties"": {
                ""name"": { ""type"": ""string"" },
                ""age"":  { ""type"": ""integer"" }
            },
            ""required"": [""name""]
        }";
        var g = JsonSchemaToGbnf.Convert(schema);
        // The root rule wraps the two property kv rules in the required-
        // then-optional pattern. Braces + separators should show up.
        Assert.Contains("root ::=", g);
        Assert.Contains("\"{\"", g);
        Assert.Contains("\"}\"", g);
        Assert.Contains("name-kv", g);
        Assert.Contains("age-kv", g);
    }

    [Fact]
    public void Enum_Generates_Alternation()
    {
        var schema = @"{""enum"": [""red"", ""green"", ""blue""]}";
        var g = JsonSchemaToGbnf.Convert(schema);
        Assert.Contains("root ::=", g);
        Assert.Contains("\"\\\"red\\\"\"",   g);
        Assert.Contains("\"\\\"green\\\"\"", g);
        Assert.Contains("\"\\\"blue\\\"\"",  g);
        Assert.Contains(" | ", g);
    }

    [Fact]
    public void Const_Generates_Single_Literal()
    {
        var schema = @"{""const"": ""the-only-value""}";
        var g = JsonSchemaToGbnf.Convert(schema);
        Assert.Contains("\"\\\"the-only-value\\\"\"", g);
    }

    [Fact]
    public void Array_With_Item_Schema_And_Size_Bounds()
    {
        var schema = @"{
            ""type"": ""array"",
            ""items"": { ""type"": ""integer"" },
            ""minItems"": 1,
            ""maxItems"": 3
        }";
        var g = JsonSchemaToGbnf.Convert(schema);
        Assert.Contains("\"[\"", g);
        Assert.Contains("\"]\"", g);
        Assert.Contains("\",\" space", g);
    }

    [Fact]
    public void Tuple_Array_With_PrefixItems()
    {
        // Primitive sub-schemas in a tuple don't get a named per-site
        // rule — visit() returns the primitive rule ("string", "integer")
        // directly. The tuple-N names only materialise when the sub-schema
        // is complex (object/array/enum). Mirrors the C++ reference.
        var schema = @"{
            ""type"": ""array"",
            ""prefixItems"": [
                { ""type"": ""string"" },
                { ""type"": ""integer"" }
            ]
        }";
        var g = JsonSchemaToGbnf.Convert(schema);
        // Root should reference both primitives with the tuple brackets/sep.
        Assert.Contains("root ::=", g);
        Assert.Contains("string", g);
        Assert.Contains("integer", g);
        Assert.Contains("\"[\"", g);
        Assert.Contains("\",\" space", g);
        Assert.Contains("\"]\"", g);
    }

    [Fact]
    public void Tuple_Array_With_Complex_PrefixItems_Names_Tuple_Slots()
    {
        // When the sub-schema has enough structure to need its own rule,
        // the tuple-N name is how the converter disambiguates it.
        var schema = @"{
            ""type"": ""array"",
            ""prefixItems"": [
                { ""enum"": [""a"", ""b""] },
                { ""type"": ""object"", ""properties"": { ""x"": { ""type"": ""number"" } } }
            ]
        }";
        var g = JsonSchemaToGbnf.Convert(schema);
        Assert.Contains("tuple-0", g);
        Assert.Contains("tuple-1", g);
    }

    [Fact]
    public void Integer_Range_Min_Max()
    {
        var schema = @"{""type"": ""integer"", ""minimum"": 0, ""maximum"": 99}";
        var g = JsonSchemaToGbnf.Convert(schema);
        // Expected to produce digit-range alternation, not the bare
        // integer primitive.
        Assert.DoesNotContain("integer ::= ", g);
        Assert.Contains("[0-9]", g);
    }

    [Fact]
    public void OneOf_Unions_Alternatives()
    {
        var schema = @"{
            ""oneOf"": [
                { ""type"": ""string"" },
                { ""type"": ""integer"" }
            ]
        }";
        var g = JsonSchemaToGbnf.Convert(schema);
        Assert.Contains("root ::=", g);
        Assert.Contains(" | ", g);
    }

    [Fact]
    public void LocalRef_Resolves_To_Named_Rule()
    {
        var schema = @"{
            ""$defs"": {
                ""tag"": { ""type"": ""string"" }
            },
            ""type"": ""object"",
            ""properties"": {
                ""first"":  { ""$ref"": ""#/$defs/tag"" },
                ""second"": { ""$ref"": ""#/$defs/tag"" }
            },
            ""required"": [""first"", ""second""]
        }";
        var g = JsonSchemaToGbnf.Convert(schema);
        // Both properties should resolve through the same ref chain.
        Assert.Contains("first-kv", g);
        Assert.Contains("second-kv", g);
    }

    [Fact]
    public void Pattern_Requires_Start_And_End_Anchors()
    {
        var bad = @"{""type"":""string"",""pattern"":""abc""}";
        var ex = Assert.Throws<JsonSchemaConversionException>(
            () => JsonSchemaToGbnf.Convert(bad));
        Assert.Contains("^", ex.Message);
    }

    [Fact]
    public void Anchored_Pattern_Compiles()
    {
        var schema = @"{""type"":""string"",""pattern"":""^[a-z]+$""}";
        var g = JsonSchemaToGbnf.Convert(schema);
        Assert.Contains("root ::= ", g);
        Assert.Contains("[a-z]", g);
    }

    [Fact]
    public void String_Date_Format_Uses_Builtin_Rule()
    {
        var schema = @"{""type"":""string"",""format"":""date""}";
        var g = JsonSchemaToGbnf.Convert(schema);
        // date-string ::= "\"" date "\"" space  (from the format rules)
        Assert.Contains("date-string", g);
    }

    [Fact]
    public void AdditionalProperties_False_By_Default_For_Known_Schema()
    {
        // With properties listed but no additionalProperties, we still
        // generate a rule; the strictness lives in the "*" wildcard
        // branch only being emitted when additionalProperties is true
        // or a schema.
        var schema = @"{
            ""type"": ""object"",
            ""properties"": { ""x"": { ""type"": ""number"" } }
        }";
        var g = JsonSchemaToGbnf.Convert(schema);
        Assert.DoesNotContain("additional-k", g);
        Assert.DoesNotContain("additional-kv", g);
    }

    [Fact]
    public void AdditionalProperties_True_Allows_Open_Keys()
    {
        var schema = @"{
            ""type"": ""object"",
            ""properties"": { ""x"": { ""type"": ""number"" } },
            ""additionalProperties"": true
        }";
        var g = JsonSchemaToGbnf.Convert(schema);
        Assert.Contains("additional-kv", g);
    }
}
