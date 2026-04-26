#!/usr/bin/env python3
"""
Compare a sensitivity-sweep score table against llama.cpp's hard-coded
ftype heuristic, per-tensor.

Usage:
    python3 analyze-sensitivity-vs-heuristic.py \
        --scores  ~/.cache/llama-models/Qwen/Qwen3-0.6B/Qwen3-0.6B.F16.scores.json \
        --ftype   q4_k_m \
        --layers  28

Output:
    1. Heuristic average bpw across all scored tensors.
    2. Mapping of τ → recipe average bpw.
    3. The τ closest to the heuristic's average bpw, used for a
       budget-matched comparison.
    4. Per-category disagreement counts.
    5. Largest bpw-delta disagreements (where the recipe wants more
       bits than the heuristic, and vice versa).

The heuristic implementation here mirrors llama.cpp/src/llama-quant.cpp
(commit pinned in third_party/llama.cpp/VERSION). Not all ftypes are
supported yet — Q4_K_M is the focus because it's the most-shipped
recipe in the wild.
"""

import argparse
import collections
import json
import re
import sys


# Bits-per-element for each candidate type, copied from
# LlamaQuantRecipe.GetBitsPerElement in the bindings.
BPW = {
    'F16': 16.0, 'BF16': 16.0, 'Q8_0': 8.5,
    'Q6_K': 6.5625, 'Q5_K': 5.5, 'Q4_K': 4.5,
    'IQ4_XS': 4.25, 'Q3_K': 3.4375, 'IQ3_S': 3.4375,
    'Q2_K': 2.625, 'IQ2_S': 2.5,
}

# Order in which the recipe walks candidates (smallest bpw first).
CANDIDATE_ORDER = [
    'IQ2_S', 'Q2_K', 'IQ3_S', 'Q3_K', 'IQ4_XS', 'Q4_K',
    'Q5_K', 'Q6_K', 'Q8_0', 'BF16', 'F16',
]


def categorize(name: str) -> str:
    """Mirror tensor_get_category() from llama-quant.cpp."""
    if name == 'output.weight':           return 'OUTPUT'
    if name == 'token_embd.weight':       return 'TOKEN_EMBD'
    for needle in ('attn_v.weight', 'attn_k.weight', 'attn_q.weight',
                   'attn_output.weight', 'ffn_up', 'ffn_gate', 'ffn_down'):
        if needle in name:
            return needle
    return 'OTHER'


def layer_index(name: str) -> int:
    m = re.match(r'blk\.(\d+)\.', name)
    return int(m.group(1)) if m else -1


def use_more_bits(i: int, n: int) -> bool:
    """Mirror the lambda at llama-quant.cpp:417."""
    return i < n // 8 or i >= 7 * n // 8 or (i - n // 8) % 3 == 2


def heuristic_q4_k_m(name: str, n_layers: int) -> str:
    """
    Reproduce llama-quant.cpp's Q4_K_M heuristic for a dense
    (non-MoE), non-Falcon model. Diverges for MoE / Falcon /
    GQA-special-cases — extend if you need those.
    """
    cat = categorize(name)
    i = layer_index(name)
    if cat == 'OUTPUT':       return 'Q6_K'   # line ~458
    if cat == 'TOKEN_EMBD':   return 'Q4_K'   # falls through to ftype default
    if cat == 'attn_v.weight':
        # line ~528 (Q5_K default) bumped to Q6_K by use_more_bits
        return 'Q6_K' if use_more_bits(i, n_layers) else 'Q5_K'
    if cat == 'ffn_down':
        # line ~590-595
        if i < n_layers // 16:
            return 'Q6_K'
        return 'Q5_K' if use_more_bits(i, n_layers) else 'Q4_K'
    # attn_q, attn_k, attn_output, ffn_up, ffn_gate — family default
    return 'Q4_K'


def recipe_pick(scores_for_tensor: dict, tau: float) -> str:
    """Smallest type whose rel-MSE ≤ τ; fallback to lowest-MSE."""
    for t in CANDIDATE_ORDER:
        v = scores_for_tensor.get(t)
        if v is not None and v <= tau:
            return t
    return min(scores_for_tensor.keys(),
               key=lambda k: scores_for_tensor[k] if scores_for_tensor[k] is not None
                                                  else float('inf'))


def avg_bpw(picks) -> float:
    vals = [BPW[p] for p in picks if p in BPW]
    return sum(vals) / len(vals) if vals else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', required=True, help='path to scores.json')
    ap.add_argument('--ftype',  default='q4_k_m', choices=['q4_k_m'],
                    help='heuristic to compare against (only Q4_K_M for now)')
    ap.add_argument('--layers', type=int, required=True,
                    help='n_layers for the model under test')
    ap.add_argument('--top',    type=int, default=15,
                    help='show this many largest +/- bpw-delta disagreements')
    args = ap.parse_args()

    with open(args.scores) as f:
        d = json.load(f)
    scores = d['scores']

    by_tensor = collections.defaultdict(dict)
    for s in scores:
        by_tensor[s['tensorName']][s['quantType']] = s['relativeMse']

    heuristic = {n: heuristic_q4_k_m(n, args.layers) for n in by_tensor}
    h_bpw = avg_bpw(heuristic.values())
    print(f'Heuristic ({args.ftype.upper()}) average bpw: {h_bpw:.3f}')

    print('\nτ → recipe avg bpw:')
    tau_grid = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    tau_avg = []
    for tau in tau_grid:
        picks = [recipe_pick(by_tensor[n], tau) for n in by_tensor]
        b = avg_bpw(picks)
        tau_avg.append((tau, b))
        print(f'  τ={tau:7.5f}  avg_bpw={b:.3f}')

    # Pick τ that's closest to the heuristic's bpw for a fair comparison.
    best_tau = min(tau_grid, key=lambda t:
                   abs(avg_bpw([recipe_pick(by_tensor[n], t) for n in by_tensor]) - h_bpw))
    print(f'\nClosest τ to heuristic ({h_bpw:.3f} bpw): τ={best_tau}')

    recipe = {n: recipe_pick(by_tensor[n], best_tau) for n in by_tensor}
    disagreements = []
    for n in by_tensor:
        h = heuristic[n]; r = recipe[n]
        if h == r: continue
        h_mse = by_tensor[n].get(h, float('nan'))
        r_mse = by_tensor[n].get(r, float('nan'))
        disagreements.append((n, h, BPW.get(h, 0.0), h_mse,
                                 r, BPW.get(r, 0.0), r_mse))

    print(f'\n{len(disagreements)} / {len(by_tensor)} tensors disagree at τ={best_tau}')
    print('\nDisagreement counts by category:')
    cat_count = collections.Counter()
    for row in disagreements:
        cat_count[categorize(row[0])] += 1
    for c, n in cat_count.most_common():
        print(f'  {c:<22} {n}')

    disagreements.sort(key=lambda x: x[5] - x[2])
    print(f'\nLargest negative bpw deltas (recipe wants LESS than heuristic):')
    for row in disagreements[:args.top]:
        n, h, hb, hm, r, rb, rm = row
        print(f'  {n:<32} {h:<8} {hb:>5.2f}  {hm:>10.2e}   →   {r:<8} {rb:>5.2f}  {rm:>10.2e}')

    print(f'\nLargest positive bpw deltas (recipe wants MORE than heuristic):')
    for row in disagreements[-args.top:]:
        n, h, hb, hm, r, rb, rm = row
        print(f'  {n:<32} {h:<8} {hb:>5.2f}  {hm:>10.2e}   →   {r:<8} {rb:>5.2f}  {rm:>10.2e}')


if __name__ == '__main__':
    sys.exit(main())
