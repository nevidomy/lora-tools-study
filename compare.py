#!/usr/bin/env python3
"""Paired comparison of two eval runs on the same held-out queries.

Joins rows on (set, query_id). For each set reports:
  - exact and invention rate for reference and comparison
    (invention = parsed & right tool & NOT schema_valid: the model emitted a value that
     is not in the enum / a missing required / an extra key — i.e. it made something up)
  - comparison exact CONDITIONAL on reference-correct rows (isolates "lacks the mapping"
    from "the paraphrase is a riddle nobody decodes")
  - the 2x2 table: both right / ref only / cmp only / neither, with McNemar's exact test

Usage:
  python compare.py --ref results/...base_pv1.rows.jsonl --ref-arm A \\
                    --cmp results/...cat4_ex100_C_s0_pv1.rows.jsonl --cmp-arm C
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict


def load(path: str, arm: str) -> dict[tuple[str, str], dict]:
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r["arm"] == arm:
            out[(r["set"], r["query_id"])] = r
    return out


def invented(r: dict) -> bool:
    sc = r["score"]
    return bool(sc["parsed"] and sc["tool_ok"] and not sc["schema_valid"])


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar on the discordant cells."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True); ap.add_argument("--ref-arm", required=True)
    ap.add_argument("--cmp", required=True); ap.add_argument("--cmp-arm", required=True)
    ap.add_argument("--rewritten-only", action="store_true", help="strong set: only rows whose text differs from natural")
    args = ap.parse_args()

    R, C = load(args.ref, args.ref_arm), load(args.cmp, args.cmp_arm)
    keys = sorted(set(R) & set(C))
    by_set = defaultdict(list)
    for k in keys:
        if args.rewritten_only and k[0] == "strong" and not C[k].get("rewritten"):
            continue
        by_set[k[0]].append(k)

    print(f"ref: {args.ref_arm}  {args.ref}\ncmp: {args.cmp_arm}  {args.cmp}\n")
    for set_name, ks in by_set.items():
        n = len(ks)
        r_ok = [R[k]["score"]["args_exact"] for k in ks]
        c_ok = [C[k]["score"]["args_exact"] for k in ks]
        r_inv = [invented(R[k]) for k in ks]
        c_inv = [invented(C[k]) for k in ks]
        both = sum(a and b for a, b in zip(r_ok, c_ok))
        ref_only = sum(a and not b for a, b in zip(r_ok, c_ok))
        cmp_only = sum(b and not a for a, b in zip(r_ok, c_ok))
        neither = n - both - ref_only - cmp_only
        n_ref_ok = sum(r_ok)
        cmp_given_ref_ok = both / n_ref_ok if n_ref_ok else float("nan")
        cmp_given_ref_bad = cmp_only / (n - n_ref_ok) if n - n_ref_ok else float("nan")
        r_tok = sum(R[k].get("prompt_tokens", 0) for k in ks) / n
        c_tok = sum(C[k].get("prompt_tokens", 0) for k in ks) / n
        r_out = sum(R[k].get("completion_tokens", 0) or 0 for k in ks) / n
        c_out = sum(C[k].get("completion_tokens", 0) or 0 for k in ks) / n

        print(f"== {set_name}  (n={n} paired{', rewritten-only' if args.rewritten_only and set_name == 'strong' else ''}) ==")
        print(f"  exact        ref={sum(r_ok)/n:.3f}   cmp={sum(c_ok)/n:.3f}   Δ={ (sum(c_ok)-sum(r_ok))/n:+.3f}")
        print(f"  invention    ref={sum(r_inv)/n:.3f}   cmp={sum(c_inv)/n:.3f}   Δ={ (sum(c_inv)-sum(r_inv))/n:+.3f}")
        print(f"  cmp exact | ref correct  = {cmp_given_ref_ok:.3f}   (n={n_ref_ok})")
        print(f"  cmp exact | ref wrong    = {cmp_given_ref_bad:.3f}   (n={n - n_ref_ok})")
        print(f"  2x2: both={both}  ref_only={ref_only}  cmp_only={cmp_only}  neither={neither}   "
              f"McNemar p={mcnemar_p(ref_only, cmp_only):.3g}")
        print(f"  tokens/req   prompt ref={r_tok:.0f} cmp={c_tok:.0f}   completion ref={r_out:.0f} cmp={c_out:.0f}")
        # per-tool conditional
        tools = defaultdict(lambda: [0, 0, 0])  # n_ref_ok, n_cmp_ok|ref_ok, n_cmp_invented
        for k in ks:
            t = tools[R[k]["tool"]]
            if R[k]["score"]["args_exact"]:
                t[0] += 1
                t[1] += int(C[k]["score"]["args_exact"])
            t[2] += int(invented(C[k]))
        print("  per tool (cmp exact | ref correct ; cmp invention):")
        for t, (nr, nc, ninv) in sorted(tools.items()):
            print(f"    {t:32s} {nc}/{nr} = {nc/nr if nr else float('nan'):.2f}   invent={ninv}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())