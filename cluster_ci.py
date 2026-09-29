#!/usr/bin/env python3
"""Tool-clustered bootstrap CI for a paired difference between two conditions (reviewer point 6).

Resamples TOOLS with replacement (not queries), so the interval reflects uncertainty over the catalog,
not just over queries within it. Seeds are averaged per tool before resampling when several files match.

Usage:
  python cluster_ci.py --ref 'results/openai_gpt_oss_20b__cat64__base_pv1.rows.jsonl' --ref-arm A \
                       --cmp 'results/openai_gpt_oss_20b__cat64__cat64_mix50_C_lr5e-4dec_s*_pv1.rows.jsonl' --cmp-arm C \
                       --set natural
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import statistics as st


def per_tool_means(pattern: str, arm: str, set_name: str, metric: str) -> dict[str, float]:
    matches = sorted(glob.glob(pattern))
    if set_name == "strong_unseen":
        files = [f for f in matches if "_unseen" in f]
        set_name = "strong"
    else:
        # prefer non-_unseen files (the primary runs); a run whose only file is *_unseen (e.g. a retrieval
        # baseline tagged base_unseen) still carries the natural rows, so fall back to it
        files = [f for f in matches if "_unseen" not in f] or matches
    if not files:
        raise SystemExit(f"no files for {pattern}")
    acc = collections.defaultdict(list)          # tool -> [per-file mean]
    for f in files:
        d = collections.defaultdict(lambda: [0, 0])
        for l in open(f):
            r = json.loads(l)
            if r["arm"] == arm and r["set"] == set_name:
                d[r["tool"]][0] += r["score"][metric]; d[r["tool"]][1] += 1
        for t, (a, n) in d.items():
            acc[t].append(a / n)
    return {t: st.mean(v) for t, v in acc.items()}, len(files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True); ap.add_argument("--ref-arm", required=True)
    ap.add_argument("--cmp", required=True); ap.add_argument("--cmp-arm", required=True)
    ap.add_argument("--set", default="natural", choices=["natural", "strong", "strong_unseen"])
    ap.add_argument("--metric", default="args_exact")
    ap.add_argument("--B", type=int, default=10000); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ref, nr = per_tool_means(args.ref, args.ref_arm, args.set, args.metric)
    cmp, nc = per_tool_means(args.cmp, args.cmp_arm, args.set, args.metric)
    tools = sorted(set(ref) & set(cmp))
    diffs = [cmp[t] - ref[t] for t in tools]
    rng = random.Random(args.seed)
    boots = []
    for _ in range(args.B):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boots.append(st.mean(sample))
    boots.sort()
    lo, hi = boots[int(0.025 * args.B)], boots[int(0.975 * args.B)]
    print(f"set={args.set} metric={args.metric}  tools={len(tools)}  ref files={nr} cmp files={nc}")
    print(f"ref mean={st.mean(ref[t] for t in tools):.3f}  cmp mean={st.mean(cmp[t] for t in tools):.3f}")
    print(f"paired Δ (cmp − ref) = {st.mean(diffs):+.5f}   95% tool-clustered bootstrap CI [{lo:+.5f}, {hi:+.5f}]   "
          f"tools with Δ>0: {sum(d > 0 for d in diffs)}/{len(tools)}   (bootstrap seed {args.seed}, B={args.B})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())