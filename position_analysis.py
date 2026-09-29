#!/usr/bin/env python3
"""Does the full-catalog penalty follow the tool or its position in the schema block?

Reads the base-model catalog runs under several schema orders — the original (rotation 0), rotations
(--schema-rotate), and optionally random shuffles (--schema-order seeds) — reconstructs each tool's
position in each run, and reports:

  1. accuracy by position quartile, pooled over the rotation runs (balanced: every tool appears in every
     quartile once, so quartile means are free of tool identity);
  2. per-tool accuracy across its positions (does a tool stay bad wherever it sits?);
  3. a variance decomposition of the tool x run cell table into tool identity, position quartile, residual;
  4. the within-run Spearman(position, penalty) that the original single-order analysis reported.

Usage:
  python position_analysis.py --catalog tools/catalog_64.json --catalog-size 64 \\
      --model openai_gpt_oss_20b --rotations 0,16,32,48 [--shuffles 1,2]
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics as st
from pathlib import Path


def per_tool_acc(path: str, arm="A", set_name="natural") -> dict[str, float]:
    d = collections.defaultdict(lambda: [0, 0])
    for l in open(path):
        r = json.loads(l)
        if r["arm"] == arm and r["set"] == set_name:
            d[r["tool"]][0] += r["score"]["args_exact"]; d[r["tool"]][1] += 1
    return {t: a / n for t, (a, n) in d.items()}


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys); mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="tools/catalog_64.json")
    ap.add_argument("--catalog-size", type=int, default=64)
    ap.add_argument("--results", default="results")
    ap.add_argument("--model", default="openai_gpt_oss_20b")
    ap.add_argument("--rotations", default="0,16,32,48")
    ap.add_argument("--shuffles", default="", help="comma list of --schema-order seeds used without --schema-rotate")
    args = ap.parse_args()

    N = args.catalog_size
    cat = json.loads(Path(args.catalog).read_text())["tools"][:N]
    names = [t["mcp"]["name"] for t in cat]
    pre = f"{args.results}/{args.model}__cat{N}__base_pv1"

    runs = []   # (label, kind, positions{tool: idx}, acc{tool: float})
    for k in [int(x) for x in args.rotations.split(",") if x != ""]:
        f = f"{pre}.rows.jsonl" if k == 0 else f"{pre}_rot{k}.rows.jsonl"
        if not Path(f).exists():
            print(f"missing {f}"); continue
        order = names[-k:] + names[:-k] if k % N else list(names)  # same rotation as eval_arms --schema-rotate
        runs.append((f"rot{k}", "rotation", {t: i for i, t in enumerate(order)}, per_tool_acc(f)))
    for sd in [int(x) for x in args.shuffles.split(",") if x != ""]:
        f = f"{pre}_order{sd}.rows.jsonl"
        if not Path(f).exists():
            print(f"missing {f}"); continue
        order = list(names); random.Random(sd).shuffle(order)
        runs.append((f"shuffle{sd}", "shuffle", {t: i for i, t in enumerate(order)}, per_tool_acc(f)))
    single_f = f"{pre}_single.rows.jsonl"
    single = per_tool_acc(single_f) if Path(single_f).exists() else {}
    if not runs:
        raise SystemExit("no runs found")
    q = lambda pos: pos * 4 // N

    print(f"{len(runs)} runs: " + ", ".join(f"{lab} (overall {st.mean(acc.values()):.3f})" for lab, _, _, acc in runs))

    # 1. balanced position means over the rotation runs
    rot = [r for r in runs if r[1] == "rotation"]
    if len(rot) == 4:
        byq = collections.defaultdict(list)
        for _, _, pos, acc in rot:
            for t in names:
                byq[q(pos[t])].append(acc[t])
        print("\n1. accuracy by position quartile, pooled over 4 rotations (each tool once per quartile):")
        print("   " + "  ".join(f"Q{i+1}={st.mean(byq[i]):.3f}" for i in range(4)) + f"   (spread {max(st.mean(byq[i]) for i in range(4)) - min(st.mean(byq[i]) for i in range(4)):.3f})")
        if single:
            # single−catalog penalty in each quartile, pooled over rotations
            byqp = {i: st.mean(single[t] - a for (_, _, pos, acc) in rot for t in names for a in [acc[t]] if q(pos[t]) == i) for i in range(4)}
            print("   penalty (single − catalog) by quartile: " + "  ".join(f"Q{i+1}={byqp[i]:+.3f}" for i in range(4)))
    else:
        print(f"\n1. only {len(rot)} rotation runs found; need 0,16,32,48 for the balanced design")

    # 2. per-tool consistency across positions
    tool_mean = {t: st.mean(acc[t] for _, _, _, acc in runs) for t in names}
    tool_range = {t: max(acc[t] for _, _, _, acc in runs) - min(acc[t] for _, _, _, acc in runs) for t in names}
    worst = sorted(names, key=lambda t: tool_mean[t])[:10]
    print("\n2. ten worst tools by mean catalog accuracy across orders — accuracy at each position:")
    print(f"   {'tool':34s} mean   " + "  ".join(f"{lab:>9s}" for lab, *_ in runs) + "   single")
    for t in worst:
        cells = "  ".join(f"{acc[t]:.2f}@Q{q(pos[t])+1}" for _, _, pos, acc in runs)
        print(f"   {t:34s} {tool_mean[t]:.3f}  {cells}   {single.get(t, float('nan')):.2f}")
    print(f"   mean within-tool range across orders: {st.mean(tool_range.values()):.3f}; "
          f"between-tool SD of tool means: {st.pstdev(tool_mean.values()):.3f}")

    # 3. variance decomposition
    def decompose(run_subset, label):
        cells = [(t, q(pos[t]), acc[t]) for _, _, pos, acc in run_subset for t in names]
        n_runs = len(run_subset)
        grand = st.mean(c[2] for c in cells)
        ss_tot = sum((c[2] - grand) ** 2 for c in cells)
        tmean = {t: st.mean(acc[t] for _, _, _, acc in run_subset) for t in names}
        ss_tool = sum(n_runs * (tmean[t] - grand) ** 2 for t in names)
        # sequential: position's explained variation AFTER tool identity (fit on residuals of the tool means)
        resid = [(c[1], c[2] - tmean[c[0]]) for c in cells]
        qres = {i: st.mean(r for qq, r in resid if qq == i) for i in range(4)}
        ss_pos_seq = sum(qres[qq] ** 2 for qq, _ in resid)
        # raw (non-sequential) quartile SS, valid as an additive term only when the design is balanced
        qmean = {i: st.mean(c[2] for c in cells if c[1] == i) for i in range(4)}
        ss_pos_raw = sum((qmean[c[1]] - grand) ** 2 for c in cells)
        print(f"   {label} (n={len(cells)} cells): tool identity {ss_tool/ss_tot:.1%}, "
              f"position quartile after identity {ss_pos_seq/ss_tot:.1%}, residual {1 - (ss_tool+ss_pos_seq)/ss_tot:.1%}"
              f"   [raw quartile SS {ss_pos_raw/ss_tot:.1%}{'' if ss_pos_raw <= ss_pos_seq + 1e-9 else ' — unbalanced, not additive'}]")
    print("\n3. variance decomposition (sequential: tool identity first, then position quartile on the residual):")
    if len(rot) == 4:
        decompose(rot, "balanced, 4 rotations only")
    decompose(runs, f"all {len(runs)} orders")

    # 4. within-run Spearman(position, penalty)
    if single:
        print("\n4. within-run Spearman(position, penalty):")
        for lab, _, pos, acc in runs:
            xs = [pos[t] for t in names]; ys = [single[t] - acc[t] for t in names]
            print(f"   {lab:10s} rho={spearman(xs, ys):+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())