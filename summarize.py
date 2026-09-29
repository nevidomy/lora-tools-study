#!/usr/bin/env python3
"""Aggregate eval summaries across seeds.

Reads results/*.summary.json, parses the tag into (config, seed), and reports
mean ± std (and n) per config × arm × set for exact, lenient, invention, and
rewritten-only exact. Seeds are recognized by a trailing `_s<digit>` in the tag;
runs without one (e.g. `base`) are reported as n=1.

Usage:
  python summarize.py                     # table
  python summarize.py --csv > table.csv   # for plotting
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
from collections import defaultdict


def parse_tag(stem: str) -> tuple[str, str, str | None, str]:
    """'openai_gpt_oss_20b__cat4__cat4_ex100_C_s0_unseen_pv1'
       -> (model, 'cat4_ex100_C', 's0', 'unseen')     variant modifies the set name"""
    parts = stem.split("__")
    model, cat, tag = parts[0], parts[1], "__".join(parts[2:])
    tag = re.sub(r"_p(v\d+)$", "", tag)             # drop prompt suffix
    if not tag.startswith(cat + "_"):                # base/base_unseen tags lack the catalog prefix
        tag = f"{cat}_{tag}"
    variant = ""
    if tag.endswith("_single"):
        tag = tag[: -len("_single")] + "_SINGLE"     # keep as part of config name, not set
    m = re.match(r"^(.*)_(unseen|repeat\d*)$", tag)
    if m:
        tag, variant = m.group(1), m.group(2)
    m = re.match(r"^(.*)_(s\d+)$", tag)
    if m:
        return model, m.group(1), m.group(2), variant
    return model, tag, None, variant


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--seeds", action="store_true", help="also list per-seed exact values")
    ap.add_argument("--filter", default=None, help="regex on config name (e.g. 'lr2e-4|^base')")
    ap.add_argument("--min-n", type=int, default=0, help="hide rows with fewer seeds (base rows always shown)")
    ap.add_argument("--sets", default=None, help="comma list of sets to show, e.g. natural,strong_unseen")
    ap.add_argument("--no-rep", action="store_true", help="hide *_rep repeat-eval rows")
    ap.add_argument("--tail-loss", action="store_true",
                    help="append per-config tail loss from runs/<tag>.json: mean and max loss/token over the last 10%% of steps")
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()
    want_sets = set(args.sets.split(",")) if args.sets else None
    flt = re.compile(args.filter) if args.filter else None

    rows = defaultdict(list)   # (model, config, arm, set) -> list of dicts
    for f in sorted(glob.glob(f"{args.results}/*.summary.json")):
        s = json.load(open(f))
        stem = f.split("/")[-1].replace(".summary.json", "")
        model, config, seed, variant = parse_tag(stem)
        for key, a in s["arms"].items():
            arm, set_name = key.split("/")
            if variant == "unseen":
                set_name = "strong_unseen" if set_name == "strong" else "natural_rep"
            elif variant.startswith("repeat"):
                set_name = f"{set_name}_rep"
            rows[(model, config, arm, set_name)].append({
                "seed": seed,
                "exact": a["args_exact"], "lenient": a["args_lenient"],
                "invention": a["tool_ok"] - a["schema_valid"],  # rates: right tool, args fail the schema
                "parsed": a["parsed"],
                "rw_exact": a.get("rewritten_args_exact"),
                "prompt_tok": a["prompt_tokens"] / max(a["n"], 1),
                "completion_tok": a["completion_tokens"] / max(a["n"], 1),
                "schema_tokens": s.get("schema_tokens"),
            })

    tails: dict[str, list[tuple[float, float]]] = defaultdict(list)   # config -> [(mean_tail, max_tail)]
    if args.tail_loss:
        for f in glob.glob(f"{args.runs}/*.json"):
            try:
                m = json.load(open(f))
            except Exception:
                continue
            steps = [x.get("loss_per_token") for x in m.get("losses", []) if x.get("loss_per_token") is not None]
            if not steps:
                continue
            tail = steps[-max(1, len(steps) // 10):]
            tag = m.get("tag", "")
            tag = re.sub(r"_(s\d+)$", "", tag)
            tails[tag].append((st.mean(tail), max(tail)))

    def ms(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "   —   "
        if len(vals) == 1:
            return f"{vals[0]:.3f}      "
        return f"{st.mean(vals):.3f}±{st.pstdev(vals):.3f}"

    if args.csv:
        print("model,config,arm,set,n,exact_mean,exact_std,invention_mean,invention_std,rw_exact_mean,prompt_tok,completion_tok")
        for (model, config, arm, set_name), rs in sorted(rows.items()):
            ex = [r["exact"] for r in rs]; inv = [r["invention"] for r in rs]
            rw = [r["rw_exact"] for r in rs if r["rw_exact"] is not None]
            print(f"{model},{config},{arm},{set_name},{len(rs)},{st.mean(ex):.4f},{st.pstdev(ex) if len(ex) > 1 else 0:.4f},"
                  f"{st.mean(inv):.4f},{st.pstdev(inv) if len(inv) > 1 else 0:.4f},"
                  f"{st.mean(rw) if rw else ''},{st.mean(r['prompt_tok'] for r in rs):.0f},{st.mean(r['completion_tok'] for r in rs):.0f}")
        return 0

    models = sorted({k[0] for k in rows})
    multi = len(models) > 1
    def short(m):  # 'thinkingmachines_Inkling_Small' -> 'Inkling_Small', 'openai_gpt_oss_20b' -> 'gpt_oss_20b'
        return m.split("_", 1)[1] if "_" in m else m
    mcol = f"{'model':16s} " if multi else ""
    print(f"{mcol}{'config':22s} {'arm':3s} {'set':13s} {'n':>2s}  {'exact':13s} {'lenient':13s} {'invention':13s} {'rw-only':13s} {'prompt':>6s} {'compl':>5s}")
    last = None
    SET_ORDER = {"natural": 0, "natural_rep": 1, "strong": 2, "strong_unseen": 3}
    def keep(config, set_name, rs):
        if flt and not flt.search(config):
            return False
        if want_sets and set_name not in want_sets:
            return False
        if args.no_rep and set_name.endswith("_rep"):
            return False
        if len(rs) < args.min_n and not config.startswith("base"):
            return False
        return True
    for (model, config, arm, set_name), rs in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], SET_ORDER.get(kv[0][3], 9))):
        if not keep(config, set_name, rs):
            continue
        if last and last != (model, config):
            print()
        last = (model, config)
        seeds = ""
        if args.seeds and len(rs) > 1:
            seeds = "   [" + " ".join(f"{r['seed'] or '-'}:{r['exact']:.3f}" for r in sorted(rs, key=lambda r: r['seed'] or '')) + "]"
        tl = ""
        if args.tail_loss and set_name == "natural" and tails.get(config):
            t = tails[config]
            tl = f"   tail loss/tok mean={st.mean(x[0] for x in t):.4f} max={st.mean(x[1] for x in t):.3f} (n={len(t)})"
        print((f"{short(model):16s} " if multi else "") + f"{config:22s} {arm:3s} {set_name:13s} {len(rs):>2d}  "
              f"{ms([r['exact'] for r in rs])} {ms([r['lenient'] for r in rs])} "
              f"{ms([r['invention'] for r in rs])} {ms([r['rw_exact'] for r in rs])} "
              f"{st.mean(r['prompt_tok'] for r in rs):6.0f} {st.mean(r['completion_tok'] for r in rs):5.0f}{seeds}{tl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())