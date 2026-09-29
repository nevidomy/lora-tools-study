#!/usr/bin/env python3
"""Strict re-score of the controlled renderer runs from stored rows. No API.

Strict = same tool, same keys, every value schema-typed and strictly equal (False != 0, "28" != 28,
no punctuation stripping). "harness" = the study harness's type-coerced score, for comparison.

Usage: python strict_rescore.py --catalog tools/catalog_64.json --catalog-size 4
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

RUNS = [
    ("gpt_oss_no_sysprompt",            "results/openai_gpt_oss_20b__cat4__rab_pv1_rno_sysprompt.rows.jsonl"),
    ("gpt_oss_low_reasoning",           "results/openai_gpt_oss_20b__cat4__rab_pv1_rlow_reasoning.rows.jsonl"),
    ("gpt_oss_medium_reasoning",        "results/openai_gpt_oss_20b__cat4__rab_pv1_rmedium_reasoning.rows.jsonl"),
    ("low_reasoning, date 2026-09-13",  "results/openai_gpt_oss_20b__cat4__rab_date13_pv1_rlow_reasoning.rows.jsonl"),
]


def type_ok(spec, v):
    t = spec.get("type")
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "boolean":
        return isinstance(v, bool)
    if t == "string":
        return isinstance(v, str) and ("enum" not in spec or v in spec["enum"])
    return True


def strict_equal(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(strict_equal(a[k], b[k]) for k in a)
    return a == b


def strict_exact(tool, truth, pred_name, pred_args):
    if pred_name != truth["name"] or not isinstance(pred_args, dict):
        return False
    props = tool["mcp"]["inputSchema"]["properties"]
    if pred_args.keys() != truth["arguments"].keys():
        return False
    return all(k in props and type_ok(props[k], pred_args[k]) for k in pred_args) \
        and strict_equal(pred_args, truth["arguments"])


def score_file(path, cat, arm_filter=None):
    """Returns {(arm, set): (n, strict_exact, harness_exact, diffs[list])}."""
    out = {}
    for l in open(path):
        r = json.loads(l)
        if arm_filter and r["arm"] != arm_filter:
            continue
        key = (r["arm"], r["set"])
        n, se_c, h_c, diffs = out.get(key, (0, 0, 0, []))
        tool = cat[r["tool"]]
        h = bool(r["score"]["args_exact"])
        se = strict_exact(tool, r["truth"], r["pred_name"], r["pred_args"])
        if se != h:
            diffs.append((r["query_id"], r["truth"]["arguments"], r["pred_args"]))
        out[key] = (n + 1, se_c + se, h_c + h, diffs)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="tools/catalog_64.json")
    ap.add_argument("--catalog-size", type=int, default=4)
    ap.add_argument("--all", action="store_true", help="sweep every results/*.rows.jsonl (catalog size parsed from the filename)")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    full = json.loads(Path(args.catalog).read_text())["tools"]

    if args.all:
        files = sorted(glob.glob(f"{args.results}/*.rows.jsonl"))
        tot_rows = tot_diff = 0
        print(f"{'file':70s} {'arm/set':16s} {'n':>5s}  {'strict':>7s}  {'harness':>7s}  diffs")
        for f in files:
            stem = Path(f).name.replace(".rows.jsonl", "")
            m = re.search(r"__cat(\d+)__", stem)
            if not m:
                continue
            size = int(m.group(1))
            cat = {t["mcp"]["name"]: t for t in full[:size]}
            for (arm, set_name), (n, se_c, h_c, diffs) in sorted(score_file(f, cat).items()):
                if arm == "B":
                    continue
                tot_rows += n; tot_diff += len(diffs)
                flag = "" if not diffs else f"  <-- {diffs[0][0]}: truth={diffs[0][1]} pred={diffs[0][2]}"
                print(f"{stem[:70]:70s} {arm + '/' + set_name:16s} {n:5d}  {se_c/n:7.3f}  {h_c/n:7.3f}  {len(diffs)}{flag}")
        print(f"\ntotal: {tot_rows} rows, {tot_diff} strict/harness disagreements ({tot_diff/max(tot_rows,1):.3%})")
        return 0

    cat = {t["mcp"]["name"]: t for t in full[: args.catalog_size]}
    print(f"{'renderer':34s} {'n':>3s}  {'strict exact':>12s}  {'harness exact':>13s}  {'strict recovered':>16s}  {'harness recovered':>17s}  diffs")
    for label, path in RUNS:
        if not Path(path).exists():
            print(f"{label:34s} (missing: {path})"); continue
        rows = [json.loads(l) for l in open(path) if json.loads(l)["arm"] == "A" and json.loads(l)["set"] == "natural"]
        n = len(rows)
        s_ex = h_ex = s_rec = h_rec = 0
        diffs = []
        for r in rows:
            tool = cat[r["tool"]]
            h = r["score"]["args_exact"]
            se = strict_exact(tool, r["truth"], r["pred_name"], r["pred_args"])
            # recovered: malformed envelope, name taken from <|constrain|>functions.NAME
            hr = r["score"].get("recovered_exact", h)
            sr = se
            if not r["score"]["parsed"] and r.get("json_in_text") and r.get("recovered_name"):
                sr = strict_exact(tool, r["truth"], r["recovered_name"], r.get("json_in_text_args"))
            s_ex += se; h_ex += h; s_rec += sr; h_rec += hr
            if se != h:
                diffs.append((r["query_id"], r["truth"]["arguments"], r["pred_args"]))
        print(f"{label:34s} {n:3d}  {s_ex/n:12.3f}  {h_ex/n:13.3f}  {s_rec/n:16.3f}  {h_rec/n:17.3f}  {len(diffs)}")
        for qid, t, p in diffs[:3]:
            print(f"      {qid}: truth={t} pred={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())