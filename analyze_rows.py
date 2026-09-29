#!/usr/bin/env python3
"""Why did args_exact fail? Field-level breakdown of eval rows.

Usage:
  python analyze_rows.py results/openai_gpt_oss_20b__cat4__base.rows.jsonl --arm A --set natural
  python analyze_rows.py results/....rows.jsonl --arm A --set natural --show 8   # print examples
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def classify(row: dict, props: dict) -> list[str]:
    sc = row["score"]
    if not sc["parsed"]:
        if row.get("json_in_text"):
            return ["json_in_text:content_ok" if sc.get("json_content_exact") else "json_in_text:content_wrong"]
        if "length" in str(row.get("stop_reason", "")).lower():
            return ["truncated"]
        return ["no_call"]
    if not sc["tool_ok"]:
        return ["wrong_tool"]
    if not sc["schema_valid"]:
        return [f"invalid:{k}" for k in ("extra_keys", "missing_required", "bad_values") if sc[k]]
    truth, pred = row["truth"]["arguments"], row["pred_args"] or {}
    out = []
    for k in set(truth) | set(pred):
        t, p = truth.get(k, "<absent>"), pred.get(k, "<absent>")
        if t == p:
            continue
        spec = props.get(k, {})
        kind = "enum" if "enum" in spec else spec.get("type", "?")
        if t == "<absent>":
            out.append(f"extra_optional:{kind}")
        elif p == "<absent>":
            out.append(f"dropped_optional:{kind}")
        elif kind == "boolean":
            out.append("bool_flipped")
        else:
            out.append(f"wrong_{kind}")
    return out or ["exact"]


def load_queries(catalog: dict[str, dict], n_tools: int, dirs: dict[str, str]) -> dict[str, dict[str, dict]]:
    """Fallback lookup of query text by id, for rows files written before `query` was logged."""
    queries: dict[str, dict[str, dict]] = {}
    for set_name, qdir in dirs.items():
        by_id = {}
        for tool_name in list(catalog.keys())[:n_tools]:
            p = Path(qdir) / f"{tool_name}.jsonl"
            if not p.exists():
                continue
            with open(p) as f:
                for line in f:
                    q = json.loads(line)
                    by_id[q["query_id"]] = q
        queries[set_name] = by_id
    return queries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--catalog", default="tools/catalog_64.json")
    ap.add_argument("--catalog-size", type=int, default=4)
    ap.add_argument("--queries-dir", default="queries")
    ap.add_argument("--strong-dir", default="queries_strong")
    ap.add_argument("--arm", default="A")
    ap.add_argument("--set", default="natural")
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()

    cat = {t["mcp"]["name"]: t for t in json.loads(Path(args.catalog).read_text())["tools"]}
    rows = [json.loads(l) for l in open(args.rows)]
    rows = [r for r in rows if r["arm"] == args.arm and r["set"] == args.set]
    fails = [r for r in rows if not r["score"]["args_exact"]]
    n_len = sum(1 for r in rows if r["score"].get("args_lenient"))
    print(f"{args.arm}/{args.set}: {len(rows)} rows, {len(fails)} not exact ({len(fails)/max(len(rows),1):.1%}); "
          f"lenient exact = {n_len/max(len(rows),1):.1%}\n")
    queries = load_queries(cat, args.catalog_size, {"natural": args.queries_dir, "strong": args.strong_dir})

    def qtext(r: dict) -> str:
        return r.get("query") or queries.get(r["set"], {}).get(r["query_id"], {}).get("query") or r["query_id"]

    causes = collections.Counter()
    by_tool = collections.defaultdict(collections.Counter)
    examples = collections.defaultdict(list)
    for r in fails:
        props = cat[r["tool"]]["mcp"]["inputSchema"]["properties"]
        for c in classify(r, props):
            causes[c] += 1
            by_tool[r["tool"]][c] += 1
            examples[c].append(r)

    print("cause                     count   (one row can have several)")
    for c, n in causes.most_common():
        print(f"  {c:24s} {n:4d}")
    print("\nby tool:")
    for t, cc in by_tool.items():
        print(f"  {t:32s} " + ", ".join(f"{c}={n}" for c, n in cc.most_common(4)))

    if args.show:
        print("\n" + "=" * 70)
        for c, _ in causes.most_common():
            print(f"\n--- {c} ---")
            for r in examples[c][: args.show]:
                print(f"  [{r['tool']}] {qtext(r)} [{r.get('completion_tokens')}]")
                print(f"    truth: {json.dumps(r['truth']['arguments'])}")
                print(f"    pred : {json.dumps(r['pred_args'])}")
                if r.get("raw_decoded"):
                    print(f"    tokens: {r['raw_decoded'][:260]!r}")
                elif r.get("raw_text"):
                    print(f"    raw  : {r['raw_text'][:160]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())