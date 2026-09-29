#!/usr/bin/env python3
"""Rebuild a query set from accepted rows + rejects under the current checker.

Why: an over-tight gate selects for awkward queries (those that dodged the
natural phrase). Re-checking everything with the current rule and preferring
the EARLIEST attempt per (tool, index) recovers the natural ones and removes
the retry bias. Costs nothing.

Usage:
  python rebuild.py --catalog tools/catalog_64.json --dir queries --per-tool 150 --held 50
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from queries_gen import PERSONAS, check_query  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--dir", default="queries")
    ap.add_argument("--per-tool", type=int, required=True)
    ap.add_argument("--held", type=int, default=50)
    ap.add_argument("--strict-echo", action="store_true", help="use for strong-mode dirs")
    args = ap.parse_args()

    d = Path(args.dir)
    cat = {t["mcp"]["name"]: t for t in json.loads(Path(args.catalog).read_text())["tools"]}

    # candidates[(tool, i)] -> list of (attempt, row)
    cands: dict[tuple[str, int], list[tuple[int, dict]]] = collections.defaultdict(list)

    for f in d.glob("*.jsonl"):
        if f.name == "rejects.jsonl":
            continue
        for line in open(f):
            r = json.loads(line)
            cands[(r["tool"], r["i"])].append((r.get("attempt", 0), r))

    n_rej = 0
    for line in open(d / "rejects.jsonl"):
        r = json.loads(line)
        n_rej += 1
        if "query" not in r or "call" not in r:
            continue  # "missing from batch" — nothing to recover
        row = {"query_id": f"{r['tool']}:{r['i']:04d}", "tool": r["tool"], "i": r["i"],
               "persona": r.get("persona"), "call": r["call"], "query": r["query"],
               "attempt": r["attempt"], "prov": r.get("prov"), "recovered": True}
        cands[(r["tool"], r["i"])].append((r["attempt"], row))

    rebuilt: dict[str, list[dict]] = collections.defaultdict(list)
    still_bad = open(d / "rejects_rebuilt.jsonl", "w")
    stats = collections.Counter()

    for (tool, i), lst in cands.items():
        lst.sort(key=lambda x: x[0])
        chosen = None
        for attempt, row in lst:
            errs, echo = check_query(cat[tool], row["call"], row["query"], strict_echo=args.strict_echo)
            if not errs:
                row["echo_score"] = echo
                row["errors_now"] = []
                chosen = row
                break
            still_bad.write(json.dumps({"tool": tool, "i": i, "attempt": attempt,
                                        "query": row["query"], "errors": errs}) + "\n")
        if chosen is None:
            stats["unrecoverable"] += 1
            continue
        if not chosen.get("persona"):
            chosen["persona"] = PERSONAS[(i + cat[tool]["catalog_index"]) % len(PERSONAS)]
        stats["recovered" if chosen.get("recovered") else "kept"] += 1
        rebuilt[tool].append(chosen)
    still_bad.close()

    cut = args.per_tool - args.held
    for tool, rows in rebuilt.items():
        rows.sort(key=lambda r: r["i"])
        for r in rows:
            r.setdefault("split", "held" if r["i"] >= cut else "train")
            r["train_rank"] = None   # validate --filter assigns ranks
        (d / f"{tool}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    total = sum(len(v) for v in rebuilt.values())
    echo_mean = sum(r["echo_score"] for v in rebuilt.values() for r in v) / max(total, 1)
    per_tool_echo = {t: round(sum(r["echo_score"] for r in v) / len(v), 3) for t, v in rebuilt.items()}
    per_tool_n = {t: (sum(r["split"] == "train" for r in v), sum(r["split"] == "held" for r in v))
                  for t, v in rebuilt.items()}

    mpath = d / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest.update({
        "rebuilt": True, "ok": total, "reject_records_original": n_rej,
        "kept": stats["kept"], "recovered": stats["recovered"], "unrecoverable": stats["unrecoverable"],
        "mean_echo_score": round(echo_mean, 4),
        "per_tool_echo_score": per_tool_echo,
        "per_tool_train_held": per_tool_n,
    })
    mpath.write_text(json.dumps(manifest, indent=2))

    print(f"rebuilt: {total} queries  (kept {stats['kept']}, recovered {stats['recovered']}, "
          f"unrecoverable {stats['unrecoverable']})")
    print(f"mean echo_score: {echo_mean:.3f}")
    for t in sorted(per_tool_n):
        tr, he = per_tool_n[t]
        print(f"  {t:36s} train={tr:3d} held={he:3d} echo={per_tool_echo[t]:.2f}")
    print(f"still failing under relaxed rule: {d}/rejects_rebuilt.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())