#!/usr/bin/env python3
"""Round-trip validation of (query -> call) pairs with a strong, non-evaluated model.

For each query, the validator sees the FULL catalog (so cross-tool ambiguity is tested)
and must (a) reproduce the exact call and (b) classify the request's intent as
"perform" (do the action) or "inquire" (ask about status). A pair passes only if the
call matches ground truth after type coercion AND intent matches the tool's verb class.

What this catches that local checks cannot:
  - inverted booleans ("without discarding" -> perform_discard: true)
  - string role confusion ("under R-0785" read as a section, not a reporter)
  - read-phrased requests to write tools, and vice versa
  - queries that also fit another tool in the catalog

Use a DIFFERENT model family from the generator (Grok wrote the queries).

Usage:
  python validate.py --catalog tools/catalog_64.json --catalog-size 4 \\
      --dir queries --provider openrouter --model anthropic/claude-sonnet-4.5 --reasoning off
  python validate.py ... --dir queries_strong                # same for the strong set
  python validate.py ... --dir queries --filter               # also write queries_valid/
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_arms import score_call  # noqa: E402
from queries_gen import LLM, PROVIDERS, READ_VERBS  # noqa: E402


def validator_prompt(tools: list[dict], items: list[dict]) -> str:
    schemas = "\n".join(t["schema_text"] for t in tools)
    lines = "\n".join(f"{i}. {json.dumps(it['query'])}" for i, it in enumerate(items))
    return f"""You are checking user requests against a catalog of tools. Catalog (JSON schemas, one per line):
{schemas}

For EACH request below, decide which single tool it calls and with exactly which arguments.
Rules:
- Include only parameters the request actually specifies; omit unmentioned optional parameters.
- Booleans: read the request literally ("without X" means false, "make sure to X" means true).
- Strings must be copied exactly as they appear in the request.
- Enum values must be one of the schema's listed values.
- Intent is judged RELATIVE TO THE TOOL. "perform" = the user wants this tool's own action carried
  out now. For a lookup/list/inspect tool, asking for the information IS performing it ("what's the
  pond chemistry?" -> perform). "inquire" = the user is asking ABOUT the action rather than
  requesting it: whether it was already done, whether it is scheduled, who did it ("has the pond
  been tested yet?" -> inquire; "is the hazard already logged?" -> inquire).
- Enum values: the request will usually DESCRIBE the choice rather than quote the token. Resolve
  it to the listed value it clearly means ("front edge" -> fore_edge, "treetop fire" -> the canopy
  value, "the biggest one" -> large). Only null the tool if the description fits two listed values
  equally or fits none.
- String values: assign each string to its parameter from how the request labels it ("reported by",
  "in section"). If an unlabeled string could belong to two parameters and only the schema's
  required/optional structure would settle it, set "tool" to null and say "ambiguous role: <value>".
- If no tool fits, or two fit equally well, set "tool" to null and explain in "note".

Requests:
{lines}

Output JSON only:
{{"results": [{{"idx": 0, "tool": "...", "arguments": {{...}}, "intent": "perform", "note": ""}}, ...]}}"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--catalog-size", type=int, required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--provider", default="openrouter", choices=list(PROVIDERS) + ["mock"])
    ap.add_argument("--model", default="google/gemini-3.8-flash", help="a strong model from a DIFFERENT family than the generator")
    ap.add_argument("--reasoning", default="off", choices=["off", "low", "default"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=15)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--filter", action="store_true", help="write <dir>_valid/ with only passing rows")
    ap.add_argument("--resume", action="store_true", help="keep definitive verdicts from validation.jsonl; re-run the rest")
    ap.add_argument("--redo", default=None,
                    help="comma list of verdicts to reopen and re-send under --resume, e.g. no_tool_fits")
    ap.add_argument("--rescore", action="store_true",
                    help="re-derive every verdict from the stored validator predictions (no API); use after a scoring-rule change")
    ap.add_argument("--target-train", type=int, default=100)
    ap.add_argument("--target-held", type=int, default=50)
    ap.add_argument("--gen-args", default="--provider openrouter --model x-ai/grok-4.6 --reasoning off",
                    help="generator flags to echo in the suggested top-up command")
    args = ap.parse_args()

    cat = json.loads(Path(args.catalog).read_text())
    tools = cat["tools"][: args.catalog_size]
    by_name = {t["mcp"]["name"]: t for t in tools}
    d = Path(args.dir)
    llm = LLM(args.provider, args.model, args.temperature, args.reasoning)
    t0 = time.time()

    rows_by_tool: dict[str, list[dict]] = {}
    for t in tools:
        f = d / f"{t['mcp']['name']}.jsonl"
        if f.exists():
            rows_by_tool[t["mcp"]["name"]] = [json.loads(l) for l in open(f)]
    all_rows = [r for rows in rows_by_tool.values() for r in rows]
    print(f"{len(all_rows)} queries across {len(rows_by_tool)} tools; catalog of {len(tools)}", file=sys.stderr)

    DEFINITIVE = {"pass", "tool_mismatch", "args_mismatch", "intent_mismatch", "no_tool_fits"}
    row_by_id = {r["query_id"]: r for r in all_rows}

    def rescore_one(v: dict) -> dict:
        """Recompute verdict from v['pred'] under current rules."""
        r = row_by_id.get(v["query_id"])
        if r is None or not v.get("pred"):
            return v
        tool = by_name[r["tool"]]
        pred_tool, pred_args, intent = v["pred"].get("tool"), v["pred"].get("arguments"), v["pred"].get("intent")
        v = dict(v); v["expect_intent"] = "perform"; v.pop("diff", None)
        if not pred_tool:
            v["verdict"] = "no_tool_fits"
        elif pred_tool != r["tool"]:
            v["verdict"] = "tool_mismatch"
        else:
            sc = score_call(tool, r["call"], pred_tool, pred_args if isinstance(pred_args, dict) else None)
            if not sc["args_exact"]:
                v["verdict"] = "args_mismatch"
                pa = pred_args if isinstance(pred_args, dict) else {}
                v["diff"] = {k: (r["call"]["arguments"].get(k, "<absent>"), pa.get(k, "<absent>"))
                             for k in set(r["call"]["arguments"]) | set(pa)
                             if r["call"]["arguments"].get(k, "<absent>") != pa.get(k, "<absent>")}
            elif intent != "perform":
                v["verdict"] = "intent_mismatch"
            else:
                v["verdict"] = "pass"
        return v

    if args.rescore and (d / "validation.jsonl").exists():
        old_v = [json.loads(l) for l in open(d / "validation.jsonl")]
        new_v = [rescore_one(v) for v in old_v]
        changed = sum(1 for a, b in zip(old_v, new_v) if a["verdict"] != b["verdict"])
        (d / "validation.jsonl").write_text("".join(json.dumps(v) + "\n" for v in new_v))
        print(f"rescore: {changed} verdicts changed under current rules (no API calls)", file=sys.stderr)
        args.resume = True   # then proceed as a resume so only non-definitive rows are re-sent

    redo = set(args.redo.split(",")) if args.redo else set()
    prior: dict[str, dict] = {}
    if args.resume and (d / "validation.jsonl").exists():
        current_text = {r["query_id"]: r["query"] for r in all_rows}
        stale = 0
        for l in open(d / "validation.jsonl"):
            v = json.loads(l)
            if v["verdict"] in DEFINITIVE and v["verdict"] not in redo and v["query_id"] in current_text:
                if v.get("query") == current_text[v["query_id"]]:
                    prior[v["query_id"]] = v
                else:
                    stale += 1
        print(f"resume: keeping {len(prior)} definitive verdicts; {stale} stale (query text changed, re-validating)"
              + (f"; reopening verdicts {sorted(redo)}" if redo else ""), file=sys.stderr)

    def mock_results(items):
        out = []
        for i, it in enumerate(items):
            r = it["row"]
            bad = (r["i"] % 11 == 0)
            args_ = dict(r["call"]["arguments"])
            if bad:
                k = next(iter(args_), None)
                if k is not None: args_[k] = "wrong"
            out.append({"idx": i, "tool": r["tool"], "arguments": args_,
                        "intent": "inquire" if (r["i"] % 13 == 0) else "perform", "note": ""})
        return {"results": out}

    def do_batch(items: list[dict]) -> list[dict]:
        obj, prov = llm.json_call(validator_prompt(tools, items), 350 * len(items) + 300,
                                  mock_fn=lambda: mock_results(items))
        results = ((obj or {}).get("results") or [])
        got = {}
        for e in results:
            if not isinstance(e, dict):
                continue
            idx = e.get("idx")
            if isinstance(idx, str) and idx.strip().isdigit():
                idx = int(idx.strip())  # some models return idx as "0"
            if isinstance(idx, int):
                got[idx] = e
        batch_err = prov.get("error")
        if batch_err or (results and not got):
            errlog.write(json.dumps({"tool": items[0]["row"]["tool"], "n": len(items), "prov": prov,
                                     "sample_keys": list(results[0].keys()) if results and isinstance(results[0], dict) else None}) + "\n")
        out = []
        for i, it in enumerate(items):
            r = it["row"]; tool = by_name[r["tool"]]
            expect_intent = "perform"   # for read tools, "look up X" IS performing the action
            e = got.get(i)
            v = {"query_id": r["query_id"], "tool": r["tool"], "split": r["split"], "query": r["query"],
                 "truth": r["call"], "expect_intent": expect_intent, "validator_model": args.model}
            if e is None:
                v.update(verdict=("api_error" if batch_err and "unparseable" not in batch_err else "parse_fail"),
                         pred=None, error=batch_err); out.append(v); continue
            pred_tool, pred_args, intent = e.get("tool"), e.get("arguments"), e.get("intent")
            v["pred"] = {"tool": pred_tool, "arguments": pred_args, "intent": intent, "note": e.get("note", "")}
            if not pred_tool:
                v["verdict"] = "no_tool_fits"
            elif pred_tool != r["tool"]:
                v["verdict"] = "tool_mismatch"
            else:
                sc = score_call(tool, r["call"], pred_tool, pred_args if isinstance(pred_args, dict) else None)
                if not sc["args_exact"]:
                    v["verdict"] = "args_mismatch"
                    v["diff"] = {k: (r["call"]["arguments"].get(k, "<absent>"), (pred_args or {}).get(k, "<absent>"))
                                 for k in set(r["call"]["arguments"]) | set(pred_args or {})
                                 if r["call"]["arguments"].get(k, "<absent>") != (pred_args or {}).get(k, "<absent>")}
                elif intent != expect_intent:
                    v["verdict"] = "intent_mismatch"
                else:
                    v["verdict"] = "pass"
            out.append(v)
        return out

    errlog = open(d / "validation_errors.jsonl", "a")
    batches = []
    for name, rows in rows_by_tool.items():
        items = [{"query": r["query"], "row": r} for r in rows if r["query_id"] not in prior]
        for b in range(0, len(items), args.batch):
            batches.append(items[b: b + args.batch])
    print(f"sending {sum(len(b) for b in batches)} queries in {len(batches)} batches", file=sys.stderr)
    verdicts: list[dict] = list(prior.values())
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(do_batch, batches):
            verdicts.extend(res)
    errlog.close()
    order = {r["query_id"]: k for k, r in enumerate(all_rows)}
    verdicts.sort(key=lambda v: order.get(v["query_id"], 1 << 30))

    (d / "validation.jsonl").write_text("".join(json.dumps(v) + "\n" for v in verdicts))
    counts = collections.Counter(v["verdict"] for v in verdicts)
    by_tool = collections.defaultdict(collections.Counter)
    for v in verdicts:
        by_tool[v["tool"]][v["verdict"]] += 1
    n = len(verdicts)
    print(f"\nvalidated {n} in {time.time()-t0:.0f}s  ({len(batches)} batches, {args.model})")
    for k, c in counts.most_common():
        print(f"  {k:16s} {c:5d}  ({c/n:.1%})")
    print("\nby tool:")
    for t, cc in by_tool.items():
        print(f"  {t:32s} pass={cc['pass']}/{sum(cc.values())}  " +
              ", ".join(f"{k}={v}" for k, v in cc.most_common() if k != "pass"))

    if args.filter:
        ok = {v["query_id"] for v in verdicts if v["verdict"] == "pass"}
        out = Path(str(d).rstrip("/") + "_valid"); out.mkdir(exist_ok=True)
        src_man = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {}
        held_only = src_man.get("split") == "held"
        kept = 0
        shortfall: dict[str, dict[str, int]] = {}
        for name, rows in rows_by_tool.items():
            keep = sorted((r for r in rows if r["query_id"] in ok), key=lambda r: r["i"])
            # cap at targets, assign nested train_rank over the validated train rows
            tr = [r for r in keep if r["split"] == "train"][: args.target_train]
            he = [r for r in keep if r["split"] == "held"][: args.target_held]
            for k, r in enumerate(tr):
                r["train_rank"] = k
            for r in he:
                r["train_rank"] = None
            keep = sorted(tr + he, key=lambda r: r["i"])
            kept += len(keep)
            need = {"train": max(0, args.target_train - len(tr)), "held": max(0, args.target_held - len(he))}
            if held_only:
                need["train"] = 0        # a --split held set has no train rows by design
            if need["train"] or need["held"]:
                shortfall[name] = need
            (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
        for extra in ("string_pools.json", "glossary.json"):
            if (d / extra).exists():
                (out / extra).write_text((d / extra).read_text())
        man = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {}
        final_counts = {}
        for name in rows_by_tool:
            fr = [json.loads(l) for l in open(out / f"{name}.jsonl")]
            final_counts[name] = (sum(r["split"] == "train" for r in fr), sum(r["split"] == "held" for r in fr))
        man.update({"validated_by": args.model, "validation_counts": dict(counts), "ok": kept,
                    "targets": {"train": args.target_train, "held": args.target_held},
                    "per_tool_train_held": final_counts})
        (out / "manifest.json").write_text(json.dumps(man, indent=2))
        print(f"\nfiltered set: {out}/  ({kept} rows, targets train={args.target_train} held={args.target_held})")
        for name, (tr, he) in man["per_tool_train_held"].items():
            flag = "" if name not in shortfall else f"   short: {shortfall[name]}"
            if held_only:
                print(f"  {name:32s} held={he:3d}{flag}")
            else:
                print(f"  {name:32s} train={tr:3d} held={he:3d}{flag}")
        derived = man.get("method") == "rewrite" or man.get("paraphrase") == "strong"
        if shortfall and derived:
            print("\nderived (strong) set: no top-up. Paired eval runs on the natural∩strong intersection.")
            for n, need in shortfall.items():
                print(f"  {n}: {need}")
        elif shortfall:
            # over-generate ~15% to cover the next round's reject rate
            bump = {n: {k: (int(v * 1.15) + 1 if v else 0) for k, v in need.items()} for n, need in shortfall.items()}
            print("\nTOP-UP (generate extras -> validate --resume --filter again):")
            print(f"  python queries_gen.py --catalog {args.catalog} --tools {args.catalog_size} "
                  f"--out {d} {args.gen_args} --topup '{json.dumps(bump)}'")
        else:
            print("\nall tools at target. strongify from this dir next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())