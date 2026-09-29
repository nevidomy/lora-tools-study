#!/usr/bin/env python3
"""strong = rewrite(natural): remove enum-value words from natural queries by minimal edit.

Why rewrite instead of generate: a negative constraint ("don't say X") is easy to
follow when X is visible in the input and hard when the model is composing from
scratch. Rewriting also makes the natural/strong pair differ ONLY in enum phrasing —
same call, same persona, same sentence structure, same strings and ints.

Per tool:
  1. glossary of validated paraphrases (cached; reuses queries_gen helpers)
  2. batch rewrite: for each query, list the exact words to remove and the values
     to keep verbatim; ask for a minimal edit
  3. strict check (echo = reject); on retry, tell the model exactly what leaked

Usage:
  python strongify.py --catalog tools/catalog_64.json --from queries --out queries_strong \\
      --tools 4 --provider openrouter --model x-ai/grok-4.3 --reasoning off
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from queries_gen import (  # noqa: E402
    LLM, PROVIDERS, _norm, _value_words, check_query, glossary_prompt, validate_glossary,
)


def rewrite_prompt(tool: dict, items: list[dict], gl: dict) -> str:
    props = tool["mcp"]["inputSchema"]["properties"]
    lines = []
    for it in items:
        call = it["call"]["arguments"]
        avoid, keep = [], []
        for k, v in call.items():
            spec = props[k]
            if "enum" in spec:
                paras = gl.get(k, {}).get(v, [])
                avoid.append(f'"{v.replace("_", " ")}" (say instead: ' + (" / ".join(paras) if paras else "describe it without those words") + ")")
            elif spec["type"] == "string":
                keep.append(f'"{v}"')
            elif spec["type"] == "integer":
                keep.append(str(v))
        fb = f"\n   PREVIOUS ATTEMPT STILL LEAKED: {it['leaked']}" if it.get("leaked") else ""
        lines.append(
            f"{it['idx']}. query: {json.dumps(it['query'])}\n"
            f"   remove these phrases: {'; '.join(avoid) or '(none)'}\n"
            f"   keep exactly: {', '.join(keep) or '(nothing)'}{fb}"
        )
    return f"""Tool schema (for context only): {tool['schema_text']}

Rewrite each query below with the SMALLEST possible edit so that it no longer contains the
listed phrases OR any of their individual words (e.g. for "low oxygen" avoid both "low" and
"oxygen" in that sense). Use the suggested replacements or an equivalent. Everything else stays:
same meaning, same tone, same numbers, same quoted values character-for-character.

{chr(10).join(lines)}

Output JSON only: {{"rewrites": [{{"idx": 0, "query": "..."}}, ...]}} with one entry per item."""


def domain_words(tool: dict) -> set[str]:
    """Words the query is allowed to contain even if a value shares them: the topic itself."""
    txt = tool["mcp"]["name"] + " " + tool["skeleton"]["domain"]
    return {w for w in re.findall(r"[a-z0-9]+", txt.lower()) if len(w) > 2}


def leaked_words(tool: dict, call: dict, query: str) -> list[str]:
    """Which enum values (or their distinctive words) are still present.
    Word-level, stricter than echo, but exempts domain/topic words."""
    props = tool["mcp"]["inputSchema"]["properties"]
    qw = set(re.findall(r"[a-z0-9]+", query.lower()))
    exempt = set(domain_words(tool))
    for k, v in call["arguments"].items():          # verbatim string values legitimately carry their words
        if props[k]["type"] == "string" and "enum" not in props[k]:
            exempt |= set(re.findall(r"[a-z0-9]+", str(v).lower()))
    out = []
    for k, v in call["arguments"].items():
        if "enum" in props[k]:
            distinctive = _value_words(v) - exempt
            hit = distinctive & qw
            if hit or _norm(v) in _norm(query):
                out.append(f"{v}: {sorted(hit) or 'phrase'}")
    return out


def mock_rewrite(tool: dict, items: list[dict]) -> dict:
    props = tool["mcp"]["inputSchema"]["properties"]
    out = []
    for it in items:
        q = it["query"]
        for k, v in it["call"]["arguments"].items():
            if "enum" in props[k]:
                q = q.replace(v.replace("_", " "), "that particular setting")
                for w in _value_words(v):
                    q = re.sub(rf"\b{re.escape(w)}\b", "thing", q, flags=re.I)
        out.append({"idx": it["idx"], "query": q})
    return {"rewrites": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--from", dest="src", required=True, help="natural-mode queries dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tools", type=int, default=None)
    ap.add_argument("--split", default=None, help="only rewrite this split (e.g. held); default all")
    ap.add_argument("--provider", default="openrouter", choices=list(PROVIDERS) + ["mock"])
    ap.add_argument("--model", default="x-ai/grok-4.6")
    ap.add_argument("--reasoning", default="off", choices=["off", "low", "default"])
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true", help="keep rows already in --out; only process missing ones")
    ap.add_argument("--avoid-glossary", default=None,
                    help="path to an existing glossary.json; generate paraphrases DISJOINT from it (held-out test set)")
    args = ap.parse_args()

    cat = json.loads(Path(args.catalog).read_text())
    tools = cat["tools"][: args.tools] if args.tools else cat["tools"]
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(args.provider, args.model, args.temperature, args.reasoning)
    t0 = time.time()

    # glossaries
    gpath = out / "glossary.json"
    glossaries = json.loads(gpath.read_text()) if gpath.exists() else {}

    avoid = json.loads(Path(args.avoid_glossary).read_text()) if args.avoid_glossary else {}

    def _norm_p(t: str) -> str:
        return re.sub(r"[^a-z0-9 ]+", "", t.lower()).strip()

    def get_gl(tool):
        name = tool["mcp"]["name"]
        if name in glossaries:
            return name, glossaries[name]
        prompt = glossary_prompt(tool)
        av = avoid.get(name, {})
        if av:
            used = [f"  {p}={v}: " + " | ".join(ps) for p, vals in av.items() for v, ps in vals.items()]
            prompt += ("\n\nThese paraphrases are ALREADY USED and must NOT be reused or closely rephrased; "
                       "give genuinely different ones (different imagery, different sentence shape):\n" + "\n".join(used))
        obj, _ = llm.json_call(prompt, 2500,
                               mock_fn=lambda: {"glossary": {p: {v: ["the special one", "that other setting", "the usual choice"]
                                                              for v in sp["enum"]}
                                                          for p, sp in tool["mcp"]["inputSchema"]["properties"].items() if "enum" in sp}})
        gl = validate_glossary(tool, obj)
        if av:  # drop anything that collides with the avoid set (normalized, or sharing >=60% of words)
            for p, vals in list(gl.items()):
                for v, ps in list(vals.items()):
                    banned = [_norm_p(x) for x in av.get(p, {}).get(v, [])]
                    keep = []
                    for x in ps:
                        nx = _norm_p(x)
                        wx = set(nx.split())
                        if nx in banned:
                            continue
                        if any(len(wx & set(b.split())) / max(len(wx), 1) >= 0.6 for b in banned):
                            continue
                        keep.append(x)
                    if keep:
                        vals[v] = keep
                    else:
                        del vals[v]
                if not vals:
                    del gl[p]
        return name, gl

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for name, gl in ex.map(get_gl, tools):
            glossaries[name] = gl
    gpath.write_text(json.dumps(glossaries, indent=1))
    if avoid:
        n_ok = sum(len(v) for g in glossaries.values() for v in g.values())
        n_av = sum(len(v) for g in avoid.values() for v in g.values())
        print(f"disjoint glossary: {n_ok} values covered (avoid set covered {n_av})", file=sys.stderr)

    rejects = open(out / "rejects.jsonl", "a" if args.resume else "w")
    stats = {"ok": 0, "failed": 0, "batches": 0, "echo_sum": 0.0}

    def do_tool(tool):
        name = tool["mcp"]["name"]
        f = src / f"{name}.jsonl"
        if not f.exists():
            return name, []
        rows = [json.loads(l) for l in open(f)]
        if args.split:
            rows = [r for r in rows if r["split"] == args.split]
        gl = glossaries.get(name, {})
        pending = []
        done = {}
        existing = out / f"{name}.jsonl"
        if args.resume and existing.exists():
            for l in open(existing):
                pr = json.loads(l)
                done[pr["i"]] = pr
            rows = [r for r in rows if r["i"] not in done]
        # items needing work: those whose natural query leaks enum words
        for r in rows:
            errs, echo = check_query(tool, r["call"], r["query"], strict_echo=True)
            if not errs and not leaked_words(tool, r["call"], r["query"]):
                done[r["i"]] = {**r, "natural_query": r["query"], "echo_score": echo, "rewritten": False}
            else:
                pending.append({"idx": r["i"], "query": r["query"], "call": r["call"], "row": r, "leaked": None})
        for attempt in range(args.retries + 1):
            if not pending:
                break
            nxt = []
            for b in range(0, len(pending), args.batch):
                chunk = pending[b: b + args.batch]
                local = [{**it, "idx": j} for j, it in enumerate(chunk)]
                obj, prov = llm.json_call(rewrite_prompt(tool, local, gl), 300 * len(chunk) + 200,
                                          mock_fn=lambda: mock_rewrite(tool, local))
                stats["batches"] += 1
                got = {e["idx"]: e["query"].strip() for e in ((obj or {}).get("rewrites") or [])
                       if isinstance(e, dict) and isinstance(e.get("idx"), int) and isinstance(e.get("query"), str)}
                for j, it in enumerate(chunk):
                    q = got.get(j)
                    r = it["row"]
                    if q is None:
                        it["leaked"] = "missing from batch"; nxt.append(it); continue
                    errs, echo = check_query(tool, r["call"], q, strict_echo=True)
                    lk = leaked_words(tool, r["call"], q)
                    if avoid:
                        qn = _norm_p(q)
                        for p_, v_ in r["call"]["arguments"].items():
                            for ban in avoid.get(name, {}).get(p_, {}).get(v_, []):
                                if _norm_p(ban) and _norm_p(ban) in qn:
                                    lk.append(f"{v_}: reused avoided paraphrase {ban!r}")
                    if errs or lk:
                        rejects.write(json.dumps({"tool": name, "i": r["i"], "attempt": attempt, "query": q,
                                                  "call": r["call"], "natural_query": r["query"],
                                                  "errors": errs, "leaked": lk}) + "\n")
                        it["leaked"] = "; ".join(lk) if lk else "; ".join(errs)
                        # keep the rewrite as next input unless it dropped a required value or leaked a token
                        it["query"] = q if not any(e.startswith(("missing", "copies")) for e in errs) else it["query"]
                        nxt.append(it); continue
                    done[r["i"]] = {**r, "natural_query": r["query"], "query": q, "echo_score": echo,
                                    "rewritten": True, "rewrite_attempt": attempt, "rewrite_prov": prov}
            pending = nxt
        for it in pending:
            stats["failed"] += 1
            rejects.write(json.dumps({"tool": name, "i": it["row"]["i"], "attempt": "final",
                                      "query": it["query"], "call": it["row"]["call"],
                                      "natural_query": it["row"]["query"],
                                      "leaked": it["leaked"], "unrecoverable": True}) + "\n")
        rows_out = [done[i] for i in sorted(done)]
        return name, rows_out

    manifest_tools = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for name, rows in ex.map(do_tool, tools):
            if not rows:
                continue
            (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            stats["ok"] += len(rows)
            stats["echo_sum"] += sum(r["echo_score"] for r in rows)
            n_rw = sum(r["rewritten"] for r in rows)
            manifest_tools[name] = {"n": len(rows), "rewritten": n_rw,
                                    "held": sum(r["split"] == "held" for r in rows),
                                    "train": sum(r["split"] == "train" for r in rows)}
            print(f"  {name}: {len(rows)} ok ({n_rw} rewritten)  ({time.time()-t0:.0f}s)", file=sys.stderr)
    rejects.close()

    src_manifest = json.loads((src / "manifest.json").read_text()) if (src / "manifest.json").exists() else {}
    manifest = {**src_manifest, "paraphrase": "strong", "derived_from": str(src), "method": "rewrite",
                "avoid_glossary": args.avoid_glossary, "split": args.split,
                "rewrite_model": args.model, "rewrite_provider": args.provider,
                "ok": stats["ok"], "failed": stats["failed"], "batches": stats["batches"],
                "mean_echo_score": round(stats["echo_sum"] / max(stats["ok"], 1), 4),
                "per_tool": manifest_tools,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{stats['ok']} strong queries, {stats['failed']} unrecoverable, {stats['batches']} batches, "
          f"echo={manifest['mean_echo_score']}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())