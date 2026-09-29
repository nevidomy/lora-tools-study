#!/usr/bin/env python3
"""Evaluate tool-call accuracy on held-out queries via Tinker, with native tool rendering.

Arms:
  A  schema in prompt, base model            (ceiling)
  B  no schema,        base model            (floor)
  C  no schema,        fine-tuned checkpoint (the curve)
  D  schema in prompt, fine-tuned checkpoint (the correct baseline)

The arm letter only names the condition; what actually varies is
  --with-tools / --no-tools    and    --model (base) vs --checkpoint (tinker://...)

Per query: render natively via the cookbook renderer, sample at temperature 0,
parse with the same renderer, score {parsed, tool_ok, schema_valid, args_exact}.
Also measures the per-model schema token cost = tokens(with tools) - tokens(without).

Usage:
  # arms A and B on the base model, natural + strong held-out sets
  python eval_arms.py --catalog tools/catalog_64.json --catalog-size 4 \\
      --queries queries --queries-strong queries_strong \\
      --model openai/gpt-oss-20b --arm A --arm B --out results

  # arm C on a checkpoint
  python eval_arms.py ... --model openai/gpt-oss-20b \\
      --checkpoint tinker://<run>/sampler_weights/final --arm C --out results

  # plumbing test, no API
  python eval_arms.py ... --provider mock
"""

from __future__ import annotations

import random
import argparse
import json
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SYSTEM_PROMPTS = {
    # v1: the prompt used for the 0.918 base_valid run
    "v1": (
        "You are an assistant that operates equipment and records by calling tools. "
        "When the user's request corresponds to a tool, respond with exactly one tool call "
        "with the correct arguments and no other text. Include only the parameters the user's "
        "request actually specifies; omit optional parameters the user did not mention."
    ),
    # v2: adds explicit-negatives handling; suspected of inducing bare-JSON output on GPT-OSS
    "v2": (
        "You are an assistant that operates equipment and records by calling tools. "
        "When the user's request corresponds to a tool, respond with exactly one tool call "
        "with the correct arguments and no other text. Include every parameter whose value the user "
        "specifies, including explicit negatives ('without X' or 'do not X' means false). Omit optional "
        "parameters the user does not mention at all."
    ),
    # v3: v1 wording plus a single negatives sentence, minimal change
    "v3": (
        "You are an assistant that operates equipment and records by calling tools. "
        "When the user's request corresponds to a tool, respond with exactly one tool call "
        "with the correct arguments and no other text. Include only the parameters the user's "
        "request actually specifies; omit optional parameters the user did not mention. "
        "A phrase like 'without X' or 'do not X' specifies X as false."
    ),
}
SYSTEM_PROMPT = SYSTEM_PROMPTS["v1"]

# (schema_in_prompt, expects_checkpoint)
ARMS = {"A": (True, False), "B": (False, False), "C": (False, True), "D": (True, True)}


def default_renderer(model: str) -> str | None:
    """Overrides of the cookbook recommendation, with the measured reason.

    gpt-oss: the recommended `gpt_oss_no_sysprompt` omits the Harmony system block; on the
    4-tool pilot that produced 9% malformed tool-call headers and 2.5% reasoning truncations.
    `gpt_oss_low_reasoning` adds the block with `Reasoning: low`: 0% and 0%, half the tokens.
    """
    if model.startswith("openai/gpt-oss"):
        return "gpt_oss_low_reasoning"
    return None


# --------------------------------------------------------------------------
# Scoring (pure, testable)
# --------------------------------------------------------------------------

def _coerce(spec: dict, v: Any) -> Any:
    t = spec.get("type")
    try:
        if t == "integer":
            if isinstance(v, bool):  # bool is a subclass of int
                return v
            if isinstance(v, float) and v.is_integer():
                return int(v)
            if isinstance(v, str) and re.fullmatch(r"-?\d+", v.strip()):
                return int(v)
            return v
        if t == "boolean":
            if isinstance(v, str) and v.strip().lower() in ("true", "false"):
                return v.strip().lower() == "true"
            return v
        if t == "string":
            v = v if isinstance(v, str) else str(v)
            return v.strip().rstrip(".!?").strip()   # sentence-final punctuation belongs to the sentence
    except Exception:
        return v
    return v


def score_call(tool: dict, truth: dict, pred_name: str | None, pred_args: Any) -> dict:
    """truth = {"name":..., "arguments": {...}} ; pred_args is parsed dict or None."""
    props = tool["mcp"]["inputSchema"]["properties"]
    required = set(tool["mcp"]["inputSchema"]["required"])
    out = {"parsed": pred_name is not None and isinstance(pred_args, dict),
           "tool_ok": pred_name == truth["name"],
           "schema_valid": False, "args_exact": False, "args_lenient": False,
           "extra_keys": [], "missing_required": [], "bad_values": []}
    if not out["parsed"] or not out["tool_ok"]:
        return out
    args = {k: _coerce(props[k], v) if k in props else v for k, v in pred_args.items()}
    # gold coerced too, so "28" vs 28 still matches
    truth = dict(truth); truth["arguments"] = {k: (_coerce(props[k], v) if k in props else v)
                                                for k, v in truth["arguments"].items()}
    out["extra_keys"] = sorted(k for k in args if k not in props)
    out["missing_required"] = sorted(k for k in required if k not in args)
    for k, v in args.items():
        if k not in props:
            continue
        spec = props[k]
        if "enum" in spec and v not in spec["enum"]:
            out["bad_values"].append(k)
        elif spec["type"] == "integer" and not (isinstance(v, int) and not isinstance(v, bool)):
            out["bad_values"].append(k)
        elif spec["type"] == "boolean" and not isinstance(v, bool):
            out["bad_values"].append(k)
        elif spec["type"] == "string" and not isinstance(v, str):
            out["bad_values"].append(k)
    out["schema_valid"] = not (out["extra_keys"] or out["missing_required"] or out["bad_values"])
    out["args_exact"] = out["schema_valid"] and args == truth["arguments"]
    # lenient: a truth-absent optional may be present with any valid value
    lenient_args = {k: v for k, v in args.items() if k in truth["arguments"] or k in required}
    out["args_lenient"] = out["schema_valid"] and lenient_args == truth["arguments"]
    return out


def parse_args_str(s: Any) -> Any:
    if isinstance(s, dict):
        return s
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_catalog(path: str, size: int) -> list[dict]:
    cat = json.loads(Path(path).read_text())
    tools = cat["tools"][:size]
    for t in tools:
        t["_spec"] = {"name": t["mcp"]["name"], "description": t["mcp"]["description"],
                      "parameters": t["mcp"]["inputSchema"]}
    return tools


def load_queries(qdir: str, tools: list[dict], split: str) -> list[dict]:
    rows = []
    names = {t["mcp"]["name"] for t in tools}
    for t in tools:
        f = Path(qdir) / f"{t['mcp']['name']}.jsonl"
        if not f.exists():
            print(f"  warning: no queries for {t['mcp']['name']} in {qdir}", file=sys.stderr)
            continue
        for line in open(f):
            r = json.loads(line)
            if r["split"] == split and r["tool"] in names:
                rows.append(r)
    return rows


# --------------------------------------------------------------------------
# Sampler backends
# --------------------------------------------------------------------------

class TinkerBackend:
    def __init__(self, model: str, checkpoint: str | None, max_tokens: int, system_prompt: str = SYSTEM_PROMPT,
                 renderer_name: str | None = None, pin_date: str | None = None):
        self.system_prompt = system_prompt
        import tinker
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        self.tinker = tinker
        svc = tinker.ServiceClient()
        self.svc = svc
        if checkpoint:
            self.sampler = svc.create_sampling_client(model_path=checkpoint)
        else:
            self.sampler = svc.create_sampling_client(base_model=model)
        tok = get_tokenizer(model)
        self.tok = tok
        rname = renderer_name or default_renderer(model) or model_info.get_recommended_renderer_name(model)
        print(f"renderer: {rname}", file=sys.stderr)
        self.renderer = renderers.get_renderer(rname, tok, model_name=model)
        self.renderer_name = rname
        if pin_date and hasattr(self.renderer, "current_date"):
            self.renderer.current_date = pin_date
            print(f"renderer.current_date pinned to {pin_date}", file=sys.stderr)
        self.stop = self.renderer.get_stop_sequences()
        self.params = tinker.SamplingParams(max_tokens=max_tokens, temperature=0.0, stop=self.stop)

    def close(self):
        for obj in (self.sampler, getattr(self, "svc", None)):
            for m in ("close", "shutdown", "aclose"):
                fn = getattr(obj, m, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
                    break

    def prompt(self, tools: list[dict], with_tools: bool, query: str):
        specs = [t["_spec"] for t in tools] if with_tools else []
        prefix = self.renderer.create_conversation_prefix_with_tools(specs, system_prompt=self.system_prompt)
        msgs = prefix + [{"role": "user", "content": query}]
        return self.renderer.build_generation_prompt(msgs)

    def n_tokens(self, model_input) -> int:
        return model_input.length

    def submit(self, model_input):
        return self.sampler.sample(prompt=model_input, num_samples=1, sampling_params=self.params)

    def parse(self, fut) -> tuple[str | None, Any, dict]:
        resp = fut.result()
        seq = resp.sequences[0]
        msg, term = self.renderer.parse_response(list(seq.tokens))
        meta = {"completion_tokens": len(seq.tokens), "stop_reason": str(seq.stop_reason),
                "termination": str(term), "cache_hit_tokens": resp.prompt_cache_hit_tokens}
        tcs = msg.get("tool_calls") or []
        if not tcs:
            txt = _text(msg)
            meta["raw_text"] = txt[:300]
            j = parse_args_str(txt.strip())
            meta["json_in_text"] = isinstance(j, dict)
            if isinstance(j, dict):
                meta["json_in_text_args"] = j
                try:
                    dec = self.tok.decode(list(seq.tokens), skip_special_tokens=False)
                    m = re.search(r"functions\.([A-Za-z0-9_]+)", dec)
                    meta["recovered_name"] = m.group(1) if m else None
                except Exception:
                    meta["recovered_name"] = None
            try:  # full sequence incl. special tokens, so channel headers are visible
                meta["raw_decoded"] = self.tok.decode(list(seq.tokens), skip_special_tokens=False)[:4000]
            except Exception:
                pass
            meta["unparsed"] = [u.model_dump() if hasattr(u, "model_dump") else str(u)
                                for u in (msg.get("unparsed_tool_calls") or [])][:2]
            c = msg.get("content")
            meta["content_parts"] = ([getattr(p_, "type", type(p_).__name__) for p_ in c][:6]
                                     if isinstance(c, list) else "str")
            return None, None, meta
        tc = tcs[0]
        meta["n_tool_calls"] = len(tcs)
        return tc.function.name, parse_args_str(tc.function.arguments), meta


def _text(msg: dict) -> str:
    c = msg.get("content", "")
    if isinstance(c, str):
        return c
    return " ".join(p.get("text", "") for p in c if isinstance(p, dict))


class MockBackend:
    """No API. With tools: correct call 90% of the time, else a plausible error.
    Without tools: parse failure 85%, wrong tool name otherwise. Exercises every score path."""

    def __init__(self, tools: list[dict], seed: int = 0):
        import random
        self.rng = random.Random(seed)
        self.tools = tools
        self.renderer_name = "mock"

    def prompt(self, tools, with_tools, query):
        return {"with_tools": with_tools, "query": query, "len": 40 + (900 if with_tools else 0)}

    def n_tokens(self, mi): return mi["len"]

    def submit(self, mi): return mi

    def parse(self, mi):
        r = self.rng.random()
        truth = mi["_truth"]
        tool = next(t for t in self.tools if t["mcp"]["name"] == truth["name"])
        props = tool["mcp"]["inputSchema"]["properties"]
        meta = {"completion_tokens": 40, "stop_reason": "stop", "termination": "STOP_SEQUENCE"}
        if not mi["with_tools"]:
            if r < 0.85:
                return None, None, {**meta, "raw_text": "I don't have a tool for that."}
            return "some_other_tool", {}, meta
        if r < 0.90:
            return truth["name"], dict(truth["arguments"]), meta
        args = dict(truth["arguments"])
        k = self.rng.choice(list(args) or list(props))
        if r < 0.94 and "enum" in props[k]:
            args[k] = "bogus_value"
        elif r < 0.97:
            args.pop(k, None)
        else:
            args["unexpected"] = 1
        return truth["name"], args, meta


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

_STOP = {"the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "with", "this", "that", "when",
         "please", "can", "you", "could", "would", "we", "it", "is", "are", "be", "by", "at", "from", "as", "my", "our"}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in _STOP}


def _tool_index(tools):
    """Bag of words per tool (name, description, param names/descriptions, enum values) for retrieval."""
    idx = {}
    for t in tools:
        m = t["mcp"]
        ws = _words(m["name"].replace("_", " ")) | _words(m["description"])
        for pn, spec in m["inputSchema"]["properties"].items():
            ws |= _words(pn.replace("_", " ")) | _words(spec.get("description", ""))
            ws |= {w for v in spec.get("enum", []) for w in _words(v.replace("_", " "))}
        idx[m["name"]] = ws
    return idx


def retrieve(idx, query: str, k: int) -> list[str]:
    """Top-k tools by IDF-weighted word overlap with the query (a simple lexical retriever)."""
    q = _words(query)
    df = {}
    for ws in idx.values():
        for w in ws:
            df[w] = df.get(w, 0) + 1
    n = len(idx)
    scored = sorted(((sum(math.log(1 + n / df[w]) for w in q & ws), name) for name, ws in idx.items()),
                    key=lambda x: -x[0])
    return [name for _, name in scored[:k]]


def run_arm(backend, tools, rows, arm: str, set_name: str, out_f, max_inflight: int,
            context: str = "catalog", retrieve_k: int = 4, schema_order: int | None = None,
            schema_rotate: bool = False) -> dict:
    """context: 'catalog' = every schema; 'single' = only the query's own tool; 'retrieve' = top-k schemas
    by lexical retrieval (a realistic no-adapter alternative). schema_order: shuffle the catalog order in
    the rendered prefix with this seed (position-confound control); queries and scoring unchanged."""
    with_tools, _ = ARMS[arm]
    by_name = {t["mcp"]["name"]: t for t in tools}
    if schema_order is not None:
        tools = list(tools)
        if schema_rotate:
            k = schema_order % len(tools)          # deterministic rotation: tool i moves to position (i+k) mod N
            tools = tools[-k:] + tools[:-k] if k else tools
        else:
            random.Random(schema_order).shuffle(tools)
    idx = _tool_index(tools) if context == "retrieve" else None
    retrieved_flags: dict[str, bool] = {}
    stats = {"retrieval_hit": 0, "n": 0, "parsed": 0, "tool_ok": 0, "schema_valid": 0, "args_exact": 0, "args_lenient": 0,
             "recovered_exact": 0, "json_in_text": 0, "json_content_exact": 0, "truncated": 0,
             "prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0}
    by_tool: dict[str, dict] = {}
    rw = {"n": 0, "args_exact": 0}
    truth_by_id = {r["query_id"]: r for r in rows}

    def one(r):
        if context == "single":
            ctx_tools = [by_name[r["tool"]]]
        elif context == "retrieve":
            names = retrieve(idx, r["query"], retrieve_k)
            retrieved_flags[r["query_id"]] = r["tool"] in names
            ctx_tools = [by_name[n_] for n_ in names]
        else:
            ctx_tools = tools
        mi = backend.prompt(ctx_tools, with_tools, r["query"])
        if isinstance(mi, dict):
            mi["_truth"] = r["call"]
        n_prompt = backend.n_tokens(mi)
        fut = backend.submit(mi)
        return r["query_id"], n_prompt, fut

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_inflight) as ex:
        futs = [ex.submit(one, r) for r in rows]
        for f in as_completed(futs):
            qid, n_prompt, fut = f.result()
            r = truth_by_id[qid]
            tool = next(t for t in tools if t["mcp"]["name"] == r["tool"])
            try:
                name, args, meta = backend.parse(fut)
                err = None
            except Exception as e:
                name, args, meta, err = None, None, {}, f"{type(e).__name__}: {str(e)[:200]}"
            sc = score_call(tool, r["call"], name, args)
            # recovered: malformed envelope but the function name is present in the header
            if meta.get("json_in_text") and meta.get("recovered_name"):
                sc["recovered_exact"] = score_call(tool, r["call"], meta["recovered_name"],
                                                   meta.get("json_in_text_args"))["args_exact"]
            else:
                sc["recovered_exact"] = sc["args_exact"]   # strict result carries over
            sc["json_content_exact"] = sc["recovered_exact"] and bool(meta.get("json_in_text"))
            row = {"arm": arm, "set": set_name, "query_id": qid, "tool": r["tool"], "query": r["query"],
                   "persona": r.get("persona"), "echo_score": r.get("echo_score"),
                   "rewritten": r.get("rewritten"),   # strong set: True if text differs from natural
                   "pred_name": name, "pred_args": args, "truth": r["call"], "score": sc,
                   "prompt_tokens": n_prompt, **meta, "error": err}
            if context == "retrieve":
                row["retrieved"] = retrieved_flags.get(qid, False)
                stats["retrieval_hit"] += int(row["retrieved"])
            out_f.write(json.dumps(row) + "\n")
            stats["n"] += 1
            for k in ("parsed", "tool_ok", "schema_valid", "args_exact", "args_lenient", "recovered_exact"):
                stats[k] += int(sc[k])
            stats["json_in_text"] += int(bool(meta.get("json_in_text")))
            stats["json_content_exact"] += int(bool(sc.get("json_content_exact")))
            stats["truncated"] += int("length" in str(meta.get("stop_reason", "")).lower())
            stats["prompt_tokens"] += n_prompt
            stats["completion_tokens"] += meta.get("completion_tokens", 0) or 0
            stats["cache_hit_tokens"] += meta.get("cache_hit_tokens", 0) or 0
            bt = by_tool.setdefault(r["tool"], {"n": 0, "args_exact": 0})
            bt["n"] += 1; bt["args_exact"] += int(sc["args_exact"])
            if r.get("rewritten"):
                rw["n"] += 1; rw["args_exact"] += int(sc["args_exact"])
    n = max(stats["n"], 1)
    summary = {k: (v / n if k in ("parsed", "tool_ok", "schema_valid", "args_exact", "args_lenient",
                                   "recovered_exact", "json_in_text", "json_content_exact", "truncated",
                                   "retrieval_hit") else v)
               for k, v in stats.items()}
    if context != "retrieve":
        summary.pop("retrieval_hit", None)
    summary["by_tool_args_exact"] = {t: round(v["args_exact"] / v["n"], 3) for t, v in by_tool.items()}
    summary["rewritten_n"] = rw["n"]
    summary["rewritten_args_exact"] = round(rw["args_exact"] / rw["n"], 3) if rw["n"] else None
    summary["seconds"] = round(time.time() - t0, 1)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--catalog-size", type=int, required=True)
    ap.add_argument("--queries", required=True, help="natural-mode dir")
    ap.add_argument("--queries-strong", default=None, help="strong-mode dir (optional)")
    ap.add_argument("--split", default="held")
    ap.add_argument("--model", required=True, help="base model name (also selects renderer/tokenizer)")
    ap.add_argument("--checkpoint", default=None, help="tinker://... sampler weights for arms C/D")
    ap.add_argument("--arm", action="append", choices=list(ARMS), required=True)
    ap.add_argument("--provider", default="tinker", choices=["tinker", "mock"])
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-inflight", type=int, default=32)
    ap.add_argument("--tag", default=None, help="label for this run (e.g. seed0_ex100)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N queries per set (smoke test)")
    ap.add_argument("--prompt", default="v1", choices=list(SYSTEM_PROMPTS), help="system prompt version")
    ap.add_argument("--context", default="catalog", choices=["catalog", "single", "retrieve"],
                    help="single: only the query's own schema (argument-filling baseline); "
                         "retrieve: top-k schemas by lexical retrieval (realistic no-adapter alternative)")
    ap.add_argument("--retrieve-k", type=int, default=4)
    ap.add_argument("--schema-order", type=int, default=None,
                    help="reorder the catalog in the rendered prefix: shuffle with this seed, or rotate by this many "
                         "positions with --schema-rotate (queries and scoring unchanged)")
    ap.add_argument("--schema-rotate", action="store_true",
                    help="treat --schema-order as a rotation amount; rotations of N/4, N/2, 3N/4 place every tool in "
                         "every quartile once across four runs (counterbalanced position design)")
    ap.add_argument("--date", default=None, help="pin the renderer's Current date (YYYY-MM-DD) for reproducible prompts")
    ap.add_argument("--renderer", default=None,
                    help="override renderer name, e.g. gpt_oss_low_reasoning (default: cookbook recommendation)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    for arm in args.arm:
        if ARMS[arm][1] and not args.checkpoint and args.provider != "mock":
            raise SystemExit(f"arm {arm} needs --checkpoint")

    tools = load_catalog(args.catalog, args.catalog_size)
    sets = {"natural": load_queries(args.queries, tools, args.split)}
    if args.queries_strong:
        sets["strong"] = load_queries(args.queries_strong, tools, args.split)
    if args.limit:
        sets = {k: v[: args.limit] for k, v in sets.items()}
    for k, v in sets.items():
        print(f"{k}: {len(v)} queries", file=sys.stderr)

    backend = MockBackend(tools) if args.provider == "mock" else \
        TinkerBackend(args.model, args.checkpoint, args.max_tokens, SYSTEM_PROMPTS[args.prompt], args.renderer, args.date)

    # schema token cost for this model + catalog (exact, no sampling)
    probe_q = next(iter(sets["natural"]))["query"] if sets["natural"] else "hello"
    with_t = backend.n_tokens(backend.prompt(tools, True, probe_q))
    without = backend.n_tokens(backend.prompt(tools, False, probe_q))
    schema_tokens = with_t - without

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("base" if not args.checkpoint else re.sub(r"[^A-Za-z0-9]+", "_", args.checkpoint)[-40:])
    if args.limit:
        tag += f"_smoke{args.limit}"
    tag += f"_p{args.prompt}"
    if args.context == "single":
        tag += "_single"
    elif args.context == "retrieve":
        tag += f"_retrieve{args.retrieve_k}"
    if args.schema_order is not None:
        tag += (f"_rot{args.schema_order}" if args.schema_rotate else f"_order{args.schema_order}")
    if args.renderer:   # an explicit renderer override never shares files with the default-renderer run
        tag += "_r" + args.renderer.replace("gpt_oss_", "")
    stem = f"{re.sub(r'[^A-Za-z0-9]+', '_', args.model)}__cat{args.catalog_size}__{tag}"
    rows_f = open(out / f"{stem}.rows.jsonl", "a")

    summaries: dict[str, Any] = {
        "model": args.model, "checkpoint": args.checkpoint, "renderer": backend.renderer_name,
        "prompt_version": args.prompt, "context": args.context, "pinned_date": args.date,
        "retrieve_k": args.retrieve_k if args.context == "retrieve" else None, "schema_order": args.schema_order,
        "schema_rotate": args.schema_rotate,
        "catalog": args.catalog, "catalog_size": args.catalog_size, "split": args.split,
        "schema_tokens": schema_tokens, "prompt_tokens_without_schema": without,
        "closed_vocab": sum(t["closed_vocab"] for t in tools), "arms": {},
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for arm in args.arm:
        for set_name, rows in sets.items():
            if not rows:
                continue
            print(f"\n== arm {arm} / {set_name} ==", file=sys.stderr)
            s = run_arm(backend, tools, rows, arm, set_name, rows_f, args.max_inflight, args.context,
                        args.retrieve_k, args.schema_order, args.schema_rotate)
            summaries["arms"][f"{arm}/{set_name}"] = s
            rh = f"  retrieval_hit={s['retrieval_hit']:.3f}" if "retrieval_hit" in s else ""
            print(f"  n={s['n']}{rh}  parsed={s['parsed']:.3f}  tool_ok={s['tool_ok']:.3f}  "
                  f"schema_valid={s['schema_valid']:.3f}  args_exact={s['args_exact']:.3f}  "
                  f"lenient={s['args_lenient']:.3f}  json_in_text={s['json_in_text']:.3f} "
                  f"(content ok {s['json_content_exact']:.3f})  truncated={s['truncated']:.3f}  "
                  f"({s['seconds']}s)", file=sys.stderr)
    rows_f.close()
    (out / f"{stem}.summary.json").write_text(json.dumps(summaries, indent=2))
    if hasattr(backend, "close"):
        backend.close()

    print(f"\nschema tokens for this model/catalog: {schema_tokens}  (prompt without schema: {without})")
    for k, s in summaries["arms"].items():
        n = max(s["n"], 1)
        rw_note = f"  [rewritten-only n={s['rewritten_n']} exact={s['rewritten_args_exact']:.3f}]" if s.get("rewritten_n") else ""
        print(f"  {k:16s} exact={s['args_exact']:.3f}  recovered={s['recovered_exact']:.3f}  lenient={s['args_lenient']:.3f}  tool_ok={s['tool_ok']:.3f}  parsed={s['parsed']:.3f}{rw_note}  |  "
              f"prompt/req={s['prompt_tokens']/n:.0f}  cache_hit/req={s['cache_hit_tokens']/n:.0f}  "
              f"completion/req={s['completion_tokens']/n:.0f}")
    print("  (with-tools minus no-tools prompt/req is the per-request schema cost the adapter would remove)")
    print(f"wrote {out}/{stem}.summary.json")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    import os
    os._exit(rc or 0)   # skip interpreter teardown: SDK poller otherwise complains or crashes