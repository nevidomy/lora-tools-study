#!/usr/bin/env python3
"""Emit a self-contained repro script for the GPT-OSS renderer report.

Embeds the 4 real ToolSpecs and the queries that failed under gpt_oss_no_sysprompt in the
controlled run (plus passing ones for contrast), then runs all three renderers on them.

Usage:
  python make_repro.py --catalog tools/catalog_64.json \
      --rows results/openai_gpt_oss_20b__cat4__rab_pv1_rno_sysprompt.rows.jsonl \
      --out repro_gptoss_renderer.py
  python repro_gptoss_renderer.py --selftest    # classifier only, no API
  python repro_gptoss_renderer.py               # needs TINKER_API_KEY; a few cents
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = r'''#!/usr/bin/env python3
"""Repro: gpt-oss-20b under tinker-cookbook's `gpt_oss_no_sysprompt` renderer emits tool calls on the
`final` channel with `<|constrain|>functions.NAME` in place of a `to=` recipient, and sometimes reasons
to `max_tokens` without calling. The `_reasoning` renderers (standard Harmony system block) substantially
reduce these failures (0/200 and 1/200 malformed, 0/200 capped, in the controlled comparison).

Requires:  pip install tinker==0.27.1 tinker-cookbook==0.5.7 ; export TINKER_API_KEY=...
Runs the same __N_QUERIES__ requests (__N_FAIL__ that failed in the original run, __N_PASS__ that passed)
under three renderers at temperature 0 with `current_date` pinned. Prints per-renderer counts grouped
by the request's original outcome, correctness against the expected call, and one malformed example
in full. Cost: a few cents.

Outputs varied across identical-prompt, temperature-0 runs in the original measurements, so the
originally-failing requests are not expected to fail identically; the failure mode should recur.

`python __SCRIPT__ --selftest` exercises the classifier on fabricated strings, no API.
"""
import json, re, sys

MODEL = "openai/gpt-oss-20b"
DATE = "__DATE__"
MAX_TOKENS = 1024
SYSTEM_PROMPT = __SYSTEM_PROMPT__
TOOLS = json.loads(__TOOLS_JSON__)
# [{"id", "query", "expected": {"name", "arguments"}, "orig": "fail"|"pass"}]
QUERIES = json.loads(__QUERIES_JSON__)
RENDERERS = ["gpt_oss_no_sysprompt", "gpt_oss_low_reasoning", "gpt_oss_medium_reasoning"]
END_TOKENS = ("<|end|>", "<|call|>", "<|return|>")


def harmony_messages(dec):
    """Split decoded Harmony text into (header, body, terminator), mirroring the cookbook parser:
    header = text from the last <|start|> (or segment start) up to <|message|>; body runs to the first
    <|end|>/<|call|>/<|return|>; terminator is '' if unterminated."""
    out, i = [], 0
    while True:
        m = dec.find("<|message|>", i)
        if m == -1:
            break
        hs = dec.rfind("<|start|>", i, m)
        header = dec[(i if hs == -1 else hs):m]
        bs = m + len("<|message|>")
        be, term = len(dec), ""
        for t in END_TOKENS:
            k = dec.find(t, bs)
            if k != -1 and k < be:
                be, term = k, t
        out.append((header, dec[bs:be], term))
        i = be + len(term) if term else len(dec)
    return out


def _type_ok(spec, v):
    t = spec.get("type")
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "boolean":
        return isinstance(v, bool)
    if t == "string":
        return isinstance(v, str) and ("enum" not in spec or v in spec["enum"])
    return True


def _strict_equal(a, b):
    """Value AND type equality; False != 0, "28" != 28."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_strict_equal(a[k], b[k]) for k in a)
    return a == b


def args_match(tool_name, args, expected):
    """Exact match: same tool, same keys, every value schema-typed and strictly equal. No coercion."""
    if tool_name != expected["name"] or not isinstance(args, dict):
        return False
    props = next((t["parameters"]["properties"] for t in TOOLS if t["name"] == tool_name), {})
    if args.keys() != expected["arguments"].keys():
        return False
    return all(k in props and _type_ok(props[k], args[k]) for k in args) \
        and _strict_equal(args, expected["arguments"])


def classify(dec, stop_reason, n_tokens, msg, expected):
    """Disjoint categories. Returns (category, detail)."""
    tcs = msg.get("tool_calls") or []
    if len(tcs) > 1:
        return "multiple_calls", "%d calls" % len(tcs)
    if tcs:
        tc = tcs[0]
        try:
            a = json.loads(tc.function.arguments)
        except Exception:
            a = None
        return ("tool_call_correct" if args_match(tc.function.name, a, expected) else "tool_call_wrong"), None
    msgs = harmony_messages(dec)
    if stop_reason == "length" and n_tokens >= MAX_TOKENS:
        last_ch = None
        if msgs:
            m = re.search(r"<\|channel\|>\s*([a-z]+)", msgs[-1][0])
            last_ch = m.group(1) if m else None
        in_analysis = bool(msgs) and last_ch == "analysis" and msgs[-1][2] == ""
        return "hit_max_tokens", ("still in analysis" if in_analysis else "last channel=%s" % last_ch)
    if msgs:
        header, body, term = msgs[-1]
        ch = re.search(r"<\|channel\|>\s*([a-z]+)", header)
        misplaced = re.search(r"<\|constrain\|>\s*functions\.([A-Za-z0-9_]+)", header)
        if ch and ch.group(1) == "final" and misplaced and not re.search(r"\bto=", header):
            try:
                body_json = json.loads(body.strip())
            except Exception:
                body_json = None
            if isinstance(body_json, dict):
                ok = args_match(misplaced.group(1), body_json, expected)
                return "bare_json_final_channel", ("args correct" if ok else "args wrong")
    return "other_no_call", (msgs[-1][0][-80:] if msgs else dec[-80:])


def selftest():
    exp = {"name": "t", "arguments": {"a": 1}}
    gb = '{"a": 1}'
    cases = [
        ("final + misplaced + json + <|return|>",
         '<|channel|>analysis<|message|>x<|end|><|start|>assistant<|channel|>final <|constrain|>functions.t<|message|>' + gb + '<|return|>',
         "stop", 20, "bare_json_final_channel"),
        ("same, whitespace after tokens",
         '<|channel|>analysis<|message|>x<|end|><|start|>assistant<|channel|>final <|constrain|> functions.t<|message|> ' + gb + ' <|return|>',
         "stop", 20, "bare_json_final_channel"),
        ("commentary channel (not the reported mode)",
         '<|start|>assistant<|channel|>commentary <|constrain|>functions.t<|message|>' + gb + '<|end|>',
         "stop", 20, "other_no_call"),
        ("final + misplaced but broken json",
         '<|start|>assistant<|channel|>final <|constrain|>functions.t<|message|>{broken<|return|>',
         "stop", 20, "other_no_call"),
        ("final, has to= recipient (well-formed-ish, no tool_calls given)",
         '<|start|>assistant to=functions.t<|channel|>final<|message|>' + gb + '<|return|>',
         "stop", 20, "other_no_call"),
        ("short response, no stop marker, NOT at cap",
         '<|channel|>analysis<|message|>thinking', "stop", 10, "other_no_call"),
        ("at cap, still in analysis",
         '<|channel|>analysis<|message|>' + 'thinking ' * 50, "length", MAX_TOKENS, "hit_max_tokens"),
        ("final + misplaced but a to=python recipient present",
         '<|start|>assistant to=python<|channel|>final <|constrain|>functions.t<|message|>' + gb + '<|return|>',
         "stop", 20, "other_no_call"),
    ]
    bad = 0
    for name, dec, sr, n, want in cases:
        got, _ = classify(dec, sr, n, {}, exp)
        bad += got != want
        print("  %s  %-52s -> %s" % ("ok " if got == want else "BAD", name, got))
    # strict scoring probes (no coercion, no punctuation stripping, bool-aware)
    global TOOLS
    saved = TOOLS
    TOOLS = [{"name": "t", "parameters": {"properties": {"n": {"type": "integer"}, "b": {"type": "boolean"}, "s": {"type": "string"}}}}]
    e = {"name": "t", "arguments": {"n": 28, "b": False, "s": "SPOT-15"}}
    probes = [
        ("exact",                    {"n": 28, "b": False, "s": "SPOT-15"}, True),
        ('"28" for integer 28',      {"n": "28", "b": False, "s": "SPOT-15"}, False),
        ('"false" for boolean',      {"n": 28, "b": "false", "s": "SPOT-15"}, False),
        ("0 for boolean False",      {"n": 28, "b": 0, "s": "SPOT-15"}, False),
        ('"SPOT-15!" for "SPOT-15"', {"n": 28, "b": False, "s": "SPOT-15!"}, False),
        ("extra key",                {"n": 28, "b": False, "s": "SPOT-15", "x": 1}, False),
    ]
    for name, a, want in probes:
        got = args_match("t", a, e)
        bad += got != want
        print("  %s  strict: %-46s -> %s" % ("ok " if got == want else "BAD", name, got))
    TOOLS = saved
    class _F: pass
    def _mk(name, args):
        tc = _F(); tc.function = _F(); tc.function.name = name; tc.function.arguments = json.dumps(args); return tc
    got, _ = classify("", "stop", 5, {"tool_calls": [_mk("t", {"a": 1}), _mk("t", {"a": 1})]}, exp)
    bad += got != "multiple_calls"
    print("  %s  %-52s -> %s" % ("ok " if got == "multiple_calls" else "BAD", "two tool calls, first correct", got))
    print("selftest:", "PASS" if not bad else "FAIL (%d)" % bad)
    return 0 if not bad else 1


def main():
    if "--selftest" in sys.argv:
        return selftest()
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    svc = tinker.ServiceClient()
    sampler = svc.create_sampling_client(base_model=MODEL)
    tok = get_tokenizer(MODEL)
    example_shown = False
    for rname in RENDERERS:
        r = renderers.get_renderer(rname, tok, model_name=MODEL)
        if hasattr(r, "current_date"):
            r.current_date = DATE
        params = tinker.SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, stop=r.get_stop_sequences())
        prefix = r.create_conversation_prefix_with_tools(TOOLS, system_prompt=SYSTEM_PROMPT)
        futs = [sampler.sample(prompt=r.build_generation_prompt(prefix + [{"role": "user", "content": q["query"]}]),
                               num_samples=1, sampling_params=params) for q in QUERIES]
        counts = {"fail": {}, "pass": {}}
        for q, f in zip(QUERIES, futs):
            seq = f.result().sequences[0]
            toks = list(seq.tokens)
            msg, term = r.parse_response(toks)
            dec = tok.decode(toks, skip_special_tokens=False)
            cat, detail = classify(dec, str(seq.stop_reason), len(toks), msg, q["expected"])
            counts[q["orig"]][cat] = counts[q["orig"]].get(cat, 0) + 1
            if cat == "bare_json_final_channel" and not example_shown:
                example_shown = True
                print("\n--- example malformed response (%s, query %s, %s) ---" % (rname, q["id"], detail))
                print(repr(dec))
                print("parse_response ->", str(term), "| tool_calls:", bool(msg.get("tool_calls")),
                      "| unparsed_tool_calls:", bool(msg.get("unparsed_tool_calls")))
        print("\n" + rname)
        for orig in ("fail", "pass"):
            n = sum(1 for q in QUERIES if q["orig"] == orig)
            print("  originally %-4s (n=%2d): " % (orig, n) + "  ".join("%s=%d" % kv for kv in sorted(counts[orig].items())))
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    import os; os._exit(rc or 0)
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--rows", required=True, help="rows file from the no_sysprompt controlled run")
    ap.add_argument("--n-pass", type=int, default=12)
    ap.add_argument("--date", default="2026-09-12")
    ap.add_argument("--out", default="repro_gptoss_renderer.py")
    args = ap.parse_args()

    cat = json.loads(Path(args.catalog).read_text())["tools"][:4]
    tools = [{"name": t["mcp"]["name"], "description": t["mcp"]["description"],
              "parameters": t["mcp"]["inputSchema"]} for t in cat]
    rows = [json.loads(l) for l in open(args.rows)]
    rows = [r for r in rows if r["arm"] == "A" and r["set"] == "natural"]
    fails = [r for r in rows if not r["score"]["parsed"]]
    passes = [r for r in rows if r["score"]["args_exact"]][: args.n_pass]
    queries = ([{"id": r["query_id"], "query": r["query"], "expected": r["truth"], "orig": "fail"} for r in fails]
               + [{"id": r["query_id"], "query": r["query"], "expected": r["truth"], "orig": "pass"} for r in passes])

    from eval_arms import SYSTEM_PROMPTS  # noqa: E402  (must match what the runs used)
    script = (TEMPLATE
              .replace("__SCRIPT__", Path(args.out).name)
              .replace("__N_QUERIES__", str(len(queries)))
              .replace("__N_FAIL__", str(len(fails)))
              .replace("__N_PASS__", str(len(passes)))
              .replace("__DATE__", args.date)
              .replace("__SYSTEM_PROMPT__", repr(SYSTEM_PROMPTS["v1"]))
              .replace("__TOOLS_JSON__", repr(json.dumps(tools)))
              .replace("__QUERIES_JSON__", repr(json.dumps(queries))))
    Path(args.out).write_text(script)
    print(f"wrote {args.out}: {len(tools)} tools, {len(fails)} failing + {len(passes)} passing queries embedded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())