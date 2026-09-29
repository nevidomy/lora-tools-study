#!/usr/bin/env python3
"""Generate (query -> tool call) pairs from a validated catalog.

Call-first: sample a concrete call from the schema, then ask the LLM to write
the user request that would produce exactly that call. Ground truth is
therefore exact by construction; the model only supplies surface language.

Per tool:
  - string params get a value pool (one LLM call per tool) so string values
    can be embedded verbatim and exact-matched
  - N calls are sampled (optionals included with p=0.5)
  - queries are requested in batches of K per API call, schema sent once
  - each query gets a persona, rotated deterministically and logged
  - local post-checks: ints and strings present verbatim, enum values NOT
    echoed, tool name not mentioned; failures logged with an echo score

Splits are nested: held-out is the last HELD per tool; train is the rest, and
train subsets of size m are prefixes, so examples-per-tool sweeps are nested.

Usage:
  python queries_gen.py --catalog tools/catalog_64.json --tools 4 --per-tool 150 \
      --provider openrouter --model x-ai/grok-4.3 --reasoning off
  python queries_gen.py --catalog tools/catalog_64.json --provider mock
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PERSONAS = [
    "terse: under 12 words, no pleasantries",
    "chatty: two or three sentences, includes a bit of small talk first",
    "context-heavy: mentions one or two irrelevant details before the actual request",
    "formal: polite, complete sentences, no contractions",
    "typos: two or three realistic typos in ordinary connective words only (never inside quoted values, numbers, names, or any word that identifies a choice or setting), otherwise natural",
    "indirect: describes the goal rather than naming the action",
    "question-form: phrased as a question",
    "imperative: a direct command, present tense",
]

HELD_DEFAULT = 50

READ_VERBS = {"query", "list", "inspect", "estimate"}


def intent_block(tool: dict) -> str:
    verb = tool["skeleton"]["verb"]
    if verb in READ_VERBS:
        return (f"- INTENT: this tool READS information (verb: {verb}). Every message must ask to "
                f"retrieve, look up, or check something. A message asking to change, create, "
                f"schedule, or record something is WRONG.")
    return (f"- INTENT: this tool PERFORMS an action that changes state (verb: {verb}). Every "
            f"message must ask to carry out that action now. A message asking to check, look up, "
            f"confirm, or find out whether something was done is WRONG.")
INT_RANGES = [(1, 12), (1, 60), (5, 500), (100, 5000)]

SYSTEM_PROMPT = (
    "You write realistic user requests to a software agent. You output strictly "
    "valid JSON and nothing else: no prose, no code fences."
)


# --------------------------------------------------------------------------
# Provider layer (same shape as tools_gen.py)
# --------------------------------------------------------------------------

PROVIDERS = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "tinker": ("https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1", "TINKER_API_KEY"),
    "local": ("http://localhost:8000/v1", None),
}


class LLM:
    def __init__(self, provider: str, model: str, temperature: float, reasoning: str):
        self.provider, self.model, self.temperature = provider, model, temperature
        self.extra_body: dict[str, Any] = {}
        if reasoning == "off":
            self.extra_body = {"reasoning": {"enabled": False}}
        elif reasoning == "low":
            self.extra_body = {"reasoning": {"effort": "low"}}
        self.client = None
        if provider != "mock":
            from openai import OpenAI
            base_url, key_env = PROVIDERS[provider]
            if key_env and not os.environ.get(key_env):
                raise SystemExit(f"{key_env} not set")
            self.client = OpenAI(base_url=base_url, api_key=os.environ.get(key_env, "local"))

    def json_call(self, user_prompt: str, max_tokens: int, mock_fn=None) -> tuple[dict | None, dict]:
        prov: dict[str, Any] = {"provider": self.provider, "model": self.model,
                                "temperature": self.temperature,
                                "prompt_sha1": hashlib.sha1(user_prompt.encode()).hexdigest()[:12]}
        if self.provider == "mock":
            return mock_fn(), prov
        kwargs: dict[str, Any] = dict(
            model=self.model, temperature=self.temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user_prompt}],
        )
        if self.extra_body:
            kwargs["extra_body"] = dict(self.extra_body)
        raw = None
        json_mode = True
        for attempt in range(6):
            try:
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                else:
                    kwargs.pop("response_format", None)
                resp = self.client.chat.completions.create(**kwargs)
                raw = resp.choices[0].message.content or ""
                prov["response_model"] = getattr(resp, "model", None)
                prov["routed_provider"] = getattr(resp, "provider", None)
                prov["finish_reason"] = getattr(resp.choices[0], "finish_reason", None)
                u = getattr(resp, "usage", None)
                if u:
                    prov["prompt_tokens"] = getattr(u, "prompt_tokens", None)
                    prov["completion_tokens"] = getattr(u, "completion_tokens", None)
                break
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:200]}"
                prov.setdefault("errors", []).append(msg)
                status = getattr(e, "status_code", None)
                low = str(e).lower()
                if "reasoning" in low and ("mandatory" in low or "cannot be disabled" in low) \
                        and kwargs.get("extra_body", {}).get("reasoning", {}).get("enabled") is False:
                    kwargs["extra_body"] = {"reasoning": {"effort": "low"}}   # endpoint refuses off -> low
                    prov["reasoning_fallback"] = "low"
                    continue
                if json_mode and (status in (400, 422) or "response_format" in low):
                    json_mode = False          # provider rejects json mode; retry without, no sleep
                    continue
                if status in (429, 500, 502, 503, 504, 408) or status is None:
                    time.sleep(min(2 ** attempt, 30) * (1 + 0.3 * random.random()))
                    continue
                prov["error"] = msg
                return None, prov
        if raw is None:
            prov["error"] = "exhausted retries: " + (prov.get("errors") or ["?"])[-1]
            return None, prov
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.S)
        try:
            return json.loads(s), prov
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", s, flags=re.S)
            try:
                return (json.loads(m.group(0)) if m else None), prov
            except json.JSONDecodeError:
                prov["raw"] = raw[:1500]
                prov["error"] = "unparseable"
                return None, prov


# --------------------------------------------------------------------------
# Call sampling
# --------------------------------------------------------------------------

def string_pool_prompt(tool: dict, string_params: list[str], n: int) -> str:
    """n = total values requested; the caller splits them into disjoint train/held sets."""
    schema = tool["schema_text"]
    enum_words = sorted({w for sp in tool["mcp"]["inputSchema"]["properties"].values() if "enum" in sp
                         for v in sp["enum"] for w in re.split(r"[_\s]+", v) if len(w) > 3})
    avoid = (f"\nDo NOT use any of these words inside a value (they are enum choices of other parameters "
             f"and would cause ambiguity): {', '.join(enum_words)}\n") if enum_words else ""
    return f"""Here is a tool schema:
{schema}
{avoid}
For each of these string parameters, give {n} realistic, distinct example values a user
might actually specify. Values should be short (1-5 words), concrete, and varied, and must
match the parameter's ROLE so a reader could tell which parameter a value belongs to: a
person/reporter field gets person-like values (names, initials), a location/section field
gets place-like values, an identifier field gets codes. Values for different parameters of
this tool must not look alike. No generic placeholders like "example" or "test". Output JSON only:
{{"pools": {{{", ".join(f'"{p}": ["..."]' for p in string_params)}}}}}"""


def sample_call(tool: dict, pools: dict[str, dict[str, list[str]]], rng: random.Random, held: bool = False) -> dict:
    """pools[param] = {"train": [...], "held": [...]}; held-out strings never appear in training."""
    props = tool["mcp"]["inputSchema"]["properties"]
    required = set(tool["mcp"]["inputSchema"]["required"])
    args: dict[str, Any] = {}
    for pname, spec in props.items():
        if pname not in required and rng.random() < 0.5:
            continue
        t = spec["type"]
        if "enum" in spec:
            args[pname] = rng.choice(spec["enum"])
        elif t == "boolean":
            args[pname] = rng.random() < 0.5
        elif t == "integer":
            lo, hi = rng.choice(INT_RANGES)
            args[pname] = rng.randint(lo, hi)
        elif t == "string":
            entry = pools.get(pname) or {}
            pool = entry.get("held" if held else "train") or entry.get("train") or ["value"]
            args[pname] = rng.choice(pool)
    return {"name": tool["mcp"]["name"], "arguments": args}


# --------------------------------------------------------------------------
# Query generation
# --------------------------------------------------------------------------

def glossary_prompt(tool: dict) -> str:
    props = tool["mcp"]["inputSchema"]["properties"]
    lines = []
    for pname, spec in props.items():
        if "enum" in spec:
            lines.append(f"  {pname} ({spec['description']}): {', '.join(spec['enum'])}")
    return f"""Tool: {tool['mcp']['name']} — {tool['mcp']['description']}
Domain: {tool['skeleton']['domain']}

Enum parameters and their values:
{chr(10).join(lines)}

For EVERY value, write 3 short ways a real user would refer to that choice WITHOUT using any
word that appears in the value itself. Describe what it is, when it's used, or its effect —
e.g. for maintenance_access: "open it up so the technicians can get in"; for storm_lock: "batten
it down, there's weather coming". Each paraphrase 3-12 words, natural, unambiguous among the
other values of the same parameter.

Output JSON only: {{"glossary": {{"<param>": {{"<value>": ["...", "...", "..."]}}}}}}"""


def _value_words(v: str) -> set[str]:
    return {w for w in re.split(r"[_\s\-]+", v.lower()) if len(w) > 2}


def validate_glossary(tool: dict, obj: dict | None) -> dict[str, dict[str, list[str]]]:
    """Keep only paraphrases with zero word overlap with their value; drop empties."""
    props = tool["mcp"]["inputSchema"]["properties"]
    got = (obj or {}).get("glossary") or {}
    out: dict[str, dict[str, list[str]]] = {}
    for pname, spec in props.items():
        if "enum" not in spec:
            continue
        pv = got.get(pname) or {}
        for v in spec["enum"]:
            cands = pv.get(v) or []
            bad = _value_words(v)
            keep = [c.strip() for c in cands if isinstance(c, str)
                    and 3 <= len(c.split()) <= 14
                    and not (_value_words(c) & bad)]
            if keep:
                out.setdefault(pname, {})[v] = keep[:3]
    return out


def glossary_block(tool: dict, gl: dict[str, dict[str, list[str]]]) -> str:
    if not gl:
        return ""
    lines = ["Paraphrase reference — use one of these (or an equivalent that avoids the value's words):"]
    for pname, vals in gl.items():
        for v, paras in vals.items():
            lines.append(f"  {pname}={v}: " + " | ".join(paras))
    return "\n".join(lines) + "\n"


ENUM_RULES = {
    "natural": (
        "- Enum values: write them the way a real user would. If the value is a proper name or a specific\n"
        "  term (a cheese, a typeface, a temperament, a place, a product), use the NAME (\"roquefort\", not\n"
        "  \"blue-veined sheep cheese\"). If the value is a descriptive phrase, natural wording is fine\n"
        "  (\"toxic gas\", \"close it during the day\"). Never require trivia to decode a value, and never\n"
        "  write the snake_case token itself (e.g. \"toxic_gas\"). Enum vocab for reference:"
    ),
    "strong": (
        "- Enum values: describe the choice WITHOUT using the value's words. Say what it means, its\n"
        "  effect, or a synonym (\"the air is dangerously thin\" for low_oxygen, \"panels catch the\n"
        "  afternoon sun\" for west_facing). Never write the value or its space-separated form.\n"
        "  Enum vocab for reference:"
    ),
}


def batch_prompt(tool: dict, calls: list[dict], personas: list[str], paraphrase: str = "natural",
                 glossary_text: str = "") -> str:
    enum_rule = ENUM_RULES[paraphrase]
    props = tool["mcp"]["inputSchema"]["properties"]
    enum_lines = []
    for pname, spec in props.items():
        if "enum" in spec:
            enum_lines.append(f"  {pname}: {', '.join(spec['enum'])}")
    enum_text = "\n".join(enum_lines) if enum_lines else "  (none)"
    items = []
    for i, (c, p) in enumerate(zip(calls, personas)):
        items.append(f"{i}. persona: {p}\n   call: {json.dumps(c['arguments'], separators=(',', ':'))}")
    items_text = "\n".join(items)
    return f"""A software agent has this tool available:
{tool['schema_text']}

Below are {len(calls)} concrete calls to this tool. For EACH one, write the user message that
would make the agent produce EXACTLY that call — same parameter values, nothing more.

{glossary_text}Hard rules:
{intent_block(tool)}
- Never write the tool's snake_case name token (e.g. report_glaze_defects); plain phrasing is fine.
{enum_rule}
{enum_text}
- Integer values must appear as digits exactly as given.
- String values must appear verbatim, exactly as given. Any string value containing a space MUST
  be enclosed in double quotes in the message (so its boundaries are unambiguous). The wording must
  make clear which parameter each one belongs to, using a cue tied to that parameter's meaning:
  a person/reporter/owner field -> "reported by X", "filed by X", "per X", "X is reporting";
  a place/section/location field -> "in X", "at X", "inside X". NEVER introduce a person with
  "from X" or "at X" — those read as locations.
- If a parameter is absent from the call, the message must not mention it or hint at it.
- Booleans: express the true/false naturally ("skip the...", "make sure to include...").
- Follow the persona for each item. Vary sentence structure across items.

Items:
{items_text}

Output JSON only: {{"queries": [{{"idx": 0, "query": "..."}}, ...]}} with one entry per item."""


_NUM_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
              "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12"}


def _norm(s: str) -> str:
    return re.sub(r"[\s_\-]+", " ", s.lower())


def check_query(tool: dict, call: dict, query: str, strict_echo: bool = False) -> tuple[list[str], float]:
    """Local, free checks. Returns (errors, echo_score). strict_echo: any natural-phrase echo is an error."""
    errs: list[str] = []
    props = tool["mcp"]["inputSchema"]["properties"]
    q = query
    qn = _norm(query)
    name = tool["mcp"]["name"]
    if re.search(rf"(?<![a-z0-9_]){re.escape(name)}(?![a-z0-9_])", q.lower()):
        errs.append("copies tool token")          # literal snake_case: hard reject
    # natural phrase form ("report glaze defects") is how users talk; allowed

    echo_hits = 0
    echo_total = 0
    for pname, val in call["arguments"].items():
        spec = props[pname]
        if "enum" in spec:
            echo_total += 1
            if "_" in val and re.search(rf"(?<![a-z0-9_]){re.escape(val)}(?![a-z0-9_])", q.lower()):
                errs.append(f"copies enum token {val!r}")   # literal snake_case leak: hard reject
            if _norm(val) in qn:
                echo_hits += 1                              # natural-phrase echo: covariate only
        elif spec["type"] == "integer":
            q_digits = re.sub(r"(?<=\d),(?=\d{3}\b)", "", q)   # "1,200" -> "1200"
            for w, d in _NUM_WORDS.items():                    # "two days" -> "2 days"
                q_digits = re.sub(rf"\b{w}\b", d, q_digits, flags=re.I)
            if not re.search(rf"(?<!\d){val}(?!\d)", q_digits):
                errs.append(f"missing int {val}")
        elif spec["type"] == "string":
            if val not in q and _norm(val) not in qn:
                errs.append(f"missing string {val!r}")
    # optionals omitted from the call must not be named in the query
    # (skip values that legitimately appear because a present param uses them, or because a
    #  present STRING value contains that word — "shiitake-spore-print SP1" carries "shiitake")
    present_vals = {_norm(str(v)) for v in call["arguments"].values() if isinstance(v, str)}
    string_text = " ".join(_norm(str(v)) for k, v in call["arguments"].items()
                           if isinstance(v, str) and "enum" not in props.get(k, {}))
    for pname, spec in props.items():
        if pname not in call["arguments"] and "enum" in spec:
            for v in spec["enum"]:
                nv = _norm(v)
                if nv in qn and nv not in present_vals and nv not in string_text:
                    errs.append(f"names omitted enum value {v!r}")
    if not (4 <= len(query.split()) <= 90):
        errs.append("length out of range")
    score = echo_hits / echo_total if echo_total else 0.0
    if strict_echo and echo_hits:
        errs.append(f"echoes enum phrase (strong mode), score={score:.2f}")
    return errs, score


def mock_queries(tool: dict, calls: list[dict]) -> dict:
    out = []
    for i, c in enumerate(calls):
        bits = []
        for k, v in c["arguments"].items():
            if isinstance(v, bool):
                bits.append(("with" if v else "without") + " " + k.replace("_", " "))
            elif isinstance(v, int):
                bits.append(f"{v} of {k.replace('_', ' ')}")
            elif isinstance(v, str) and "enum" in tool["mcp"]["inputSchema"]["properties"][k]:
                bits.append(f"the {k.replace('_', ' ')} option number {hash(v) % 7}")
            else:
                bits.append(f"for '{v}'")
        out.append({"idx": i, "query": f"Please handle {tool['skeleton']['domain']} " + ", ".join(bits) + "."})
    return {"queries": out}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--tools", type=int, default=None, help="use first N tools (nested prefix)")
    ap.add_argument("--per-tool", type=int, default=350)
    ap.add_argument("--held", type=int, default=HELD_DEFAULT)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--pool-size", type=int, default=36, help="train-only string values per param")
    ap.add_argument("--held-pool-size", type=int, default=12, help="held-only string values per param (disjoint)")
    ap.add_argument("--only", default=None, help="comma-separated tool names to (re)generate; others untouched")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip tools that already have a <tool>.jsonl in --out (use when growing the catalog)")
    ap.add_argument("--force-pools", action="store_true", help="regenerate string pools for the selected tools")
    ap.add_argument("--topup", default=None,
                    help="JSON like '{\"tool_name\": {\"train\": 7, \"held\": 2}}' — generate that many EXTRA "
                         "rows per tool/split with fresh indices, appended to existing files (validate prints this)")
    ap.add_argument("--provider", default="openrouter", choices=list(PROVIDERS) + ["mock"])
    ap.add_argument("--model", default="x-ai/grok-4.6")
    ap.add_argument("--reasoning", default="off", choices=["off", "low", "default"])
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--out", default="queries")
    ap.add_argument("--paraphrase", default="natural", choices=list(ENUM_RULES))
    ap.add_argument("--pools", default=None, help="reuse string_pools.json from another run so calls are identical")
    args = ap.parse_args()

    cat = json.loads(Path(args.catalog).read_text())
    tools = cat["tools"][: args.tools] if args.tools else cat["tools"]
    if args.only:
        only = {n.strip() for n in args.only.split(",")}
        tools = [t for t in tools if t["mcp"]["name"] in only]
        if not tools:
            raise SystemExit(f"--only matched no tools: {args.only}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and not args.topup:
        before = len(tools)
        tools = [t for t in tools if not (out / f"{t['mcp']['name']}.jsonl").exists()]
        print(f"skip-existing: {before - len(tools)} tools already present, generating {len(tools)}", file=sys.stderr)
        if not tools:
            print("nothing to do"); return 0
    llm = LLM(args.provider, args.model, args.temperature, args.reasoning)
    t0 = time.time()

    # ---- string pools (one call per tool with string params; cached) ----
    pools_path = out / "string_pools.json"
    src = Path(args.pools) if args.pools else pools_path
    pools: dict[str, dict[str, list[str]]] = json.loads(src.read_text()) if src.exists() else {}
    if args.force_pools:
        for t in tools:
            pools.pop(t["mcp"]["name"], None)

    def get_pool(tool: dict) -> tuple[str, dict[str, list[str]]]:
        name = tool["mcp"]["name"]
        props = tool["mcp"]["inputSchema"]["properties"]
        sparams = [p for p, s in props.items() if s["type"] == "string" and "enum" not in s]
        if not sparams:
            return name, {}
        if name in pools and all(p in pools[name] and isinstance(pools[name][p], dict) for p in sparams):
            return name, pools[name]
        total = args.pool_size + args.held_pool_size
        prompt = string_pool_prompt(tool, sparams, total)
        obj, _ = llm.json_call(prompt, 2500,
                               mock_fn=lambda: {"pools": {p: [f"{p} value {i}" for i in range(total)] for p in sparams}})
        got = (obj or {}).get("pools", {})
        cleaned = {}
        for p in sparams:
            vals, seen = [], set()
            for v in got.get(p, []):
                if isinstance(v, (str, int)) and str(v).strip() and str(v).strip() not in seen:
                    seen.add(str(v).strip()); vals.append(str(v).strip())
            vals = vals[:total]
            need = args.held_pool_size + 8
            if len(vals) < need:
                raise SystemExit(
                    f"string pool for {name}.{p}: got {len(vals)} values, need >= {need}. "
                    f"Not writing pools. LLM error: {_.get('error')!r}; raw: {str(_.get('raw'))[:300]!r}")
            n_held = min(args.held_pool_size, len(vals) // 4)  # //4 keeps train from starving if the LLM under-filled
            cleaned[p] = {"train": vals[:-n_held], "held": vals[-n_held:]}  # suffix is held-out; prefix is train
        return name, cleaned

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for name, pool in ex.map(get_pool, tools):
            if pool:
                pools[name] = pool
    pools_path.write_text(json.dumps(pools, indent=1))
    print(f"string pools ready for {len(pools)} tools ({time.time()-t0:.0f}s)", file=sys.stderr)
    for t in tools:
        for p, entry in pools.get(t["mcp"]["name"], {}).items():
            print(f"    {t['mcp']['name']}.{p}: train e.g. {entry['train'][:3]}  held e.g. {entry['held'][:2]}", file=sys.stderr)

    # ---- paraphrase glossaries (strong mode only; cached) ----
    glossaries: dict[str, dict] = {}
    if args.paraphrase == "strong":
        gpath = out / "glossary.json"
        glossaries = json.loads(gpath.read_text()) if gpath.exists() else {}

        def get_gl(tool: dict):
            name = tool["mcp"]["name"]
            if name in glossaries:
                return name, glossaries[name]
            obj, _ = llm.json_call(glossary_prompt(tool), 2500,
                                   mock_fn=lambda: {"glossary": {p: {v: [f"the {i} kind", f"option {i} here", f"variant {i}"]
                                                                  for i, v in enumerate(sp["enum"])}
                                                              for p, sp in tool["mcp"]["inputSchema"]["properties"].items() if "enum" in sp}})
            return name, validate_glossary(tool, obj)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for name, gl in ex.map(get_gl, tools):
                glossaries[name] = gl
        gpath.write_text(json.dumps(glossaries, indent=1))
        n_vals = sum(len(v) for g in glossaries.values() for v in g.values())
        n_all = sum(len(sp["enum"]) for t in tools for sp in t["mcp"]["inputSchema"]["properties"].values() if "enum" in sp)
        print(f"glossary: {n_vals}/{n_all} enum values have validated paraphrases ({time.time()-t0:.0f}s)", file=sys.stderr)

    # ---- per-tool generation ----
    topup = json.loads(args.topup) if args.topup else None
    rejects = open(out / "rejects.jsonl", "a" if topup is not None else "w")
    stats = {"ok": 0, "rejected": 0, "batches": 0, "echo_sum": 0.0}

    def gen_tool(tool: dict) -> list[dict]:
        name = tool["mcp"]["name"]
        cut = args.per_tool - args.held
        if topup is None:
            rng = random.Random(f"{args.seed}:{name}")
            idxs_all = list(range(args.per_tool))
            split_of = {i: ("held" if i >= cut else "train") for i in idxs_all}
        else:
            need = topup.get(name, {})
            n_tr, n_he = int(need.get("train", 0)), int(need.get("held", 0))
            if n_tr + n_he == 0:
                return []
            existing = out / f"{name}.jsonl"
            # continue i after the last existing row (or after the original per-tool range)
            start = 1 + max((json.loads(l)["i"] for l in open(existing)), default=args.per_tool - 1) if existing.exists() else args.per_tool
            rng = random.Random(f"{args.seed}:{name}:topup:{start}")
            idxs_all = list(range(start, start + n_tr + n_he))
            split_of = {i: ("train" if k < n_tr else "held") for k, i in enumerate(idxs_all)}
        calls = {i: sample_call(tool, pools.get(name, {}), rng, held=(split_of[i] == "held")) for i in idxs_all}
        # catalog_index offset: tools don't all start on the same persona
        personas = {i: PERSONAS[(i + tool["catalog_index"]) % len(PERSONAS)] for i in idxs_all}
        rows: list[dict] = []
        pending = list(idxs_all)
        for attempt in range(args.retries + 1):
            if not pending:
                break
            next_pending: list[int] = []
            for b in range(0, len(pending), args.batch):
                idxs = pending[b: b + args.batch]
                bcalls = [calls[i] for i in idxs]
                bpers = [personas[i] for i in idxs]
                obj, prov = llm.json_call(batch_prompt(tool, bcalls, bpers, args.paraphrase,
                                                       glossary_block(tool, glossaries.get(name, {}))), 400 * len(idxs) + 200,
                                          mock_fn=lambda: mock_queries(tool, bcalls))
                stats["batches"] += 1
                got = {}
                for e in ((obj or {}).get("queries") or []):
                    if isinstance(e, dict) and isinstance(e.get("idx"), int) and isinstance(e.get("query"), str):
                        got[e["idx"]] = e["query"].strip()
                for j, i in enumerate(idxs):
                    q = got.get(j)
                    if q is None:
                        next_pending.append(i)
                        rejects.write(json.dumps({"tool": name, "i": i, "attempt": attempt,
                                                  "errors": ["missing from batch"], "prov": prov}) + "\n")
                        continue
                    errs, echo = check_query(tool, calls[i], q, strict_echo=(args.paraphrase == "strong"))
                    if errs:
                        next_pending.append(i)
                        rejects.write(json.dumps({"tool": name, "i": i, "attempt": attempt, "query": q,
                                                  "call": calls[i], "errors": errs, "echo": echo,
                                                  "persona": personas[i], "prov": prov}) + "\n")
                        continue
                    rows.append({"query_id": f"{name}:{i:04d}", "tool": name, "i": i,
                                 "persona": personas[i], "call": calls[i], "query": q,
                                 "echo_score": echo, "attempt": attempt, "prov": prov})
            pending = next_pending
        rows.sort(key=lambda r: r["i"])
        for r in rows:
            r["split"] = split_of[r["i"]]
            r["train_rank"] = None   # assigned by validate --filter over the validated set
        return rows

    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(gen_tool, t): t for t in tools}
        for fut in as_completed(futs):
            rows = fut.result()
            t = futs[fut]
            mode = "a" if topup is not None else "w"
            with open(out / f"{t['mcp']['name']}.jsonl", mode) as fh:
                fh.write("".join(json.dumps(r) + "\n" for r in rows))
            all_rows.extend(rows)
            stats["ok"] += len(rows)
            stats["echo_sum"] += sum(r["echo_score"] for r in rows)
            print(f"  {t['mcp']['name']}: {len(rows)}/{args.per_tool} ok  ({time.time()-t0:.0f}s)", file=sys.stderr)
    rejects.close()

    n_rej = sum(1 for _ in open(out / "rejects.jsonl"))
    manifest = {
        "catalog": args.catalog, "n_tools": len(tools), "per_tool": args.per_tool, "held": args.held,
        "provider": args.provider, "model": args.model, "reasoning": args.reasoning,
        "temperature": args.temperature, "seed": args.seed, "personas": PERSONAS, "paraphrase": args.paraphrase,
        "ok": stats["ok"], "reject_records": n_rej, "batches": stats["batches"],
        "mean_echo_score": (stats["echo_sum"] / stats["ok"]) if stats["ok"] else None,
        "train_per_tool": {f.stem: sum(1 for l in open(f) if json.loads(l)["split"] == "train")
                           for f in sorted(out.glob("*.jsonl")) if f.name not in ("rejects.jsonl", "validation.jsonl", "validation_errors.jsonl")},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{stats['ok']} queries ok, {n_rej} reject records, {stats['batches']} API batches, "
          f"{time.time()-t0:.0f}s\nmanifest: {out}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())