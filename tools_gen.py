#!/usr/bin/env python3
"""Generate a validated, nested catalog of synthetic MCP-style tools.

Design:
  1. Seeded skeleton sampler (programmatic, deterministic): domain, verb,
     param count, per-slot type from a fixed mix, required/optional.
  2. LLM fill: names, descriptions, domain-specific enum vocab.
  3. Meta-schema validation against the skeleton. Reject + retry.
  4. Nested catalogs 1 ⊂ 4 ⊂ 16 ⊂ 64 as prefixes of the seeded order.

Usage:
  python tools_gen.py --provider openrouter --model x-ai/grok-4.3 --seed 0
  python tools_gen.py --provider mock --seed 0      # exercise pipeline, no API
  python tools_gen.py --dry-run                     # print skeletons only

Env: OPENROUTER_API_KEY / XAI_API_KEY / TINKER_API_KEY depending on provider.
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Curated lists. Niche on purpose: the base model must not be able to guess
# names or enum vocab without the schema (arm B ≈ 0 is the novelty test).
# --------------------------------------------------------------------------

DOMAINS = [
    "kiln firing schedules", "beehive inspection", "vinyl record pressing",
    "orchard grafting", "telescope collimation", "sourdough starter maintenance",
    "marine buoy telemetry", "glacier mass-balance survey", "hand loom weaving",
    "cheese cave aging", "harpsichord tuning", "cider fermentation",
    "falconry mews management", "letterpress print shop", "bonsai training",
    "ham radio contest logging", "koi pond water chemistry",
    "sailing regatta race committee", "mushroom cultivation", "tidal pool survey",
    "bell foundry casting", "bicycle frame brazing", "planetarium show scheduling",
    "archival film preservation", "climbing gym route setting",
    "maple syrup sugarbush", "tannery hide processing", "coral nursery",
    "bat colony acoustic monitoring", "blacksmith forge", "rooftop solar array",
    "community seed library", "model railway layout", "chocolate tempering",
    "stained glass studio", "sled dog kennel", "hydroponic greenhouse",
    "pipe organ maintenance", "wildfire lookout tower", "fossil dig site",
    "oyster farm", "handmade paper mill", "escape room operations",
    "ice rink resurfacing", "hot air balloon launches", "aquarium quarantine tanks",
    "bookbinding workshop", "wind tunnel test scheduling", "lighthouse maintenance",
    "observatory dome operations", "ceramic glaze formulation", "saffron harvest",
    "rowing boathouse", "cave rescue logistics", "permafrost borehole logging",
    "vineyard canopy management", "seismograph station", "cricket pitch groundskeeping",
    "dry-stone walling", "mead brewing", "taxidermy studio", "glassblowing furnace",
    "bird banding station", "curling ice preparation", "weather balloon launches",
    "herbarium specimen cataloging", "chairlift operations", "peat bog restoration",
    "camera obscura tours", "clock tower winding", "distillery cask warehouse",
    "silkworm rearing", "ferry berth scheduling", "antique typewriter repair",
    "carillon concerts", "reef survey dive planning", "avalanche control blasting",
    "pottery wheel studio", "cranberry bog flooding", "water tower inspection",
    "surf break forecasting", "steam locomotive shed", "tea leaf withering",
    "windmill sail maintenance", "tattoo studio booking", "alpaca shearing",
]

VERBS = [
    "schedule", "query", "set", "list", "create", "cancel", "update", "reserve",
    "log", "estimate", "assign", "flag", "archive", "calibrate", "allocate",
    "inspect", "report", "adjust",
]

# Fixed type mix. At least one enum is forced per tool (see sampler) so every
# tool has closed vocabulary to internalize; pure-extraction tools are trivial.
TYPE_MIX = [("enum", 0.40), ("bool", 0.20), ("int", 0.20), ("string", 0.20)]
P_REQUIRED = 0.70
PARAM_COUNT_RANGE = (2, 5)
ENUM_VALUES_RANGE = (3, 6)

# Things a base model would guess without any schema. Reject on sight.
GENERIC_ENUM_VALUES = {
    "low", "medium", "high", "small", "large", "yes", "no", "true", "false",
    "on", "off", "none", "default", "auto", "other", "unknown", "normal",
    "standard", "basic", "advanced", "premium", "min", "max", "all", "any",
    "active", "inactive", "enabled", "disabled", "open", "closed", "pending",
    "done", "complete", "completed", "cancelled", "canceled", "error", "ok",
}
GENERIC_TOOL_NAMES = {
    "get_weather", "search", "web_search", "send_email", "get_time",
    "calculator", "get_location", "create_event", "set_reminder", "translate",
}

SNAKE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


# --------------------------------------------------------------------------
# Skeletons
# --------------------------------------------------------------------------

@dataclass
class ParamSlot:
    slot: str            # p1..pN
    type: str            # enum | bool | int | string
    required: bool


@dataclass
class Skeleton:
    tool_id: int
    domain: str
    verb: str
    params: list[ParamSlot]

    def key(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]


def _sample_type(rng: random.Random) -> str:
    r = rng.random()
    acc = 0.0
    for t, p in TYPE_MIX:
        acc += p
        if r < acc:
            return t
    return TYPE_MIX[-1][0]


def sample_skeletons(n: int, seed: int) -> list[Skeleton]:
    rng = random.Random(seed)
    if n > len(DOMAINS):
        raise SystemExit(f"n={n} exceeds domain pool ({len(DOMAINS)}); add domains")
    domains = rng.sample(DOMAINS, n)
    out: list[Skeleton] = []
    for i, dom in enumerate(domains):
        k = rng.randint(*PARAM_COUNT_RANGE)
        types = [_sample_type(rng) for _ in range(k)]
        if "enum" not in types:
            types[rng.randrange(k)] = "enum"
        params = [
            ParamSlot(slot=f"p{j+1}", type=t, required=(rng.random() < P_REQUIRED))
            for j, t in enumerate(types)
        ]
        # Force at least one required param so every call has content.
        if not any(p.required for p in params):
            params[0].required = True
        out.append(Skeleton(tool_id=i, domain=dom, verb=rng.choice(VERBS), params=params))
    return out


# --------------------------------------------------------------------------
# LLM fill
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You design compact, domain-specific tool APIs for software agents. "
    "You output strictly valid JSON and nothing else: no prose, no code fences."
)


def build_fill_prompt(sk: Skeleton) -> str:
    slots = []
    for p in sk.params:
        req = "required" if p.required else "optional"
        extra = " — give 3 to 6 enum values" if p.type == "enum" else ""
        slots.append(f"  {p.slot}: type={p.type}, {req}{extra}")
    slot_text = "\n".join(slots)
    return f"""Define ONE tool for this domain.

Domain: {sk.domain}
Action verb: {sk.verb}

Rules:
- name: snake_case, 2-4 words, begins with "{sk.verb}", specific to the domain.
  Must not be a generic name (no get_weather, search, send_email, etc.).
- description: one sentence, 8-30 words, says WHEN an agent should call it.
- params: exactly {len(sk.params)}, in this exact order and with these exact types:
{slot_text}
- Every param needs: "slot" (copy from above), "name" (snake_case, unique in this tool),
  "description" (<= 20 words).
- Enum params additionally need "values": 3-6 distinct lowercase snake_case values that are
  specific to the domain. Forbidden: low/medium/high, small/medium/large, yes/no, on/off,
  true/false, none/default/auto/other, or any other generic word. Prefer real domain
  jargon (e.g. for kiln firing: bisque, glaze, raku, crystalline).
- Do not add a "type" or "required" field; those are fixed by the slot list.

Output JSON only:
{{"name": "...", "description": "...", "params": [{{"slot": "p1", "name": "...", "description": "...", "values": ["..."]}}, ...]}}"""


PROVIDERS = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "tinker": ("https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1", "TINKER_API_KEY"),
    "local": ("http://localhost:8000/v1", None),
}


class Filler:
    """Thin provider layer. `mock` produces deterministic valid-ish fills for testing."""

    def __init__(self, provider: str, model: str, temperature: float):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.client = None
        if provider != "mock":
            from openai import OpenAI
            base_url, key_env = PROVIDERS[provider]
            api_key = os.environ.get(key_env, "local") if key_env else "local"
            if key_env and not os.environ.get(key_env):
                raise SystemExit(f"{key_env} not set")
            self.client = OpenAI(base_url=base_url, api_key=api_key)

    def fill(self, sk: Skeleton, attempt: int) -> tuple[dict | None, dict]:
        """Returns (parsed_json_or_None, provenance)."""
        prompt = build_fill_prompt(sk)
        prov: dict[str, Any] = {
            "provider": self.provider, "model": self.model,
            "temperature": self.temperature, "attempt": attempt,
            "prompt_sha1": hashlib.sha1(prompt.encode()).hexdigest()[:12],
        }
        if self.provider == "mock":
            return _mock_fill(sk, attempt), prov

        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=1200,
        )
        raw = None
        for use_json_mode in (True, False):
            try:
                if use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                else:
                    kwargs.pop("response_format", None)
                resp = self.client.chat.completions.create(**kwargs)
                raw = resp.choices[0].message.content or ""
                prov["response_model"] = getattr(resp, "model", None)
                prov["routed_provider"] = getattr(resp, "provider", None)
                usage = getattr(resp, "usage", None)
                if usage:
                    prov["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
                    prov["completion_tokens"] = getattr(usage, "completion_tokens", None)
                break
            except Exception as e:  # json mode unsupported → retry without
                prov["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                if not use_json_mode:
                    return None, prov
        if raw is None:
            return None, prov
        prov["raw_sha1"] = hashlib.sha1(raw.encode()).hexdigest()[:12]
        return _parse_json(raw), prov


def _parse_json(raw: str) -> dict | None:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def _mock_fill(sk: Skeleton, attempt: int) -> dict:
    stem = re.sub(r"[^a-z]+", "_", sk.domain.lower()).strip("_").split("_")
    params = []
    for i, p in enumerate(sk.params):
        entry = {"slot": p.slot, "name": f"{stem[min(i, len(stem)-1)]}_{p.type}_{i}",
                 "description": f"mock {p.type} parameter {i} for {sk.domain}"}
        if p.type == "enum":
            entry["values"] = [f"{stem[0]}_mode_{j}" for j in range(3 + (i % 3))]
        params.append(entry)
    # attempt 0 for tool_id % 7 == 3 deliberately breaks the enum rule to exercise rejects
    if attempt == 0 and sk.tool_id % 7 == 3:
        for e in params:
            if "values" in e:
                e["values"] = ["low", "medium", "high"]
    return {"name": f"{sk.verb}_{stem[0]}_{stem[-1]}",
            "description": f"Use this to {sk.verb} things for {sk.domain} when an agent needs to act on it.",
            "params": params}


# --------------------------------------------------------------------------
# Validation against skeleton
# --------------------------------------------------------------------------

def validate(sk: Skeleton, obj: dict | None, seen_names: set[str]) -> list[str]:
    errs: list[str] = []
    if not isinstance(obj, dict):
        return ["not a JSON object"]
    name = obj.get("name", "")
    desc = obj.get("description", "")
    params = obj.get("params")

    if not isinstance(name, str) or not SNAKE.match(name):
        errs.append(f"name not snake_case: {name!r}")
    else:
        if not name.startswith(sk.verb + "_"):
            errs.append(f"name must start with verb {sk.verb!r}: {name!r}")
        nwords = name.count("_") + 1
        if not (2 <= nwords <= 4):
            errs.append(f"name must be 2-4 words: {name!r}")
        if name in GENERIC_TOOL_NAMES:
            errs.append(f"generic tool name: {name!r}")
        if name in seen_names:
            errs.append(f"duplicate tool name: {name!r}")

    if not isinstance(desc, str):
        errs.append("description missing")
    else:
        nw = len(desc.split())
        if not (8 <= nw <= 30):
            errs.append(f"description must be 8-30 words, got {nw}")

    if not isinstance(params, list) or len(params) != len(sk.params):
        errs.append(f"expected {len(sk.params)} params, got "
                    f"{len(params) if isinstance(params, list) else 'non-list'}")
        return errs

    pnames: set[str] = set()
    for slot, entry in zip(sk.params, params):
        if not isinstance(entry, dict):
            errs.append(f"{slot.slot}: not an object"); continue
        if entry.get("slot") != slot.slot:
            errs.append(f"{slot.slot}: slot mismatch ({entry.get('slot')!r})")
        pn = entry.get("name", "")
        if not isinstance(pn, str) or not SNAKE.match(pn):
            errs.append(f"{slot.slot}: param name not snake_case: {pn!r}")
        elif pn in pnames:
            errs.append(f"{slot.slot}: duplicate param name {pn!r}")
        else:
            pnames.add(pn)
        pd = entry.get("description", "")
        if not isinstance(pd, str) or not (1 <= len(pd.split()) <= 20):
            errs.append(f"{slot.slot}: param description must be 1-20 words")
        if "type" in entry or "required" in entry:
            errs.append(f"{slot.slot}: must not set type/required")
        vals = entry.get("values")
        if slot.type == "enum":
            if not isinstance(vals, list):
                errs.append(f"{slot.slot}: enum needs values list"); continue
            lo, hi = ENUM_VALUES_RANGE
            if not (lo <= len(vals) <= hi):
                errs.append(f"{slot.slot}: enum needs {lo}-{hi} values, got {len(vals)}")
            if len(set(vals)) != len(vals):
                errs.append(f"{slot.slot}: duplicate enum values")
            for v in vals:
                if not isinstance(v, str) or not SNAKE.match(v):
                    errs.append(f"{slot.slot}: enum value not snake_case: {v!r}")
                elif v in GENERIC_ENUM_VALUES:
                    errs.append(f"{slot.slot}: generic enum value {v!r}")
                elif v in pnames:
                    errs.append(f"{slot.slot}: enum value collides with a param name: {v!r}")
        elif vals is not None:
            errs.append(f"{slot.slot}: non-enum param must not have values")
    return errs


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

JSON_TYPE = {"enum": "string", "bool": "boolean", "int": "integer", "string": "string"}


def to_mcp(sk: Skeleton, obj: dict) -> dict:
    props: dict[str, Any] = {}
    required: list[str] = []
    for slot, entry in zip(sk.params, obj["params"]):
        spec: dict[str, Any] = {"type": JSON_TYPE[slot.type], "description": entry["description"]}
        if slot.type == "enum":
            spec["enum"] = list(entry["values"])
        props[entry["name"]] = spec
        if slot.required:
            required.append(entry["name"])
    return {
        "name": obj["name"],
        "description": obj["description"],
        "inputSchema": {"type": "object", "properties": props, "required": required,
                        "additionalProperties": False},
    }


def closed_vocab(sk: Skeleton, obj: dict) -> int:
    n = 1 + len(obj["params"])  # tool name + each param name
    for slot, entry in zip(sk.params, obj["params"]):
        if slot.type == "enum":
            n += len(entry["values"])
    return n


def render_schema_text(mcp: dict) -> str:
    """Compact text rendering for arm A prompts. Token count of this is the
    per-tool 'schema cost' in the cost model."""
    return json.dumps(mcp, separators=(",", ":"))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="openrouter", choices=list(PROVIDERS) + ["mock"])
    ap.add_argument("--model", default="x-ai/grok-4.6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=80, help="skeletons to attempt")
    ap.add_argument("--keep", type=int, default=64, help="tools in the largest catalog")
    ap.add_argument("--catalogs", default="1,4,16,64")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="tools")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    skeletons = sample_skeletons(args.n, args.seed)
    if args.dry_run:
        for sk in skeletons[:10]:
            print(sk.tool_id, sk.verb, "|", sk.domain, "|",
                  " ".join(f"{p.slot}:{p.type}{'*' if p.required else ''}" for p in sk.params))
        print(f"... {len(skeletons)} skeletons")
        return 0

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rejects = open(out / "rejects.jsonl", "w")
    filler = Filler(args.provider, args.model, args.temperature)

    seen_names: set[str] = set()
    results: dict[int, dict] = {}
    t0 = time.time()

    def work(sk: Skeleton) -> tuple[int, dict | None, list[dict]]:
        attempts: list[dict] = []
        for a in range(args.retries):
            obj, prov = filler.fill(sk, a)
            errs = validate(sk, obj, set())  # cross-tool name dedup happens after
            attempts.append({"tool_id": sk.tool_id, "errors": errs, "obj": obj, "prov": prov})
            if not errs:
                return sk.tool_id, {"obj": obj, "prov": prov, "attempts": a + 1}, attempts
        return sk.tool_id, None, attempts

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, sk): sk for sk in skeletons}
        for fut in as_completed(futs):
            tid, ok, attempts = fut.result()
            for a in attempts:
                if a["errors"]:
                    rejects.write(json.dumps(a) + "\n")
            if ok:
                results[tid] = ok
            done = len(results)
            if done % 10 == 0:
                print(f"  {done} ok / {len(skeletons)} attempted  ({time.time()-t0:.0f}s)", file=sys.stderr)

    # Cross-tool dedup on names, in seeded order.
    tools: list[dict] = []
    for sk in skeletons:
        r = results.get(sk.tool_id)
        if not r:
            continue
        name = r["obj"]["name"]
        if name in seen_names:
            rejects.write(json.dumps({"tool_id": sk.tool_id, "errors": [f"duplicate name {name}"],
                                      "obj": r["obj"], "prov": r["prov"]}) + "\n")
            continue
        seen_names.add(name)
        mcp = to_mcp(sk, r["obj"])
        tools.append({
            "tool_id": sk.tool_id,
            "skeleton": asdict(sk),
            "skeleton_key": sk.key(),
            "mcp": mcp,
            "schema_text": render_schema_text(mcp),
            "closed_vocab": closed_vocab(sk, r["obj"]),
            "provenance": r["prov"],
            "attempts": r["attempts"],
        })
    rejects.close()

    if len(tools) < args.keep:
        print(f"WARNING: only {len(tools)} valid tools, wanted {args.keep}. "
              f"Increase --n or --retries, or read rejects.jsonl.", file=sys.stderr)
    tools = tools[: args.keep]
    for i, t in enumerate(tools):
        t["catalog_index"] = i

    sizes = [int(s) for s in args.catalogs.split(",")]
    catalogs = {f"catalog_{s}": [t["mcp"]["name"] for t in tools[:s]] for s in sizes if s <= len(tools)}

    meta = {
        "seed": args.seed, "provider": args.provider, "model": args.model,
        "temperature": args.temperature, "n_attempted": args.n, "n_valid": len(tools),
        "type_mix": TYPE_MIX, "p_required": P_REQUIRED,
        "param_count_range": PARAM_COUNT_RANGE, "enum_values_range": ENUM_VALUES_RANGE,
        "nested": True, "catalogs": catalogs,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_closed_vocab": {k: sum(t["closed_vocab"] for t in tools[:len(v)]) for k, v in catalogs.items()},
    }
    (out / f"catalog_{len(tools)}.json").write_text(json.dumps({"meta": meta, "tools": tools}, indent=2))

    print(f"\nwrote {out}/catalog_{len(tools)}.json  ({len(tools)} tools, {time.time()-t0:.0f}s)")
    print(f"rejects: {out}/rejects.jsonl")
    print("closed-vocab per catalog:", meta["total_closed_vocab"])
    return 0


if __name__ == "__main__":
    sys.exit(main())