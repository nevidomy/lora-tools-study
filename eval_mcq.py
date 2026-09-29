#!/usr/bin/env python3
"""Capability retention after tool-internalization fine-tuning (reviewer / Shemla et al. comparison).

Scores the base model and each adapter checkpoint on 100 MMLU multiple-choice questions drawn from the
five subjects Shemla et al. (2026) used — high_school_computer_science, high_school_geography,
logical_fallacies, marketing, miscellaneous — 20 per subject, fixed order. Temperature 0. No tools in
the prompt. Reports accuracy and retention = adapter / base.

Usage:
  python eval_mcq.py --model openai/gpt-oss-20b \\
      --checkpoint mix50_r32='tinker://.../sampler_weights/cat64_mix50_C_lr5e-4dec_s0' \\
      --checkpoint mix50_r1='tinker://.../sampler_weights/cat64_mix50_C_lr5e-4dec_r1_s0' \\
      --out results/mcq_gptoss.json
Requires `datasets` (pip install datasets); MMLU is fetched from the HF hub (cais/mmlu).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

SUBJECTS = ["high_school_computer_science", "high_school_geography", "logical_fallacies", "marketing", "miscellaneous"]
LETTERS = "ABCD"
SYSTEM = "Answer the multiple-choice question. Reply with only the letter of the correct choice: A, B, C, or D."


def load_questions(per_subject: int) -> list[dict]:
    from datasets import load_dataset
    qs = []
    for subj in SUBJECTS:
        ds = load_dataset("cais/mmlu", subj, split="test")
        for i, ex in enumerate(ds):
            if i >= per_subject:
                break
            qs.append({"subject": subj, "question": ex["question"], "choices": ex["choices"], "answer": int(ex["answer"])})
    return qs


def format_q(q: dict) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(q["choices"]))
    return f"{q['question']}\n\n{opts}\n\nAnswer:"


def extract_letter(text: str) -> str | None:
    m = re.search(r"\b([ABCD])\b", text.strip().upper())
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--checkpoint", action="append", default=[], help="name=tinker://... (repeatable)")
    ap.add_argument("--per-subject", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--renderer", default=None)
    ap.add_argument("--date", default="2026-09-12")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    sys.path.insert(0, ".")
    from eval_arms import default_renderer  # same renderer choice as the tool-calling harness

    qs = load_questions(args.per_subject)
    print(f"{len(qs)} questions across {len(SUBJECTS)} subjects", file=sys.stderr)
    tok = get_tokenizer(args.model)
    rname = args.renderer or default_renderer(args.model)
    renderer = renderers.get_renderer(rname, tok, model_name=args.model)
    if args.date and hasattr(renderer, "current_date"):
        renderer.current_date = args.date
    stop = renderer.get_stop_sequences()
    params = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=0.0, stop=stop)
    svc = tinker.ServiceClient()

    runs = [("base", None)] + [c.split("=", 1) for c in args.checkpoint]
    results = {}
    for name, ckpt in runs:
        sampler = (svc.create_sampling_client(model_path=ckpt) if ckpt
                   else svc.create_sampling_client(base_model=args.model))

        def one(q):
            msgs = [{"role": "user", "content": SYSTEM + "\n\n" + format_q(q)}]   # renderer adds its own system block
            mi = renderer.build_generation_prompt(msgs)
            resp = sampler.sample(prompt=mi, num_samples=1, sampling_params=params).result()
            toks = list(resp.sequences[0].tokens)
            msg, _ = renderer.parse_response(toks)
            content = msg.get("content")
            text = content if isinstance(content, str) else " ".join(
                getattr(p, "text", "") for p in (content or []) if getattr(p, "type", "") == "text")
            if not text:
                text = tok.decode(toks, skip_special_tokens=True)
            pred = extract_letter(text)
            tcs = msg.get("tool_calls") or []
            dec = tok.decode(toks, skip_special_tokens=False)
            # expecting tool call
            is_call = bool(tcs) or ("<|constrain|>" in dec and "<|message|>{" in dec) or ("content_invoke_tool" in dec)
            call_name = tcs[0].function.name if tcs else None
            return pred == LETTERS[q["answer"]], pred is None, q["subject"], is_call, call_name, dec[:300]

        with ThreadPoolExecutor(max_workers=16) as ex:
            outs = list(ex.map(one, qs))
        acc = sum(o[0] for o in outs) / len(outs)
        unparsed = sum(o[1] for o in outs)
        n_call = sum(o[3] for o in outs)
        import collections
        called = collections.Counter(o[4] for o in outs if o[4])
        by_subj = {s: sum(o[0] for o in outs if o[2] == s) / max(1, sum(1 for o in outs if o[2] == s)) for s in SUBJECTS}
        examples = [o[5] for o in outs if o[1]][:3]
        results[name] = {"accuracy": acc, "unparsed": unparsed, "tool_call_rate": n_call / len(outs),
                         "tools_called": dict(called.most_common(5)), "by_subject": by_subj,
                         "unparsed_examples": examples, "checkpoint": ckpt}
        base = results["base"]["accuracy"]
        ret = f"  retention={acc / base:.3f}" if name != "base" and base else ""
        print(f"{name:12s} acc={acc:.3f}  unparsed={unparsed}  tool_call={n_call}/{len(outs)}{ret}   " +
              " ".join(f"{s[:12]}={v:.2f}" for s, v in by_subj.items()), file=sys.stderr)
        if called:
            print(f"             tools called: {dict(called.most_common(4))}", file=sys.stderr)
        for e in examples[:1]:
            print(f"             e.g. {e!r}", file=sys.stderr)

    if args.out:
        json.dump({"model": args.model, "renderer": rname, "n": len(qs), "subjects": SUBJECTS, "results": results},
                  open(args.out, "w"), indent=1)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    import os; os._exit(rc or 0)