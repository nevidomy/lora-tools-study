#!/usr/bin/env python3
"""LoRA SFT on (query -> tool call) pairs via Tinker, rendered natively.

Arm C: train WITHOUT the schema in the prompt (the adapter must internalize it).
Arm D: train WITH the schema in the prompt (the correct baseline).
The assistant target is a native tool call built by the same renderer the eval uses,
so train and eval formats cannot drift.

Examples-per-tool subsets are nested prefixes by train_rank (10 ⊂ 30 ⊂ 100 ⊂ 300).
Two seeds are logged separately: LoRA init (Tinker's `seed`) and data order.

Usage:
  python train_sft.py --catalog tools/catalog_64.json --catalog-size 4 \\
      --queries queries_valid --examples-per-tool 100 --arm C \\
      --model openai/gpt-oss-20b --rank 32 --seed 0 --tag cat4_ex100_C_s0

Then:
  python eval_arms.py ... --checkpoint <printed tinker:// path> --arm C --tag cat4_ex100_C_s0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_arms import SYSTEM_PROMPTS, default_renderer, load_catalog  # noqa: E402

READ_VERBS = {"query", "list", "inspect", "estimate"}


def load_train(qdir: str, tools: list[dict], per_tool: int) -> list[dict]:
    rows = []
    for t in tools:
        f = Path(qdir) / f"{t['mcp']['name']}.jsonl"
        if not f.exists():
            print(f"  warning: no queries for {t['mcp']['name']}", file=sys.stderr)
            continue
        tr = [json.loads(l) for l in open(f)]
        tr = sorted((r for r in tr if r["split"] == "train"),
                    key=lambda r: (r["train_rank"] if r.get("train_rank") is not None else 10**9 + r["i"]))  # unranked after ranked
        if len(tr) < per_tool:
            print(f"  warning: {t['mcp']['name']} has {len(tr)} train rows < {per_tool}", file=sys.stderr)
        rows.extend(tr[:per_tool])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--catalog-size", type=int, required=True)
    ap.add_argument("--queries", required=True, help="validated natural-mode dir")
    ap.add_argument("--examples-per-tool", type=int, required=True)
    ap.add_argument("--queries-strong", default=None, help="second source (paraphrased) for mixed training")
    ap.add_argument("--strong-per-tool", type=int, default=0, help="examples per tool from --queries-strong")
    ap.add_argument("--arm", choices=["C", "D"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0, help="LoRA init seed (Tinker) and data-order seed")
    ap.add_argument("--data-seed", type=int, default=None, help="override data-order seed")
    ap.add_argument("--lr", type=float, default=None, help="default: cookbook get_lr(model, lora) or 5e-4")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--decay", action="store_true", help="linear LR decay to 0 after warmup (default: constant)")
    ap.add_argument("--max-length", type=int, default=16384,
                    help="hard cap on example length; the script aborts if any example exceeds it (no silent truncation)")
    ap.add_argument("--reduction", default="none", choices=["none", "mean"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--dry-run", action="store_true", help="build examples, print one, no API")
    ap.add_argument("--save-state", action="store_true", help="also save full optimizer state (2.7GB, billed storage)")
    ap.add_argument("--prompt", default="v1", choices=list(SYSTEM_PROMPTS), help="system prompt version (must match eval)")
    ap.add_argument("--renderer", default=None, help="renderer override (must match eval)")
    ap.add_argument("--date", default=None, help="pin renderer Current date (must match eval)")
    args = ap.parse_args()

    import tinker
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.renderers import ToolCall, TrainOnWhat
    from tinker_cookbook.supervised.common import datum_from_model_input_weights
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tools = load_catalog(args.catalog, args.catalog_size)
    rows = load_train(args.queries, tools, args.examples_per_tool)
    n_nat = len(rows)
    if args.queries_strong and args.strong_per_tool:
        strong_rows = load_train(args.queries_strong, tools, args.strong_per_tool)
        for r in strong_rows:
            r["source"] = "strong"
        rows = rows + strong_rows
    for r in rows:
        r.setdefault("source", "natural")
    data_seed = args.data_seed if args.data_seed is not None else args.seed
    print(f"{len(rows)} training examples ({args.examples_per_tool} natural"
          + (f" + {args.strong_per_tool} strong" if args.strong_per_tool else "")
          + f" per tool × {len(tools)} tools), arm {args.arm}", file=sys.stderr)

    tok = get_tokenizer(args.model)
    rname = args.renderer or default_renderer(args.model) or model_info.get_recommended_renderer_name(args.model)
    print(f"renderer: {rname}", file=sys.stderr)
    renderer = renderers.get_renderer(rname, tok, model_name=args.model)
    if args.date and hasattr(renderer, "current_date"):
        renderer.current_date = args.date
        print(f"renderer.current_date pinned to {args.date}", file=sys.stderr)
    with_tools = args.arm == "D"
    specs = [t["_spec"] for t in tools] if with_tools else []
    prefix = renderer.create_conversation_prefix_with_tools(specs, system_prompt=SYSTEM_PROMPTS[args.prompt])

    # ---- build datums ----
    datums, n_tok, n_target, target_counts = [], 0, 0, []
    longest = 0
    for r in rows:
        tc = ToolCall(function=ToolCall.FunctionBody(
            name=r["call"]["name"], arguments=json.dumps(r["call"]["arguments"], separators=(",", ":"))))
        msgs = prefix + [{"role": "user", "content": r["query"]},
                         {"role": "assistant", "content": "", "tool_calls": [tc]}]
        # loss only on the last assistant message (the tool-call tokens)
        mi, w = renderer.build_supervised_example(msgs, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
        longest = max(longest, mi.length)
        if mi.length > args.max_length:
            raise SystemExit(f"example for {r['query_id']} is {mi.length} tokens > --max-length {args.max_length}; "
                             f"raise --max-length rather than let it truncate the target")
        d = datum_from_model_input_weights(mi, w, max_length=args.max_length, reduction=args.reduction)
        datums.append(d)
        n_tok += mi.length
        tc = int((w > 0).sum())
        target_counts.append(tc)
        n_target += tc
    print(f"tokens: {n_tok} total, {n_target} target (loss-bearing); "
          f"{n_tok/len(datums):.0f}/example, longest {longest}", file=sys.stderr)
    est = n_tok * args.epochs / 1e6 * 0.396
    print(f"estimated train cost at GPT-OSS-20B rate: ~${est:.2f}", file=sys.stderr)

    if args.dry_run:
        mi, w = renderer.build_supervised_example(
            prefix + [{"role": "user", "content": rows[0]["query"]},
                      {"role": "assistant", "content": "", "tool_calls": [ToolCall(function=ToolCall.FunctionBody(
                          name=rows[0]["call"]["name"], arguments=json.dumps(rows[0]["call"]["arguments"])))]}],
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
        ids = mi.to_ints()
        print("\n--- example (loss-bearing tokens in [brackets]) ---")
        out = []
        for t_id, wt in zip(ids, w.tolist()):
            s = tok.decode([t_id])
            out.append(f"[{s}]" if wt > 0 else s)
        print("".join(out)[-1200:])
        return 0

    # ---- LR ----
    lr = args.lr
    if lr is None:
        try:
            from tinker_cookbook.hyperparam_utils import get_lr
            lr = get_lr(args.model, is_lora=True)
        except Exception:
            lr = 5e-4
    steps_per_epoch = (len(datums) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_frac))
    print(f"lr={lr:.2e}  steps={total_steps} ({steps_per_epoch}/epoch × {args.epochs})  warmup={warmup}",
          file=sys.stderr)

    # ---- train ----
    svc = tinker.ServiceClient()
    tc_client = svc.create_lora_training_client(base_model=args.model, rank=args.rank, seed=args.seed,
                                                user_metadata={"tag": args.tag, "arm": args.arm})
    print(f"console: {tc_client.get_console_url()}", file=sys.stderr)
    rng = random.Random(data_seed)
    losses = []
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        order = list(range(len(datums)))
        rng.shuffle(order)
        for b in range(0, len(order), args.batch_size):
            idx = order[b: b + args.batch_size]
            batch = [datums[i] for i in idx]
            n_tgt = sum(target_counts[i] for i in idx)
            cur_lr = lr * min(1.0, (step + 1) / warmup)
            if args.decay and step >= warmup:
                cur_lr = lr * max(0.0, (total_steps - step) / max(1, total_steps - warmup))
            fb = tc_client.forward_backward(batch, loss_fn="cross_entropy")
            op = tc_client.optim_step(tinker.AdamParams(learning_rate=cur_lr, beta1=0.9, beta2=0.95, eps=1e-8))
            fb_out = fb.result()
            op.result()
            m = dict(fb_out.metrics or {})
            loss_sum = m.get("loss:sum", m.get("loss", None))
            loss_tok = (loss_sum / n_tgt) if (loss_sum is not None and n_tgt) else None
            losses.append({"step": step, "epoch": epoch, "lr": cur_lr, "target_tokens": n_tgt,
                           "loss_sum": loss_sum, "loss_per_token": loss_tok, "metrics": m})
            if step == 0:
                print(f"  metrics keys: {sorted(m)}", file=sys.stderr)
            if step % 5 == 0 or step == total_steps - 1:
                print(f"  step {step:4d}/{total_steps}  epoch {epoch}  lr {cur_lr:.2e}  "
                      + (f"loss/tok {loss_tok:.4f}  (sum {loss_sum:.1f} over {n_tgt} tok)" if loss_tok is not None
                         else str({k: round(v, 4) for k, v in m.items()}))
                      + f"  ({time.time()-t0:.0f}s)", file=sys.stderr)
            step += 1

    resp = tc_client.save_weights_for_sampler(name=args.tag).result()
    ckpt = resp.path
    if args.save_state:
        try:
            tc_client.save_state(name=args.tag + "_state").result()
        except Exception as e:
            print(f"  (save_state failed: {e})", file=sys.stderr)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "tag": args.tag, "arm": args.arm, "model": args.model, "renderer": rname, "rank": args.rank,
        "prompt_version": args.prompt, "pinned_date": args.date,
        "lora_seed": args.seed, "data_seed": data_seed, "lr": lr, "decay": args.decay, "epochs": args.epochs,
        "batch_size": args.batch_size, "warmup_steps": warmup, "total_steps": total_steps,
        "reduction": args.reduction, "max_length": args.max_length,
        "catalog": args.catalog, "catalog_size": args.catalog_size, "queries": args.queries,
        "examples_per_tool": args.examples_per_tool, "n_examples": len(datums),
        "queries_strong": args.queries_strong, "strong_per_tool": args.strong_per_tool, "n_natural": n_nat,
        "tokens_total": n_tok, "tokens_target": n_target,
        "checkpoint": ckpt, "losses": losses, "seconds": round(time.time() - t0, 1),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / f"{args.tag}.json").write_text(json.dumps(manifest, indent=2))
    # best-effort clean shutdown of SDK background tasks (avoids the exit-time poller crash)
    for obj in (tc_client, svc):
        for m in ("close", "shutdown", "aclose"):
            fn = getattr(obj, m, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                break

    print(f"\ncheckpoint: {ckpt}")
    print(f"manifest:   {out}/{args.tag}.json")
    print(f"\nnext:\n  python eval_arms.py --catalog {args.catalog} --catalog-size {args.catalog_size} "
          f"--queries {args.queries} --queries-strong queries_strong_valid --model {args.model} "
          f"--checkpoint '{ckpt}' --arm {args.arm} --tag {args.tag} --out results")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    import os
    os._exit(rc or 0)   # skip interpreter teardown; everything is already written