#!/usr/bin/env python3
"""Per-tool analysis: where does the in-context 'context tax' land, and does the adapter pay it?

Joins, per tool:
  single   base model, only this tool's schema in context   (argument-filling ability)
  catalog  base model, all N schemas in context              (deployment reality)
  adapter  mean over seeds of the chosen adapter config      (no schema)
plus covariates from the catalog (verb, #params, #enum values, closed_vocab) and a confusability
proxy: Jaccard similarity of each tool's word set (name + description + param names + enum values)
to its nearest neighbour in the catalog.

Also: failure-class decomposition of the catalog run per tool, the confusion pairs behind wrong-tool
errors, and Spearman rank correlations between covariates and outcomes.

Usage:
  python per_tool.py --catalog tools/catalog_64.json --catalog-size 64 \\
      --model openai_gpt_oss_20b --adapter 'cat64_mix50_C_lr5e-4dec_s*' [--csv out.csv]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import statistics as st
from pathlib import Path

STOP = {"the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "with", "this", "that", "when",
        "call", "use", "tool", "agent", "should", "is", "are", "be", "by", "at", "from", "as", "it"}


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOP}


def tool_wordset(t: dict) -> set[str]:
    m = t["mcp"]
    ws = words(m["name"].replace("_", " ")) | words(m["description"])
    for pname, spec in m["inputSchema"]["properties"].items():
        ws |= words(pname.replace("_", " ")) | words(spec.get("description", ""))
        for v in spec.get("enum", []):
            ws |= words(v.replace("_", " "))
    return ws


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def load_rows(path: str, arm: str, set_name: str) -> list[dict]:
    out = []
    for l in open(path):
        r = json.loads(l)
        if r["arm"] == arm and r["set"] == set_name:
            out.append(r)
    return out


def classify(r: dict, props: dict) -> str:
    sc = r["score"]
    if not sc["parsed"]:
        return "no_call"
    if not sc["tool_ok"]:
        return "wrong_tool"
    if not sc["schema_valid"]:
        return "invention"
    return "wrong_value" if not sc["args_exact"] else "exact"


def per_tool_acc(rows: list[dict]) -> dict[str, float]:
    d = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        d[r["tool"]][0] += r["score"]["args_exact"]; d[r["tool"]][1] += 1
    return {t: a / n for t, (a, n) in d.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="tools/catalog_64.json")
    ap.add_argument("--catalog-size", type=int, default=64)
    ap.add_argument("--results", default="results")
    ap.add_argument("--model", default="openai_gpt_oss_20b", help="model stem as it appears in results filenames")
    ap.add_argument("--adapter", default="cat64_mix50_C_lr5e-4dec_s*", help="glob for adapter tags (seeds averaged)")
    ap.add_argument("--adapter-arm", default="C")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    N = args.catalog_size
    cat = json.loads(Path(args.catalog).read_text())["tools"][:N]
    by_name = {t["mcp"]["name"]: t for t in cat}
    wsets = {t["mcp"]["name"]: tool_wordset(t) for t in cat}

    # ---- covariates ----
    cov = {}
    for t in cat:
        name = t["mcp"]["name"]
        props = t["mcp"]["inputSchema"]["properties"]
        nn_sim, nn = max(((jaccard(wsets[name], wsets[o]), o) for o in wsets if o != name), default=(0, ""))
        cov[name] = {
            "verb": t["skeleton"]["verb"],
            "verb_shared": sum(1 for o in cat if o["skeleton"]["verb"] == t["skeleton"]["verb"]) - 1,
            "n_params": len(props),
            "n_enum_values": sum(len(s.get("enum", [])) for s in props.values()),
            "closed_vocab": t["closed_vocab"],
            "nn_sim": round(nn_sim, 3), "nn_tool": nn,
            "catalog_index": t["catalog_index"],
        }

    # ---- runs ----
    pre = f"{args.results}/{args.model}__cat{N}__"
    f_single = f"{pre}base_pv1_single.rows.jsonl"
    f_cat = f"{pre}base_pv1.rows.jsonl"
    f_cat_unseen = f"{pre}base_unseen_pv1.rows.jsonl"
    adapters = sorted(f for f in glob.glob(f"{pre}{args.adapter}_pv1.rows.jsonl") if "_unseen" not in f)
    adapters_unseen = sorted(glob.glob(f"{pre}{args.adapter}_unseen_pv1.rows.jsonl"))
    for f in (f_single, f_cat):
        if not Path(f).exists():
            raise SystemExit(f"missing {f}")

    single = per_tool_acc(load_rows(f_single, "A", "natural"))
    single_un = per_tool_acc(load_rows(f_single, "A", "strong"))
    catalog_rows = load_rows(f_cat, "A", "natural")
    catalog = per_tool_acc(catalog_rows)
    catalog_un = per_tool_acc(load_rows(f_cat_unseen, "A", "strong")) if Path(f_cat_unseen).exists() else {}

    def seed_mean(files, arm, set_name):
        accs = [per_tool_acc(load_rows(f, arm, set_name)) for f in files]
        return {t: st.mean(a[t] for a in accs if t in a) for t in by_name if any(t in a for a in accs)}
    adapter = seed_mean(adapters, args.adapter_arm, "natural") if adapters else {}
    adapter_un = seed_mean(adapters_unseen, args.adapter_arm, "strong") if adapters_unseen else {}
    print(f"single: {f_single.split('/')[-1]}\ncatalog: {f_cat.split('/')[-1]}\nadapter: {len(adapters)} seed file(s) matching {args.adapter}\n")

    # ---- failure classes + confusion pairs for the catalog run ----
    fc = collections.defaultdict(collections.Counter)
    conf = collections.Counter()
    for r in catalog_rows:
        c = classify(r, by_name[r["tool"]]["mcp"]["inputSchema"]["properties"])
        fc[r["tool"]][c] += 1
        if c == "wrong_tool":
            conf[(r["tool"], r["pred_name"])] += 1

    # ---- table ----
    rows = []
    for name in by_name:
        c = cov[name]
        s, k = single.get(name, float("nan")), catalog.get(name, float("nan"))
        a = adapter.get(name, float("nan"))
        rows.append({"tool": name, **c, "single": s, "catalog": k, "ctx_loss": s - k,
                     "adapter": a, "adapter_minus_single": a - s if adapters else float("nan"),
                     "single_unseen": single_un.get(name, float("nan")),
                     "catalog_unseen": catalog_un.get(name, float("nan")),
                     "adapter_unseen": adapter_un.get(name, float("nan")),
                     **{f"fc_{k_}": fc[name][k_] for k_ in ("no_call", "wrong_tool", "invention", "wrong_value")}})
    rows.sort(key=lambda r: -r["ctx_loss"])

    print(f"{'tool':34s} {'verb':10s} {'nn_sim':>6s} {'vocab':>5s} {'single':>6s} {'catalog':>7s} {'loss':>5s} "
          f"{'adapt':>5s} {'a-s':>5s} {'un_s':>5s} {'un_c':>5s} {'un_a':>5s}  fail classes (catalog)")
    def line(r):
        fcs = " ".join(f"{k_[3:]}={r[k_]}" for k_ in ("fc_no_call", "fc_wrong_tool", "fc_invention", "fc_wrong_value") if r[k_])
        print(f"{r['tool']:34s} {r['verb']:10s} {r['nn_sim']:6.2f} {r['closed_vocab']:5d} {r['single']:6.2f} {r['catalog']:7.2f} "
              f"{r['ctx_loss']:+5.2f} {r['adapter']:5.2f} {r['adapter_minus_single']:+5.2f} "
              f"{r['single_unseen']:5.2f} {r['catalog_unseen']:5.2f} {r['adapter_unseen']:5.2f}  {fcs}")
    for r in rows[: args.top]:
        line(r)
    print("   ...")
    for r in rows[-4:]:
        line(r)

    # ---- confusion pairs ----
    print(f"\ntop confusion pairs in the catalog run (truth -> predicted, count):")
    for (t, p), n in conf.most_common(10):
        sim = jaccard(wsets[t], wsets.get(p, set()))
        print(f"  {t:34s} -> {str(p):34s} {n:3d}   jaccard={sim:.2f}  same_verb={by_name[t]['skeleton']['verb'] == (by_name[p]['skeleton']['verb'] if p in by_name else None)}")

    # ---- correlations ----
    print("\nSpearman rank correlations (n=%d tools):" % len(rows))
    def corr(x, y, label):
        pairs = [(r[x], r[y]) for r in rows if r[x] == r[x] and r[y] == r[y]]
        if len(pairs) < 3:
            print(f"  {label:48s} (n/a)"); return
        xs, ys = zip(*pairs)
        print(f"  {label:48s} rho={spearman(list(xs), list(ys)):+.2f}  (n={len(pairs)})")
    corr("nn_sim", "ctx_loss", "context loss vs nearest-neighbour similarity")
    corr("verb_shared", "ctx_loss", "context loss vs #tools sharing the verb")
    corr("closed_vocab", "ctx_loss", "context loss vs closed_vocab")
    corr("n_params", "ctx_loss", "context loss vs #params")
    corr("catalog_index", "ctx_loss", "context loss vs position in catalog")
    if adapters:
        corr("closed_vocab", "adapter", "adapter natural vs closed_vocab")
        corr("nn_sim", "adapter", "adapter natural vs nearest-neighbour similarity")
        corr("single", "adapter", "adapter natural vs single-schema (same tool)")
    if adapters_unseen:
        corr("closed_vocab", "adapter_unseen", "adapter unseen vs closed_vocab")
        corr("n_enum_values", "adapter_unseen", "adapter unseen vs #enum values")
        corr("single_unseen", "adapter_unseen", "adapter unseen vs single-schema unseen")
    tot = collections.Counter()
    for c_ in fc.values():
        tot.update(c_)
    n_all = sum(tot.values())
    print("\ncatalog-run failure classes overall:", {k: f"{v/n_all:.3f}" for k, v in tot.items() if k != "exact"})

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())