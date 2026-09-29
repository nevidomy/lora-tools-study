# In-Context vs. LoRA-Adapter Tool Calling as the Catalog Grows

An agent with N tools normally carries all N schemas in every prompt. Question is: can a per-catalog LoRA adapter internalize them, emitting correct calls with no schema in context, and how does that compare as N grows?

## Quick summary of results

Tool-calling agents usually carry every tool schema in the prompt. This work measures what that costs at different catalog sizes and whether a per-catalog LoRA adapter can replace it. On 64 synthetic MCP-style tools with nested 4/16/64-tool catalogs, two models (GPT-OSS-20B; Inkling-Small, 276B/12B-active), three held-out phrasings (natural, seen paraphrases, unseen paraphrases from a disjoint glossaries), three training seeds, and tool-clustered bootstrap intervals. Findings: 

  * Full-catalog prompting degrades with catalog size — GPT-OSS-20B scores 0.890 with one schema in context and 0.583 with 64, on the same queries; most of the loss is the correct tool with a wrong argument; tool's position in the schema block matters but less then tool itself (7% position vs. 80.5% for tool identity). 
  * A word-overlap top-4 retriever recovers most of it 0.845 with no adapter (natural phrasing; 0.431 on unseen paraphrases). 
  * An adapter trained on 50 natural + 50 paraphrased examples per tool, with no schema in the prompt, scores 0.951 on natural phrasing and 0.736 on unseen paraphrases at 64 tools — +0.368 [+0.313, +0.423] and +0.491 [+0.440, +0.541] over the full catalog, +0.106 and +0.305 over retrieval — at 152 input tokens instead of 7,251, and stays stable from 4 to 64 tools at 100 examples per tool. 
  * Given the same data, fine-tuning with the schema in context is still better (though difference is small: +0.011 / +0.051 with one D seed at 64) while paying the full prefix price.
  * Adapter rank barely matters for natural phrasing: rank 1 (14M parameters) matches rank 32 within 0.01. It was shown before that adapter can beat a full-catalog promptin by Shemla et al. (2026); this work characterizes the gap against catalog size, retrieval, and matched with-schema fine-tuning.


### Four options, side by side

For a fixed 64-tool catalog on GPT-OSS-20B, ordered by input tokens per request:


| option                                  | training | natural | unseen | input tokens | notes                                                                              |
| --------------------------------------- | -------- | ------- | ------ | ------------ | ---------------------------------------------------------------------------------- |
| full catalog in context                 | none     | 0.583   | 0.243  | 7,251        | worst in this setting; position-sensitive                                          |
| fine-tune, no schema (C, mix50)         | ~$1.50   | 0.951   | 0.736  | 152          | fewest tokens                                                                      |
| top-4 lexical retrieval                 | none     | 0.845   | 0.431  | 645          | recovers most of the full-catalog loss with no training; misses 15% of paraphrases |
| fine-tune, schema in context (D, mix50) | ~$56     | 0.962   | 0.786  | 7,251        | highest observed accuracy; fewer invented enums than C; pays the full prefix       |


Details, methods, and all tables: [docs/report.md](docs/report.md) —
in particular [§3.1 the full-catalog penalty](docs/report.md#31-the-full-catalog-penalty)
and [§3.3 with vs. without the schema on the same data](docs/report.md#33-fine-tuning-with-the-schema-vs-without-on-the-same-data).

## Repository layout

- `tools_gen.py`, `queries_gen.py`, `validate.py`, `strongify.py` — data pipeline
- `eval_arms.py`, `train_sft.py`, `run_seed.sh` — harness and training
- `summarize.py`, `compare.py`, `cluster_ci.py`, `per_tool.py`, `position_analysis.py`,
`strict_rescore.py`, `eval_mcq.py` — analysis
- `configs_*.txt` — every training configuration used
- `results/` — per-run summaries and per-response rows
- `docs/report.md` — the report



## Reproducing

Requires `tinker==0.27.1`, `tinker-cookbook==0.5.7`, a Tinker API key, and an OpenRouter key
for data generation. Pipeline order: tools → queries → validate → strongify (seen, then unseen
with `--avoid-glossary`) → validate → `run_seed.sh` → `summarize.py`. Each run's manifest in
`runs/` records model, renderer, schedule, and date.

Total cost of the study: ~$242 ($205 Tinker, $37 OpenRouter).

## Related

- [Cookbook issue #947](https://github.com/thinking-machines-lab/tinker-cookbook/issues/947): the recommended GPT-OSS renderer emits malformed tool-call headers
on ~8% of requests — found while building the harness.

