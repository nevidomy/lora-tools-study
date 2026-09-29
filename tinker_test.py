#!/usr/bin/env python3
"""Diagnose Tinker access: auth, model availability, billing usage, live sampling.

Total spend if the final probe succeeds: ~5 output tokens (<$0.00001).
Run:  python tinker_probe.py
"""

import os
import sys
import traceback
from datetime import datetime, timedelta, timezone

import tinker

CHEAP_MODEL = "openai/gpt-oss-20b"  # $0.45/M sample


def rule(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def classify(e):
    """Map an exception to a verdict."""
    name = type(e).__name__
    status = getattr(e, "status_code", None)
    print(f"  !! {name}" + (f" (HTTP {status})" if status else ""))
    print(f"     {str(e)[:400]}")

    if isinstance(e, tinker.BillingError):
        return "CREDITS EXHAUSTED / BILLING BLOCKED"
    if isinstance(e, tinker.AuthenticationError):
        return "AUTH FAILED - key invalid or revoked"
    if isinstance(e, tinker.PermissionDeniedError):
        return "PERMISSION DENIED - org/project access issue"
    if isinstance(e, tinker.NotFoundError):
        return "NOT FOUND - model not available to this org"
    if isinstance(e, tinker.RateLimitError):
        return "RATE LIMITED - not a credit problem"
    if isinstance(e, (tinker.APIConnectionError, tinker.APITimeoutError)):
        return "NETWORK - could not reach the API"
    return f"UNCLASSIFIED ({name})"


def main():
    if not os.environ.get("TINKER_API_KEY"):
        print("TINKER_API_KEY is not set. Export it and re-run.")
        return 1

    verdicts = {}

    rule("0. Client construction")
    try:
        svc = tinker.ServiceClient()
        rest = svc.create_rest_client()
        print("  ok")
    except Exception as e:
        verdicts["construct"] = classify(e)
        print(f"\nVERDICT: {verdicts['construct']}")
        return 1

    # ---- 1. whoami: auth + org identity. No billing involved. ----
    rule("1. whoami  (auth / org identity)")
    try:
        who = rest.whoami().result()
        print(f"  {who}")
        verdicts["auth"] = "OK"
    except Exception as e:
        verdicts["auth"] = classify(e)

    # ---- 2. server capabilities: which models this org may touch ----
    rule("2. get_server_capabilities  (model access)")
    try:
        caps = svc.get_server_capabilities()
        models = [m.model_name for m in getattr(caps, "supported_models", [])]
        print(f"  {len(models)} models visible")
        for m in models[:12]:
            print(f"    - {m}")
        if len(models) > 12:
            print(f"    ... +{len(models) - 12} more")
        print(f"  {CHEAP_MODEL} visible: {CHEAP_MODEL in models}")
        verdicts["models"] = "OK" if models else "EMPTY MODEL LIST"
    except Exception as e:
        verdicts["models"] = classify(e)

    # ---- 3. billing usage across the stated grant window ----
    rule("3. get_billing_usage  (Jun 1 -> now)")
    try:
        start = datetime(2026, 6, 1, tzinfo=timezone.utc)
        # API requires UTC hour alignment on both bounds
        end = (datetime.now(timezone.utc) + timedelta(hours=2)).replace(
            minute=0, second=0, microsecond=0
        )
        usage = rest.get_billing_usage(starting_on=start, ending_before=end).result()
        print(f"  {usage}")
        verdicts["billing"] = "OK"
    except Exception as e:
        verdicts["billing"] = classify(e)

    # ---- 4. the definitive probe: actually sample 5 tokens ----
    rule(f"4. live sample  ({CHEAP_MODEL}, max_tokens=5)")
    try:
        sampler = svc.create_sampling_client(base_model=CHEAP_MODEL)
        tok = sampler.get_tokenizer()
        prompt = tinker.ModelInput.from_ints(tok.encode("The capital of France is"))
        params = tinker.SamplingParams(max_tokens=5, temperature=0.0)
        resp = sampler.sample(
            prompt=prompt, sampling_params=params, num_samples=1
        ).result()
        for seq in resp.sequences:
            print(f"  -> {tok.decode(seq.tokens)!r}  (stop={seq.stop_reason}, n={len(seq.tokens)})")
        print(f"  prompt_cache_hit_tokens={resp.prompt_cache_hit_tokens}")
        verdicts["sample"] = "OK - CREDITS ARE LIVE"
    except Exception as e:
        verdicts["sample"] = classify(e)

    rule("SUMMARY")
    for k, v in verdicts.items():
        print(f"  {k:10s} {v}")

    s = verdicts.get("sample", "")
    print()
    if s.startswith("OK"):
        print("  Sampling works. The zero balance display is not blocking API use.")
        print("  -> Proceed. Sample as much as possible before the window closes.")
    elif "CREDITS" in s:
        print("  Credits are exhausted or the grant is closed.")
        print("  -> Email tinker@thinkingmachines.ai with this output.")
        print("  -> No deadline pressure remains; replan for September.")
    elif "AUTH" in s or "PERMISSION" in s:
        print("  Access problem, not a spending problem. Recoverable via support.")
    else:
        print("  Inconclusive. Send this whole output to support.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)