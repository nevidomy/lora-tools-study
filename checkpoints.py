#!/usr/bin/env python3
"""Tinker checkpoint cleanup via the SDK RestClient, in three gated stages. Dry-run by default.

  python checkpoints.py                         # stage 1: classify, print keep/delete with sizes
  python checkpoints.py --download ./adapters   # stage 2: download the KEEP set (archives + extraction)
  python checkpoints.py --download ./adapters --delete
                                                # stage 3: delete everything NOT kept — only after every keep
                                                #          downloaded; asks you to type DELETE

Keep set = checkpoints whose tinker path matches --keep (default: the 64-tool study checkpoints).
Training-state checkpoints (`*_state`, `/weights/`) are never kept.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

DEFAULT_KEEP = (r"sampler_weights/cat64_("
               r"mix50_C_lr5e-4dec(_r1)?_s[012]"
               r"|mix100_C_lr5e-4dec_s0"
               r"|ex100_D_lr5e-4dec_ep1_s0"
               r"|mix50_C_lr5e-4dec_Inkling_Small_s[01])$")


def to_plain(obj):
    """SDK objects -> plain dicts/lists so we can walk them."""
    if hasattr(obj, "model_dump"):
        return to_plain(obj.model_dump())
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, int, float, bool)):
        return {k: to_plain(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return obj


def find_checkpoints(obj, out):
    if isinstance(obj, dict):
        path = next((v for v in obj.values() if isinstance(v, str) and v.startswith("tinker://")), None)
        if path:
            size = next((v for k, v in obj.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                         and re.search(r"size|bytes", k, re.I)), None)
            out.append({"path": path, "size": size, "raw": obj})
        for v in obj.values():
            find_checkpoints(v, out)
    elif isinstance(obj, list):
        for v in obj:
            find_checkpoints(v, out)


def human(n):
    if n is None:
        return "     ?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:6.1f} {unit}"
        n /= 1024
    return f"{n:6.1f} TB"


def download(rest, path: str, out_dir: Path) -> bool:
    name = path.rsplit("/", 1)[-1]
    dest = out_dir / name
    if dest.exists() and any(dest.iterdir()):
        print(f"  have    {name}"); return True
    print(f"  fetch   {name} ...", end="", flush=True)
    try:
        resp = rest.get_checkpoint_archive_url_from_tinker_path(path).result()
        url = getattr(resp, "url", None) or to_plain(resp).get("url")
        archive = out_dir / f"{name}.archive"
        with urllib.request.urlopen(url) as r, open(archive, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        dest.mkdir(exist_ok=True)
        try:
            with tarfile.open(archive) as tf:
                tf.extractall(dest)
            archive.unlink()
        except tarfile.ReadError:
            archive.rename(dest / "archive.bin")   # not a tar; keep the raw archive
        print(f" ok ({human(sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()))})")
        return True
    except Exception as e:  # noqa: BLE001
        print(f" FAILED: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", default=DEFAULT_KEEP)
    ap.add_argument("--download", default=None, help="directory to download the keep set into")
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dump-listing", action="store_true", help="print the raw listing (for debugging the shape)")
    args = ap.parse_args()

    import tinker
    rest = tinker.ServiceClient().create_rest_client()
    listing = to_plain(rest.list_user_checkpoints().result())
    if args.dump_listing:
        print(json.dumps(listing, indent=1, default=str)[:3000]); return 0
    cps: list[dict] = []
    find_checkpoints(listing, cps)
    seen = set()
    cps = [c for c in cps if not (c["path"] in seen or seen.add(c["path"]))]
    if not cps:
        raise SystemExit("no tinker:// paths found; run with --dump-listing and paste the output")

    keep_re = re.compile(args.keep)
    keep = [c for c in cps if keep_re.search(c["path"]) and "_state" not in c["path"] and "/weights/" not in c["path"]]
    drop = [c for c in cps if c not in keep]
    tot = lambda xs: sum(c["size"] or 0 for c in xs)

    print(f"{len(cps)} checkpoints total\n")
    print(f"KEEP ({len(keep)}, {human(tot(keep))}):")
    for c in sorted(keep, key=lambda c: c["path"]):
        print(f"  {human(c['size'])}  {c['path']}")
    print(f"\nDELETE ({len(drop)}, {human(tot(drop))}):")
    for c in sorted(drop, key=lambda c: -(c["size"] or 0))[:15]:
        print(f"  {human(c['size'])}  {c['path']}")
    if len(drop) > 15:
        print(f"  ... and {len(drop) - 15} more")
    if not args.download:
        print("\n(dry run) add --download DIR to fetch the keep set; add --delete after that to remove the rest")
        return 0

    out = Path(args.download); out.mkdir(parents=True, exist_ok=True)
    ok = [download(rest, c["path"], out) for c in keep]
    (out / "MANIFEST.json").write_text(json.dumps([c["raw"] for c in keep], indent=1, default=str))
    if not all(ok):
        print(f"\n{ok.count(False)} download(s) failed; not deleting anything."); return 1
    print(f"\nall {len(keep)} kept checkpoints present in {out}")
    if not args.delete:
        return 0
    if not args.yes and input(f"\nDelete {len(drop)} checkpoints ({human(tot(drop))}) from Tinker? Permanent. [type DELETE] ").strip() != "DELETE":
        print("aborted"); return 0
    failed = 0
    for i, c in enumerate(drop, 1):
        try:
            rest.delete_checkpoint_from_tinker_path(c["path"]).result()
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"  failed: {c['path']}: {e}")
        if i % 10 == 0 or i == len(drop):
            print(f"  deleted {i - failed}/{len(drop)}")
    print("done. `tinker checkpoint list` to confirm.")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    import os; os._exit(rc or 0)