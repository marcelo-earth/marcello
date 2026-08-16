"""Verify data/processed/ matches its committed manifest.

data/processed/ is gitignored except for manifest.json, so this is the only
way to tell whether a local rebuild (`make data`) produced the same corpus
the committed numbers in PLAN.md refer to, rather than drifting silently.

Usage:
    python scripts/verify_corpus.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk

from marcello.data.manifest import corpus_hash


def main():
    parser = argparse.ArgumentParser(description="Verify data/processed/ matches manifest.json")
    parser.add_argument("--path", type=str, default="data/processed")
    args = parser.parse_args()

    root = Path(args.path)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path}. Run `make data` first.")
        raise SystemExit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    train = load_from_disk(root / "train")
    val = load_from_disk(root / "val")
    current_hash = corpus_hash(list(train["text"]) + list(val["text"]))
    committed_hash = manifest["text_hash"]

    if current_hash != committed_hash:
        print(
            "CORPUS DRIFT: data/processed/{train,val} does not match "
            f"{manifest_path}.\n  committed hash: {committed_hash}\n"
            f"  current hash:   {current_hash}\n"
            "Re-run `make data` from clean raw inputs, or update the manifest "
            "if the drift is intentional."
        )
        raise SystemExit(1)

    print(f"Corpus matches {manifest_path} (hash={current_hash[:12]}...)")


if __name__ == "__main__":
    main()
