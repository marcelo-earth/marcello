"""Run generate_base_negatives.py on a Modal GPU and save the JSONL locally (issue #12).

Usage:
    modal run scripts/modal_generate_base_negatives.py
    modal run scripts/modal_generate_base_negatives.py --limit 8   # smoke test
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "/root/marcello"
OUTPUT = "data/raw/negative_samples/base_generations_gpu.jsonl"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0,<3.0.0",
        "transformers>=4.40.0,<6.0.0",
        "datasets>=2.18.0",
        "accelerate>=0.28.0",
        "rich>=13.0",
        "sentencepiece>=0.2.1",
    )
    .add_local_dir(ROOT / "src", f"{REMOTE}/src")
    .add_local_dir(ROOT / "scripts", f"{REMOTE}/scripts")
    .add_local_dir(ROOT / "data/processed/train", f"{REMOTE}/data/processed/train")
)

app = modal.App("marcello-base-negatives", image=image)


@app.function(gpu="A10G", timeout=60 * 60)
def generate(limit: int | None = None) -> str:
    cmd = [
        "python",
        "scripts/generate_base_negatives.py",
        "--output",
        OUTPUT,
        "--batch-size",
        "16",
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    subprocess.run(
        cmd, cwd=REMOTE, env={"PYTHONPATH": "src", "PATH": "/usr/local/bin:/usr/bin"}, check=True
    )
    return (Path(REMOTE) / OUTPUT).read_text(encoding="utf-8")


@app.local_entrypoint()
def main(limit: int = 0):
    text = generate.remote(limit or None)
    out = ROOT / OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Saved {len(text.splitlines())} generations to {out}")
