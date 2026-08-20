"""Verify completion length distribution is not pinned at max_new_tokens.

Addresses Issue #30: Analyzes GRPO run outputs to confirm length bonus
is not acting as monotone longer-is-better signal.

Usage:
    python scripts/verify_completion_length.py --completions outputs/grpo_run/completions.jsonl
    python scripts/verify_completion_length.py --completions outputs/grpo_run/completions.jsonl --max-new-tokens 256 --target 60
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

def load_completions(path: str) -> list[str]:
    """Load completions from JSONL file."""
    completions = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Handle both raw string and structured formats
                if isinstance(data, str):
                    completions.append(data)
                elif isinstance(data, dict):
                    text = data.get('completion', data.get('text', data.get('output', '')))
                    if text:
                        completions.append(text)
    return completions

def analyze_lengths(completions: list[str], max_new_tokens: int, target: int) -> dict:
    """Analyze token length distribution of completions."""
    # Simple token estimation: split on whitespace + punctuation
    # For production, use the actual tokenizer
    lengths = [len(c.split()) for c in completions]
    
    if not lengths:
        return {"error": "No completions found"}
    
    total = len(lengths)
    at_cap = sum(1 for l in lengths if l >= max_new_tokens * 0.9)  # Within 10% of cap
    near_target = sum(1 for l in lengths if abs(l - target) <= target * 0.5)
    
    result = {
        "total_completions": total,
        "min_length": min(lengths),
        "max_length": max(lengths),
        "median_length": statistics.median(lengths),
        "mean_length": statistics.mean(lengths),
        "p90_length": sorted(lengths)[int(total * 0.9)] if total > 0 else 0,
        "at_or_near_cap_count": at_cap,
        "at_or_near_cap_pct": (at_cap / total) * 100,
        "near_target_count": near_target,
        "near_target_pct": (near_target / total) * 100,
        "target": target,
        "max_new_tokens": max_new_tokens,
        "verdict": "PASS" if at_cap < total * 0.3 and near_target > total * 0.2 else "FAIL"
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Verify completion length distribution")
    parser.add_argument("--completions", required=True, help="Path to completions JSONL")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max generation tokens")
    parser.add_argument("--target", type=int, default=60, help="Target length from corpus median")
    args = parser.parse_args()
    
    if not Path(args.completions).exists():
        print(f"ERROR: Completions file not found: {args.completions}")
        print("This script requires actual GRPO run outputs.")
        print("Run train_grpo.py first to generate completions.")
        return
    
    completions = load_completions(args.completions)
    if not completions:
        print("ERROR: No completions loaded")
        return
    
    result = analyze_lengths(completions, args.max_new_tokens, args.target)
    
    print("=" * 60)
    print("COMPLETION LENGTH DISTRIBUTION ANALYSIS")
    print("=" * 60)
    for k, v in result.items():
        print(f"{k:30s}: {v}")
    print("=" * 60)
    
    if result["verdict"] == "PASS":
        print("✓ Length distribution is healthy - not pinned at max_new_tokens")
    else:
        print("✗ WARNING: Length distribution suggests reward hacking")
        print("  - Check if length bonus is monotone")
        print("  - Consider adjusting target or penalty shape")

if __name__ == "__main__":
    main()
