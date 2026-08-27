"""Reward-hacking audit for top-k completions (Issue #15).

Analyzes GRPO run outputs to detect classifier gaming and reward hacking patterns.
Identifies cases where total reward rises while style components stay flat,
indicating length bonus or other shortcuts are driving optimization.

Usage:
    python scripts/audit_rewards.py --generations outputs/eval/run.json --top-k 10
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_generations(path: str) -> list[dict]:
    """Load generations from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return data.get('generations', data.get('completions', []))
    return []


def extract_ngrams(text: str, n: int = 3) -> list[tuple[str, ...]]:
    """Extract word-level n-grams from text."""
    words = text.lower().split()
    return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]


def find_shared_phrases(completions: list[str], min_count: int = 2, ngram_size: int = 4) -> list[tuple[str, int]]:
    """Find n-grams shared across multiple completions."""
    all_ngrams: Counter = Counter()
    for comp in completions:
        ngrams = set(extract_ngrams(comp, ngram_size))
        all_ngrams.update(ngrams)
    
    shared = [(ng, count) for ng, count in all_ngrams.items() if count >= min_count]
    shared.sort(key=lambda x: x[1], reverse=True)
    return shared[:10]


def analyze_completion(gen: dict) -> dict:
    """Extract and structure completion data for analysis."""
    completion_text = gen.get('completion', gen.get('text', gen.get('output', '')))
    
    # Extract reward breakdown if available
    breakdown = gen.get('reward_breakdown', gen.get('breakdown', {}))
    total_reward = gen.get('total_reward', gen.get('reward', 0))
    
    return {
        'text': completion_text,
        'total_reward': total_reward,
        'raw_style_prob': breakdown.get('raw_style_prob', breakdown.get('style', 0)),
        'length_bonus': breakdown.get('length_bonus', breakdown.get('length', 0)),
        'format_score': breakdown.get('format_score', breakdown.get('format', 0)),
        'other_components': {k: v for k, v in breakdown.items() 
                            if k not in ('raw_style_prob', 'style', 'length_bonus', 'length', 'format_score', 'format')},
        'length_tokens': len(completion_text.split())
    }


def detect_hacking_pattern(analyses: list[dict]) -> dict:
    """Detect reward hacking: total reward rising while style stays flat."""
    if len(analyses) < 2:
        return {'detected': False, 'reason': 'Insufficient data'}
    
    # Sort by total reward descending
    sorted_by_reward = sorted(analyses, key=lambda x: x['total_reward'], reverse=True)
    
    top_half = sorted_by_reward[:len(sorted_by_reward)//2]
    bottom_half = sorted_by_reward[len(sorted_by_reward)//2:]
    
    avg_top_reward = sum(a['total_reward'] for a in top_half) / len(top_half)
    avg_bottom_reward = sum(a['total_reward'] for a in bottom_half) / len(bottom_half)
    
    avg_top_style = sum(a['raw_style_prob'] for a in top_half) / len(top_half)
    avg_bottom_style = sum(a['raw_style_prob'] for a in bottom_half) / len(bottom_half)
    
    avg_top_length = sum(a['length_bonus'] for a in top_half) / len(top_half)
    avg_bottom_length = sum(a['length_bonus'] for a in bottom_half) / len(bottom_half)
    
    # Hacking pattern: reward gap > 0.3 but style gap < 0.1 and length gap > 0.2
    reward_gap = avg_top_reward - avg_bottom_reward
    style_gap = abs(avg_top_style - avg_bottom_style)
    length_gap = avg_top_length - avg_bottom_length
    
    detected = reward_gap > 0.3 and style_gap < 0.1 and length_gap > 0.2
    
    return {
        'detected': detected,
        'reward_gap': round(reward_gap, 4),
        'style_gap': round(style_gap, 4),
        'length_gap': round(length_gap, 4),
        'avg_top_reward': round(avg_top_reward, 4),
        'avg_bottom_reward': round(avg_bottom_reward, 4),
        'avg_top_style': round(avg_top_style, 4),
        'avg_bottom_style': round(avg_bottom_style, 4),
        'avg_top_length': round(avg_top_length, 4),
        'avg_bottom_length': round(avg_bottom_length, 4)
    }


def main():
    parser = argparse.ArgumentParser(description="Audit reward hacking in top-k completions")
    parser.add_argument('--generations', required=True, help='Path to generations JSON')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top/bottom completions to show')
    args = parser.parse_args()
    
    if not Path(args.generations).exists():
        print(f"ERROR: File not found: {args.generations}")
        print("This script requires actual GRPO run output.")
        return
    
    generations = load_generations(args.generations)
    if not generations:
        print("ERROR: No generations loaded")
        return
    
    analyses = [analyze_completion(g) for g in generations]
    analyses.sort(key=lambda x: x['total_reward'], reverse=True)
    
    print("=" * 80)
    print(f"REWARD HACKING AUDIT — {len(analyses)} completions analyzed")
    print("=" * 80)
    
    # Show top-k highest reward
    print(f"\n🏆 TOP {args.top_k} BY TOTAL REWARD:")
    print("-" * 80)
    for i, a in enumerate(analyses[:args.top_k]):
        preview = a['text'][:100].replace('\n', ' ')
        print(f"{i+1:2d}. Reward={a['total_reward']:+.4f} | Style={a['raw_style_prob']:.4f} | "
              f"LenBonus={a['length_bonus']:.4f} | Tokens={a['length_tokens']}")
        print(f"    \"{preview}...\"")
    
    # Show bottom-k lowest reward
    print(f"\n📉 BOTTOM {args.top_k} BY TOTAL REWARD:")
    print("-" * 80)
    for i, a in enumerate(analyses[-args.top_k:]):
        preview = a['text'][:100].replace('\n', ' ')
        idx = len(analyses) - args.top_k + i + 1
        print(f"{idx:2d}. Reward={a['total_reward']:+.4f} | Style={a['raw_style_prob']:.4f} | "
              f"LenBonus={a['length_bonus']:.4f} | Tokens={a['length_tokens']}")
        print(f"    \"{preview}...\"")
    
    # Hacking detection
    hack_result = detect_hacking_pattern(analyses)
    print("\n🔍 HACKING PATTERN ANALYSIS:")
    print("-" * 80)
    for k, v in hack_result.items():
        print(f"  {k:25s}: {v}")
    
    if hack_result['detected']:
        print("\n⚠️  WARNING: Potential reward hacking detected!")
        print("   Total reward increases while style score stays flat → length bonus may be dominating.")
    else:
        print("\n✓ No obvious reward hacking pattern detected.")
    
    # Shared phrases
    top_texts = [a['text'] for a in analyses[:args.top_k]]
    shared = find_shared_phrases(top_texts, min_count=max(2, args.top_k // 3))
    
    if shared:
        print(f"\n🔗 SHARED PHRASES IN TOP-{args.top_k} (potential hacks):")
        print("-" * 80)
        for phrase, count in shared:
            phrase_str = ' '.join(phrase)
            print(f"  [{count}/{args.top_k}] \"{phrase_str}\"")
    else:
        print(f"\n✓ No suspiciously shared phrases in top-{args.top_k}.")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
