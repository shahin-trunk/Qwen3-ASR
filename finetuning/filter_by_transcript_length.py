#!/usr/bin/env python3
"""Filter JSONL dataset by transcript length.

Removes samples with transcripts exceeding a character or token limit.
This prevents OOM errors during training when batch_size=1 still fails.

Usage:
    # Filter by character count (fast, no model loading):
    python filter_by_transcript_length.py \
        --input_file "train.jsonl" \
        --output_file "train_filtered.jsonl" \
        --max_chars 2000

    # Filter by token count (more accurate, requires tokenizer):
    python filter_by_transcript_length.py \
        --input_file "train.jsonl" \
        --output_file "train_filtered.jsonl" \
        --max_tokens 1024 \
        --model_path "Qwen/Qwen3-ASR-1.7B"
"""

import argparse
import json
import re
from pathlib import Path

from loguru import logger
from tqdm import tqdm


def extract_transcript(text: str) -> str:
    """
    Extract the actual transcript from the Qwen3-ASR format.
    
    Format: "language {LANG}<asr_text>{transcript}"
    """
    match = re.search(r"<asr_text>(.*)$", text)
    if match:
        return match.group(1)
    return text


def count_chars(text: str) -> int:
    """Count characters in transcript."""
    return len(extract_transcript(text))


def has_excessive_repetition(text: str, max_repeat: int = 10) -> bool:
    """
    Check if text has excessive character repetition (corrupted data).
    
    Returns True if any character is repeated more than max_repeat times consecutively.
    """
    if not text:
        return False
    
    # Match any character repeated more than max_repeat times
    pattern = r'(.)\1{' + str(max_repeat) + r',}'
    return bool(re.search(pattern, text))


def main():
    parser = argparse.ArgumentParser(
        description="Filter JSONL by transcript length"
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Input JSONL file"
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output filtered JSONL file"
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=None,
        help="Maximum transcript character count (default: None)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum transcript token count (requires --model_path)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Model path for tokenizer (required if using --max_tokens)"
    )
    parser.add_argument(
        "--rejected_file",
        type=str,
        default=None,
        help="Optional file to save rejected samples for analysis"
    )
    parser.add_argument(
        "--max_repeat",
        type=int,
        default=10,
        help="Max consecutive repeated characters allowed (default: 10). Detects corrupted data."
    )
    
    args = parser.parse_args()
    
    if args.max_chars is None and args.max_tokens is None:
        logger.error("Must specify --max_chars or --max_tokens")
        return
    
    if args.max_tokens and not args.model_path:
        logger.error("--model_path required when using --max_tokens")
        return
    
    logger.info("=" * 60)
    logger.info("FILTER BY TRANSCRIPT LENGTH")
    logger.info("=" * 60)
    logger.info(f"Input:      {args.input_file}")
    logger.info(f"Output:     {args.output_file}")
    if args.max_chars:
        logger.info(f"Max chars:  {args.max_chars}")
    if args.max_tokens:
        logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Max repeat: {args.max_repeat} (consecutive chars)")
    
    # Load tokenizer if needed
    tokenizer = None
    if args.max_tokens:
        logger.info(f"Loading tokenizer from {args.model_path}...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Count input lines
    input_path = Path(args.input_file)
    with open(input_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    logger.info(f"Total samples: {total_lines:,}")
    
    # Process
    kept = 0
    rejected = 0
    rejected_repetition = 0
    rejected_samples = []
    
    # Track stats for rejected samples
    max_rejected_chars = 0
    max_rejected_tokens = 0
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, total=total_lines, desc="Filtering"):
            sample = json.loads(line)
            text = sample.get("text", "")
            transcript = extract_transcript(text)
            
            keep = True
            reject_reason = None
            char_count = len(transcript)
            token_count = None
            
            # Check for corrupted data (excessive repetition)
            if has_excessive_repetition(transcript, args.max_repeat):
                keep = False
                reject_reason = "repetition"
                rejected_repetition += 1
            
            # Check character limit
            if keep and args.max_chars and char_count > args.max_chars:
                keep = False
                reject_reason = "char_limit"
            
            # Check token limit
            if keep and args.max_tokens and tokenizer:
                tokens = tokenizer.encode(transcript, add_special_tokens=False)
                token_count = len(tokens)
                if token_count > args.max_tokens:
                    keep = False
                    reject_reason = "token_limit"
            
            if keep:
                f_out.write(line)
                kept += 1
            else:
                rejected += 1
                max_rejected_chars = max(max_rejected_chars, char_count)
                if token_count:
                    max_rejected_tokens = max(max_rejected_tokens, token_count)
                
                if args.rejected_file:
                    rejected_samples.append({
                        "audio": sample.get("audio", ""),
                        "char_count": char_count,
                        "token_count": token_count,
                        "reason": reject_reason,
                        "transcript_preview": transcript[:200] + "..." if len(transcript) > 200 else transcript
                    })
    
    # Save rejected samples if requested
    if args.rejected_file and rejected_samples:
        # Sort by length (descending)
        rejected_samples.sort(key=lambda x: x["char_count"], reverse=True)
        
        with open(args.rejected_file, 'w', encoding='utf-8') as f:
            for sample in rejected_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(rejected_samples)} rejected samples to {args.rejected_file}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("FILTERING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total input:   {total_lines:,}")
    logger.info(f"Kept:          {kept:,} ({100*kept/total_lines:.2f}%)")
    logger.info(f"Rejected:      {rejected:,} ({100*rejected/total_lines:.2f}%)")
    if rejected_repetition > 0:
        logger.info(f"  - Repetition: {rejected_repetition:,} (corrupted data)")
    logger.info(f"Output file:   {args.output_file}")
    
    if rejected > 0:
        logger.info(f"Max rejected chars:  {max_rejected_chars:,}")
        if max_rejected_tokens:
            logger.info(f"Max rejected tokens: {max_rejected_tokens:,}")


if __name__ == "__main__":
    main()
