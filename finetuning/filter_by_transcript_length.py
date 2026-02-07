#!/usr/bin/env python3
"""Filter JSONL dataset by transcript length (parallel processing).

Removes samples with transcripts exceeding a character/token limit or with corrupted data.
This prevents OOM errors during training when batch_size=1 still fails.

Usage:
    # Filter by character count (fast, no model loading):
    python filter_by_transcript_length.py \
        --input_file "train.jsonl" \
        --output_file "train_filtered.jsonl" \
        --max_chars 2000 \
        --num_proc 64

    # Filter by token count (more accurate, requires tokenizer):
    python filter_by_transcript_length.py \
        --input_file "train.jsonl" \
        --output_file "train_filtered.jsonl" \
        --max_tokens 1024 \
        --model_path "Qwen/Qwen3-ASR-1.7B" \
        --num_proc 64
"""

import argparse
import json
import multiprocessing as mp
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

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


def has_excessive_repetition(text: str, max_repeat: int = 10) -> bool:
    """
    Check if text has excessive character repetition (corrupted data).
    
    Returns True if any character is repeated more than max_repeat times consecutively.
    """
    if not text:
        return False
    
    pattern = r'(.)\1{' + str(max_repeat) + r',}'
    return bool(re.search(pattern, text))


def process_shard(
    shard_idx: int,
    input_file: str,
    start_line: int,
    end_line: int,
    output_dir: str,
    max_chars: int,
    max_tokens: int,
    model_path: str,
    max_repeat: int,
    save_rejected: bool
) -> Tuple[int, int, int, int, List[Dict]]:
    """
    Process a shard of the input file.
    
    Returns:
        (shard_idx, kept, rejected, rejected_repetition, rejected_samples)
    """
    shard_path = f"{output_dir}/shard_{shard_idx:04d}.jsonl"
    
    # Load tokenizer if needed (each worker loads its own)
    tokenizer = None
    if max_tokens and model_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    kept = 0
    rejected = 0
    rejected_repetition = 0
    rejected_samples = []
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(shard_path, 'w', encoding='utf-8') as f_out:
        
        for i, line in enumerate(f_in):
            if i < start_line:
                continue
            if i >= end_line:
                break
            
            sample = json.loads(line)
            text = sample.get("text", "")
            transcript = extract_transcript(text)
            
            keep = True
            reject_reason = None
            char_count = len(transcript)
            token_count = None
            
            # Check for corrupted data (excessive repetition)
            if has_excessive_repetition(transcript, max_repeat):
                keep = False
                reject_reason = "repetition"
                rejected_repetition += 1
            
            # Check character limit
            if keep and max_chars and char_count > max_chars:
                keep = False
                reject_reason = "char_limit"
            
            # Check token limit
            if keep and max_tokens and tokenizer:
                tokens = tokenizer.encode(transcript, add_special_tokens=False)
                token_count = len(tokens)
                if token_count > max_tokens:
                    keep = False
                    reject_reason = "token_limit"
            
            if keep:
                f_out.write(line)
                kept += 1
            else:
                rejected += 1
                
                if save_rejected:
                    rejected_samples.append({
                        "audio": sample.get("audio", ""),
                        "char_count": char_count,
                        "token_count": token_count,
                        "reason": reject_reason,
                        "transcript_preview": transcript[:200] + "..." if len(transcript) > 200 else transcript
                    })
    
    return shard_idx, kept, rejected, rejected_repetition, rejected_samples


def main():
    parser = argparse.ArgumentParser(
        description="Filter JSONL by transcript length (parallel)"
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
        "--max_repeat",
        type=int,
        default=10,
        help="Max consecutive repeated characters allowed (default: 10). Detects corrupted data."
    )
    parser.add_argument(
        "--rejected_file",
        type=str,
        default=None,
        help="Optional file to save rejected samples for analysis"
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=64,
        help="Number of parallel processes (default: 64)"
    )
    
    args = parser.parse_args()
    
    if args.max_chars is None and args.max_tokens is None:
        logger.error("Must specify --max_chars or --max_tokens")
        return
    
    if args.max_tokens and not args.model_path:
        logger.error("--model_path required when using --max_tokens")
        return
    
    logger.info("=" * 60)
    logger.info("FILTER BY TRANSCRIPT LENGTH (PARALLEL)")
    logger.info("=" * 60)
    logger.info(f"Input:      {args.input_file}")
    logger.info(f"Output:     {args.output_file}")
    if args.max_chars:
        logger.info(f"Max chars:  {args.max_chars}")
    if args.max_tokens:
        logger.info(f"Max tokens: {args.max_tokens}")
        logger.info(f"Model:      {args.model_path}")
    logger.info(f"Max repeat: {args.max_repeat} (consecutive chars)")
    logger.info(f"Num proc:   {args.num_proc}")
    
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temp directory for shards
    temp_dir = output_path.parent / ".temp_filter_shards"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Count input lines
        logger.info("Counting input lines...")
        with open(input_path, 'r') as f:
            total_lines = sum(1 for _ in f)
        logger.info(f"Total samples: {total_lines:,}")
        
        # Calculate shard ranges
        num_shards = min(args.num_proc, total_lines)
        lines_per_shard = total_lines // num_shards
        remainder = total_lines % num_shards
        
        shard_ranges = []
        start = 0
        for i in range(num_shards):
            end = start + lines_per_shard + (1 if i < remainder else 0)
            shard_ranges.append((i, start, end))
            start = end
        
        # Process shards in parallel
        logger.info(f"Processing {num_shards} shards with {args.num_proc} processes...")
        
        with mp.Pool(processes=args.num_proc) as pool:
            process_args = [
                (idx, str(input_path), start, end, str(temp_dir), 
                 args.max_chars, args.max_tokens, args.model_path,
                 args.max_repeat, args.rejected_file is not None)
                for idx, start, end in shard_ranges
            ]
            results = list(tqdm(
                pool.starmap(process_shard, process_args),
                total=num_shards,
                desc="Filtering shards"
            ))
        
        # Aggregate results
        total_kept = sum(r[1] for r in results)
        total_rejected = sum(r[2] for r in results)
        total_rejected_repetition = sum(r[3] for r in results)
        all_rejected_samples = []
        for r in results:
            all_rejected_samples.extend(r[4])
        
        # Merge shards into final file
        logger.info(f"Merging {num_shards} shards into {output_path}...")
        shard_files = sorted(temp_dir.glob("shard_*.jsonl"))
        
        with open(output_path, 'wb') as f_out:
            for shard_file in tqdm(shard_files, desc="Merging"):
                with open(shard_file, 'rb') as f_in:
                    shutil.copyfileobj(f_in, f_out)
                shard_file.unlink()
        
        # Save rejected samples if requested
        if args.rejected_file and all_rejected_samples:
            all_rejected_samples.sort(key=lambda x: x["char_count"], reverse=True)
            
            with open(args.rejected_file, 'w', encoding='utf-8') as f:
                for sample in all_rejected_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"Saved {len(all_rejected_samples)} rejected samples to {args.rejected_file}")
        
    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Summary
    logger.info("=" * 60)
    logger.info("FILTERING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total input:   {total_lines:,}")
    logger.info(f"Kept:          {total_kept:,} ({100*total_kept/total_lines:.2f}%)")
    logger.info(f"Rejected:      {total_rejected:,} ({100*total_rejected/total_lines:.2f}%)")
    if total_rejected_repetition > 0:
        logger.info(f"  - Repetition: {total_rejected_repetition:,} (corrupted data)")
    logger.info(f"Output file:   {args.output_file}")


if __name__ == "__main__":
    main()
