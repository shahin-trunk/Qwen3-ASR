#!/usr/bin/env python3
"""Convert parquet dataset to train.jsonl and eval.jsonl for Qwen3-ASR finetuning.

Features:
- Parallel writing using multiprocessing Pool
- Train and eval splits processed separately

Usage:
    python export_to_jsonl.py \
        --input_dataset "./qwen3_asr_training_data" \
        --output_dir "./qwen3_asr_jsonl" \
        --split_ratio 0.003 \
        --num_proc 64
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
from pathlib import Path
from typing import Tuple

from datasets import Dataset, load_dataset
from loguru import logger
from tqdm import tqdm


def save_shard(shard_idx: int, ds: Dataset, num_shards: int, output_dir: str, split_name: str) -> Tuple[int, int]:
    """
    Write a shard of the dataset to a JSONL file.
    
    Returns:
        (shard_idx, num_written)
    """
    shard = ds.shard(num_shards=num_shards, index=shard_idx)
    shard_path = os.path.join(output_dir, f"{split_name}_shard_{shard_idx:04d}.jsonl")
    
    num_written = 0
    with open(shard_path, 'w', encoding='utf-8') as f:
        for sample in shard:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            num_written += 1
    
    return shard_idx, num_written


def write_split_parallel(
    ds: Dataset,
    output_path: Path,
    num_proc: int,
    split_name: str,
    temp_dir: Path
) -> int:
    """
    Write a dataset split to JSONL using parallel processing.
    """
    total_samples = len(ds)
    
    if total_samples == 0:
        output_path.touch()
        return 0
    
    num_shards = min(num_proc, total_samples)
    temp_dir_str = str(temp_dir)
    
    # Write shards in parallel
    logger.info(f"Writing {num_shards} {split_name} shards with {num_proc} processes...")
    
    with mp.Pool(processes=num_proc) as pool:
        args = [(i, ds, num_shards, temp_dir_str, split_name) for i in range(num_shards)]
        results = list(tqdm(
            pool.starmap(save_shard, args),
            total=num_shards,
            desc=f"Writing {split_name} shards"
        ))
    
    total_written = sum(r[1] for r in results)
    
    # Merge shards into final file
    logger.info(f"Merging {num_shards} shards into {output_path}...")
    shard_files = sorted(temp_dir.glob(f"{split_name}_shard_*.jsonl"))
    
    with open(output_path, 'wb') as f_out:
        for shard_file in tqdm(shard_files, desc=f"Merging {split_name}"):
            with open(shard_file, 'rb') as f_in:
                shutil.copyfileobj(f_in, f_out)
            shard_file.unlink()  # Delete shard after merging
    
    return total_written


def main():
    parser = argparse.ArgumentParser(
        description="Export parquet dataset to train.jsonl and eval.jsonl (parallel)"
    )
    parser.add_argument(
        "--input_dataset",
        required=True,
        help="Path to HuggingFace dataset (parquet shards directory)"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for JSONL files"
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.003,
        help="Ratio of data to use for eval split (default: 0.003 = 0.3%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split (default: 42)"
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=64,
        help="Number of parallel processes (default: 64)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("EXPORT PARQUET TO JSONL (PARALLEL)")
    logger.info("=" * 60)
    logger.info(f"Input:       {args.input_dataset}")
    logger.info(f"Output dir:  {args.output_dir}")
    logger.info(f"Split ratio: {args.split_ratio} ({args.split_ratio * 100:.2f}% eval)")
    logger.info(f"Num proc:    {args.num_proc}")
    logger.info(f"Seed:        {args.seed}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temp directory for shards
    temp_dir = output_dir / ".temp_shards"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load dataset
        logger.info("Loading dataset...")
        ds = load_dataset(args.input_dataset, split="train", num_proc=args.num_proc)
        total_samples = len(ds)
        logger.info(f"Loaded {total_samples:,} samples")
        
        # Split into train and eval
        logger.info("Splitting dataset...")
        splits = ds.train_test_split(test_size=args.split_ratio, seed=args.seed)
        train_ds = splits["train"]
        eval_ds = splits["test"]
        
        train_count = len(train_ds)
        eval_count = len(eval_ds)
        logger.info(f"Train: {train_count:,} samples ({100 * train_count / total_samples:.2f}%)")
        logger.info(f"Eval:  {eval_count:,} samples ({100 * eval_count / total_samples:.2f}%)")
        
        # Write train.jsonl in parallel
        train_path = output_dir / "train.jsonl"
        train_written = write_split_parallel(
            train_ds, train_path, args.num_proc, "train", temp_dir
        )
        
        # Write eval.jsonl in parallel
        eval_path = output_dir / "eval.jsonl"
        eval_written = write_split_parallel(
            eval_ds, eval_path, args.num_proc, "eval", temp_dir
        )
        
    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Summary
    logger.info("=" * 60)
    logger.info("EXPORT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Train file: {train_path} ({train_written:,} samples)")
    logger.info(f"Eval file:  {eval_path} ({eval_written:,} samples)")
    
    # Print sample
    logger.info("Sample from train.jsonl:")
    with open(train_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            sample = json.loads(line)
            logger.info(f"  {json.dumps(sample, ensure_ascii=False)[:100]}...")


if __name__ == "__main__":
    main()
