#!/usr/bin/env python3
"""Convert parquet dataset to train.jsonl and eval.jsonl for Qwen3-ASR finetuning.

Features:
- Parallel writing: each process writes to a separate shard, then merge
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
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from datasets import Dataset, load_dataset
from loguru import logger
from tqdm import tqdm


def write_shard(args: Tuple[int, int, int, str, str]) -> Tuple[int, int, str]:
    """
    Write a shard of the dataset to a JSONL file.
    
    Args:
        args: (shard_id, start_idx, end_idx, dataset_path, output_path)
    
    Returns:
        (shard_id, num_written, output_path)
    """
    shard_id, start_idx, end_idx, dataset_path, output_path = args
    
    # Load the dataset in this process
    ds = load_dataset(dataset_path, split="train")
    
    # Select the shard range
    shard_ds = ds.select(range(start_idx, end_idx))
    
    # Write to JSONL
    num_written = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in shard_ds:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            num_written += 1
    
    return shard_id, num_written, output_path


def write_split_parallel(
    ds: Dataset,
    output_path: Path,
    num_proc: int,
    split_name: str,
    temp_dir: str
) -> int:
    """
    Write a dataset split to JSONL using parallel processing.
    
    Each process writes to a separate shard file, then all shards are merged.
    """
    total_samples = len(ds)
    
    if total_samples == 0:
        # Create empty file
        output_path.touch()
        return 0
    
    # Save dataset temporarily for workers to load
    temp_dataset_path = os.path.join(temp_dir, f"{split_name}_dataset")
    ds.save_to_disk(temp_dataset_path)
    
    # Calculate shard boundaries
    num_shards = min(num_proc, total_samples)
    samples_per_shard = total_samples // num_shards
    remainder = total_samples % num_shards
    
    shard_args = []
    current_idx = 0
    for shard_id in range(num_shards):
        shard_size = samples_per_shard + (1 if shard_id < remainder else 0)
        end_idx = current_idx + shard_size
        shard_output = os.path.join(temp_dir, f"{split_name}_shard_{shard_id:04d}.jsonl")
        shard_args.append((shard_id, current_idx, end_idx, temp_dataset_path, shard_output))
        current_idx = end_idx
    
    # Process shards in parallel
    shard_files = []
    total_written = 0
    
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        futures = {executor.submit(write_shard, args): args[0] for args in shard_args}
        
        with tqdm(total=num_shards, desc=f"Writing {split_name} shards") as pbar:
            for future in as_completed(futures):
                shard_id, num_written, shard_path = future.result()
                shard_files.append((shard_id, shard_path))
                total_written += num_written
                pbar.update(1)
    
    # Sort shards by ID to maintain order
    shard_files.sort(key=lambda x: x[0])
    
    # Merge shards into final file
    logger.info(f"Merging {len(shard_files)} shards into {output_path}...")
    with open(output_path, 'wb') as f_out:
        for _, shard_path in tqdm(shard_files, desc=f"Merging {split_name}"):
            with open(shard_path, 'rb') as f_in:
                shutil.copyfileobj(f_in, f_out)
    
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
    
    # Create temp directory for shards
    with tempfile.TemporaryDirectory(prefix="export_jsonl_") as temp_dir:
        # Write train.jsonl in parallel
        train_path = output_dir / "train.jsonl"
        logger.info(f"Writing {train_path} with {args.num_proc} processes...")
        train_written = write_split_parallel(
            train_ds, train_path, args.num_proc, "train", temp_dir
        )
        
        # Write eval.jsonl in parallel
        eval_path = output_dir / "eval.jsonl"
        logger.info(f"Writing {eval_path} with {args.num_proc} processes...")
        eval_written = write_split_parallel(
            eval_ds, eval_path, args.num_proc, "eval", temp_dir
        )
    
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
