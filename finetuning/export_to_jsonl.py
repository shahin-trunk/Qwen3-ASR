#!/usr/bin/env python3
"""Convert parquet dataset to train.jsonl and eval.jsonl for Qwen3-ASR finetuning.

Usage:
    python export_to_jsonl.py \
        --input_dataset "./qwen3_asr_training_data" \
        --output_dir "./qwen3_asr_jsonl" \
        --split_ratio 0.003
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Export parquet dataset to train.jsonl and eval.jsonl"
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
    logger.info("EXPORT PARQUET TO JSONL")
    logger.info("=" * 60)
    logger.info(f"Input:       {args.input_dataset}")
    logger.info(f"Output dir:  {args.output_dir}")
    logger.info(f"Split ratio: {args.split_ratio} ({args.split_ratio * 100:.2f}% eval)")
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
    
    # Write train.jsonl
    train_path = output_dir / "train.jsonl"
    logger.info(f"Writing {train_path}...")
    with open(train_path, 'w', encoding='utf-8') as f:
        for sample in tqdm(train_ds, desc="Writing train.jsonl"):
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Write eval.jsonl
    eval_path = output_dir / "eval.jsonl"
    logger.info(f"Writing {eval_path}...")
    with open(eval_path, 'w', encoding='utf-8') as f:
        for sample in tqdm(eval_ds, desc="Writing eval.jsonl"):
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Summary
    logger.info("=" * 60)
    logger.info("EXPORT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Train file: {train_path} ({train_count:,} samples)")
    logger.info(f"Eval file:  {eval_path} ({eval_count:,} samples)")
    
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
