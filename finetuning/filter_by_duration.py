#!/usr/bin/env python3
"""Filter dataset by audio duration and save as parquet shards.

Removes audio samples with duration greater than a threshold (default: 30 seconds).

Usage:
    python filter_by_duration.py \
        --input_dataset "./qwen3_asr_training_data" \
        --output_dir "./qwen3_asr_filtered" \
        --max_duration 30.0 \
        --num_shards 256 \
        --num_proc 64
"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path
from typing import Tuple

import torchaudio
from datasets import Dataset, load_dataset
from loguru import logger
from tqdm import tqdm


def get_audio_duration(audio_path: str) -> float:
    """
    Get audio duration in seconds using torchaudio.
    
    Returns -1.0 if audio cannot be read.
    """
    try:
        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate
    except Exception:
        return -1.0


def check_duration(sample: dict, max_duration: float) -> dict:
    """
    Check if audio duration is within threshold.
    
    Returns sample with 'keep' flag.
    """
    audio_path = sample.get("audio", "")
    
    if not audio_path or not os.path.exists(audio_path):
        return {**sample, "duration": -1.0, "keep": False}
    
    duration = get_audio_duration(audio_path)
    
    keep = 0.0 < duration <= max_duration
    
    return {**sample, "duration": duration, "keep": keep}


def save_shard(shard_idx: int, ds: Dataset, num_shards: int, output_dir: str) -> Tuple[int, int]:
    """
    Write a shard of the dataset to parquet.
    
    Returns:
        (shard_idx, num_written)
    """
    shard = ds.shard(num_shards=num_shards, index=shard_idx)
    shard_path = os.path.join(output_dir, f"{shard_idx:05d}.parquet")
    
    shard.to_parquet(shard_path)
    
    return shard_idx, len(shard)


def main():
    parser = argparse.ArgumentParser(
        description="Filter dataset by audio duration"
    )
    parser.add_argument(
        "--input_dataset",
        required=True,
        help="Path to HuggingFace dataset (parquet shards directory)"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for filtered parquet shards"
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds (default: 30.0)"
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=256,
        help="Number of output shards (default: 256)"
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=64,
        help="Number of parallel processes (default: 64)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("FILTER DATASET BY AUDIO DURATION")
    logger.info("=" * 60)
    logger.info(f"Input:        {args.input_dataset}")
    logger.info(f"Output dir:   {args.output_dir}")
    logger.info(f"Max duration: {args.max_duration} seconds")
    logger.info(f"Num shards:   {args.num_shards}")
    logger.info(f"Num proc:     {args.num_proc}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info("Loading dataset...")
    ds = load_dataset(args.input_dataset, split="train", num_proc=args.num_proc)
    total_samples = len(ds)
    logger.info(f"Loaded {total_samples:,} samples")
    
    # Check durations and filter (parallel)
    logger.info(f"Checking audio durations (filtering > {args.max_duration}s)...")
    
    from functools import partial
    check_fn = partial(check_duration, max_duration=args.max_duration)
    
    ds_checked = ds.map(
        check_fn,
        num_proc=args.num_proc,
        desc="Checking durations"
    )
    
    # Filter valid samples
    ds_filtered = ds_checked.filter(
        lambda x: x["keep"],
        num_proc=args.num_proc,
        desc="Filtering"
    )
    
    filtered_count = len(ds_filtered)
    removed_count = total_samples - filtered_count
    
    logger.info(f"Kept:    {filtered_count:,} samples")
    logger.info(f"Removed: {removed_count:,} samples (>{args.max_duration}s or invalid)")
    
    # Remove helper columns
    ds_filtered = ds_filtered.remove_columns(["duration", "keep"])
    
    # Shuffle before sharding
    logger.info("Shuffling dataset...")
    ds_filtered = ds_filtered.shuffle(seed=42)
    
    # Write parquet shards in parallel
    num_shards = min(args.num_shards, filtered_count)
    output_dir_str = str(output_dir)
    
    logger.info(f"Writing {num_shards} parquet shards with {args.num_proc} processes...")
    
    with mp.Pool(processes=args.num_proc) as pool:
        shard_args = [(i, ds_filtered, num_shards, output_dir_str) for i in range(num_shards)]
        results = list(tqdm(
            pool.starmap(save_shard, shard_args),
            total=num_shards,
            desc="Writing shards"
        ))
    
    total_written = sum(r[1] for r in results)
    
    # Summary
    logger.info("=" * 60)
    logger.info("FILTERING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total input:   {total_samples:,}")
    logger.info(f"Total output:  {total_written:,}")
    logger.info(f"Removed:       {removed_count:,} ({100*removed_count/total_samples:.2f}%)")
    logger.info(f"Output shards: {num_shards} parquet files in {args.output_dir}")
    logger.info(f"Samples/shard: ~{filtered_count // num_shards:,}")


if __name__ == "__main__":
    main()
