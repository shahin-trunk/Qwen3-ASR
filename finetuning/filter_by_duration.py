#!/usr/bin/env python3
"""Filter dataset by audio duration and save as parquet shards.

Removes audio samples with duration greater than a threshold (default: 30 seconds).

Single machine usage:
    python filter_by_duration.py \
        --input_dataset "./qwen3_asr_training_data" \
        --output_dir "./qwen3_asr_filtered" \
        --max_duration 30.0 \
        --num_proc 64

Distributed usage (12 machines):
    # On machine 0:
    python filter_by_duration.py --input_dataset ... --output_dir ... --machine_id 0 --num_machines 12
    # On machine 1:
    python filter_by_duration.py --input_dataset ... --output_dir ... --machine_id 1 --num_machines 12
    # ... repeat for machines 2-11

    # After all machines complete, merge outputs (outputs are in output_dir/machine_XX/)
"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path
from typing import Tuple

from torchcodec.decoders import AudioDecoder
from datasets import Dataset, load_dataset
from loguru import logger
from tqdm import tqdm


def get_audio_duration(audio_path: str) -> float:
    """
    Get audio duration in seconds using TorchCodec AudioDecoder.
    
    Returns -1.0 if audio cannot be read.
    """
    try:
        decoder = AudioDecoder(audio_path)
        return decoder.metadata.duration_seconds
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
        "--num_proc",
        type=int,
        default=64,
        help="Number of parallel processes (default: 64)"
    )
    parser.add_argument(
        "--machine_id",
        type=int,
        default=None,
        help="Machine ID for distributed processing (0 to num_machines-1)"
    )
    parser.add_argument(
        "--num_machines",
        type=int,
        default=1,
        help="Total number of machines for distributed processing (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Validate distributed args
    distributed = args.machine_id is not None
    if distributed:
        if args.machine_id < 0 or args.machine_id >= args.num_machines:
            logger.error(f"machine_id must be in range [0, {args.num_machines-1}]")
            return
    
    logger.info("=" * 60)
    logger.info("FILTER DATASET BY AUDIO DURATION")
    logger.info("=" * 60)
    logger.info(f"Input:        {args.input_dataset}")
    logger.info(f"Output dir:   {args.output_dir}")
    logger.info(f"Max duration: {args.max_duration} seconds")
    logger.info(f"Num proc:     {args.num_proc}")
    if distributed:
        logger.info(f"Machine:      {args.machine_id + 1}/{args.num_machines}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    if distributed:
        output_dir = output_dir / f"machine_{args.machine_id:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find input parquet files
    input_path = Path(args.input_dataset)
    parquet_files = sorted(input_path.glob("*.parquet"))
    total_input_shards = len(parquet_files)
    
    if total_input_shards == 0:
        logger.error(f"No parquet files found in {args.input_dataset}")
        return
    
    logger.info(f"Found {total_input_shards} input parquet shards")
    
    # Determine which shards this machine processes
    if distributed:
        shards_per_machine = total_input_shards // args.num_machines
        remainder = total_input_shards % args.num_machines
        
        start_shard = args.machine_id * shards_per_machine + min(args.machine_id, remainder)
        if args.machine_id < remainder:
            shards_per_machine += 1
        end_shard = start_shard + shards_per_machine
        
        parquet_files = parquet_files[start_shard:end_shard]
        logger.info(f"Processing shards {start_shard}-{end_shard-1} ({len(parquet_files)} shards)")
    
    # Load only the relevant parquet files
    logger.info("Loading dataset shards...")
    ds = load_dataset(
        "parquet",
        data_files=[str(f) for f in parquet_files],
        split="train",
        num_proc=args.num_proc
    )
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
    
    # Write output shards (1 shard per input shard processed)
    num_output_shards = len(parquet_files)
    output_dir_str = str(output_dir)
    
    if filtered_count == 0:
        logger.warning("No samples passed filtering!")
        total_written = 0
        num_output_shards = 0
    else:
        num_output_shards = min(num_output_shards, filtered_count)
        logger.info(f"Writing {num_output_shards} parquet shards...")
        
        with mp.Pool(processes=args.num_proc) as pool:
            shard_args = [(i, ds_filtered, num_output_shards, output_dir_str) for i in range(num_output_shards)]
            results = list(tqdm(
                pool.starmap(save_shard, shard_args),
                total=num_output_shards,
                desc="Writing shards"
            ))
        
        total_written = sum(r[1] for r in results)
    
    # Summary
    logger.info("=" * 60)
    logger.info("FILTERING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total input:   {total_samples:,}")
    logger.info(f"Total output:  {total_written:,}")
    if total_samples > 0:
        logger.info(f"Removed:       {removed_count:,} ({100*removed_count/total_samples:.2f}%)")
    logger.info(f"Output shards: {num_output_shards} parquet files in {output_dir}")
    
    if distributed:
        logger.info("")
        logger.info("NOTE: This is a distributed run. After all machines complete,")
        logger.info(f"merge outputs from {args.output_dir}/machine_*/")


if __name__ == "__main__":
    main()
