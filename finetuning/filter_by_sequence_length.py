#!/usr/bin/env python3
"""Filter JSONL dataset by actual sequence length using the real DataCollator.

This is the most accurate filtering method as it:
1. Loads the actual audio file
2. Processes it through the feature extractor (mel spectrogram → audio tokens)
3. Applies the chat template (prefix tokens)
4. Tokenizes the full transcript
5. Computes the exact input_ids length that would be used during training

Usage:
    python filter_by_sequence_length.py \
        --input_file "train.jsonl" \
        --output_file "train_filtered.jsonl" \
        --model_path "Qwen/Qwen3-ASR-1.7B" \
        --max_seq_len 2048 \
        --num_proc 32
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
from loguru import logger
from tqdm import tqdm


def load_audio(path: str, sr: int = 16000):
    """Load audio file and return waveform."""
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array):
    """Build chat messages for prefix."""
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def compute_sequence_length(
    sample: Dict[str, Any],
    processor: Any,
    sampling_rate: int = 16000
) -> int:
    """
    Compute the actual sequence length for a sample using the processor.
    
    This replicates the DataCollator logic to get the exact input_ids length.
    """
    audio_path = sample.get("audio", "")
    text = sample.get("text", "")
    prompt = sample.get("prompt", "")
    
    # Load audio
    audio = load_audio(audio_path, sr=sampling_rate)
    
    # Build prefix text (chat template)
    prefix_msgs = build_prefix_messages(prompt, None)
    prefix_text = processor.apply_chat_template(
        [prefix_msgs], add_generation_prompt=True, tokenize=False
    )[0]
    
    # Build full text
    eos = processor.tokenizer.eos_token or ""
    full_text = prefix_text + text + eos
    
    # Process through processor to get actual input_ids length
    inputs = processor(
        text=[full_text],
        audio=[audio],
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
    
    seq_len = inputs["input_ids"].shape[1]
    return seq_len


def process_shard(
    shard_idx: int,
    input_file: str,
    start_line: int,
    end_line: int,
    output_dir: str,
    model_path: str,
    max_seq_len: int,
    sampling_rate: int,
    save_rejected: bool
) -> Tuple[int, int, int, List[Dict]]:
    """
    Process a shard of the input file.
    
    Returns:
        (shard_idx, kept, rejected, rejected_samples)
    """
    # Load processor in each worker
    from qwen_asr import Qwen3ASRModel
    import torch
    
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map=None,
    )
    processor = asr_wrapper.processor
    
    shard_path = f"{output_dir}/shard_{shard_idx:04d}.jsonl"
    
    kept = 0
    rejected = 0
    rejected_samples = []
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(shard_path, 'w', encoding='utf-8') as f_out:
        
        for i, line in enumerate(f_in):
            if i < start_line:
                continue
            if i >= end_line:
                break
            
            sample = json.loads(line)
            
            try:
                seq_len = compute_sequence_length(sample, processor, sampling_rate)
                
                if seq_len <= max_seq_len:
                    f_out.write(line)
                    kept += 1
                else:
                    rejected += 1
                    if save_rejected:
                        # Extract transcript for preview
                        text = sample.get("text", "")
                        match = re.search(r"<asr_text>(.*)$", text)
                        transcript = match.group(1) if match else text
                        
                        rejected_samples.append({
                            "audio": sample.get("audio", ""),
                            "seq_len": seq_len,
                            "transcript_len": len(transcript),
                            "transcript_preview": transcript[:200] + "..." if len(transcript) > 200 else transcript
                        })
            except Exception as e:
                # Skip samples that fail to process
                rejected += 1
                if save_rejected:
                    rejected_samples.append({
                        "audio": sample.get("audio", ""),
                        "seq_len": -1,
                        "error": str(e)[:100],
                        "transcript_preview": ""
                    })
    
    return shard_idx, kept, rejected, rejected_samples


def main():
    parser = argparse.ArgumentParser(
        description="Filter JSONL by actual sequence length (using DataCollator)"
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
        "--model_path",
        type=str,
        default="Qwen/Qwen3-ASR-1.7B",
        help="Model path for processor (default: Qwen/Qwen3-ASR-1.7B)"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=16000,
        help="Audio sampling rate (default: 16000)"
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
        default=32,
        help="Number of parallel processes (default: 32)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("FILTER BY SEQUENCE LENGTH (DataCollator)")
    logger.info("=" * 60)
    logger.info(f"Input:       {args.input_file}")
    logger.info(f"Output:      {args.output_file}")
    logger.info(f"Model:       {args.model_path}")
    logger.info(f"Max seq len: {args.max_seq_len}")
    logger.info(f"Num proc:    {args.num_proc}")
    
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temp directory for shards
    temp_dir = output_path.parent / ".temp_seqlen_shards"
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
        logger.info("(Each worker loads the model processor - this may take a moment)")
        
        with mp.Pool(processes=args.num_proc) as pool:
            process_args = [
                (idx, str(input_path), start, end, str(temp_dir),
                 args.model_path, args.max_seq_len, args.sampling_rate,
                 args.rejected_file is not None)
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
        all_rejected_samples = []
        for r in results:
            all_rejected_samples.extend(r[3])
        
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
            # Sort by seq_len descending
            all_rejected_samples.sort(key=lambda x: x.get("seq_len", 0), reverse=True)
            
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
    logger.info(f"Output file:   {args.output_file}")
    
    # Show distribution of rejected seq_lens
    if all_rejected_samples:
        seq_lens = [s.get("seq_len", 0) for s in all_rejected_samples if s.get("seq_len", 0) > 0]
        if seq_lens:
            logger.info(f"Rejected seq_len range: {min(seq_lens):,} - {max(seq_lens):,}")


if __name__ == "__main__":
    main()
