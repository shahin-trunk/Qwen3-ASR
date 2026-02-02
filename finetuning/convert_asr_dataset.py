#!/usr/bin/env python3
"""Convert ASR dataset to Qwen3-ASR training format with language detection.

Input format (HuggingFace dataset):
    {
        "messages": [
            {"content": "<audio>", "role": "user"},
            {"content": "transcription text", "role": "assistant"}
        ],
        "audios": ["path/to/audio.wav"]
    }

Output format (JSONL):
    {"audio": "/path/to/audio.wav", "text": "language English<asr_text>transcription text"}

Features:
- Uses langdetect for automatic language detection (supports Malayalam)
- Supports 256 output shards for distributed training
- Efficient batch processing using datasets library

Usage:
    python convert_asr_dataset.py \
        --input_dataset "data/asr/.dset/dpo/enriched/qasr/sft" \
        --output_dir "./qwen3_asr_training_data" \
        --num_shards 256 \
        --num_proc 64
"""

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import Optional, Dict, Any

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

# Language code mapping from langdetect to Qwen3-ASR format
# Limited to: Arabic, Malayalam, Chinese, English, Hindi
LANGDETECT_TO_QWEN = {
    "ar": "Arabic",
    "ml": "Malayalam",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "en": "English",
    "hi": "Hindi",
}

# Only these languages are supported in this dataset
SUPPORTED_LANGUAGES = {"Arabic", "Malayalam", "Chinese", "English", "Hindi"}


def detect_language(text: str) -> str:
    """
    Detect language from text using langdetect.
    
    Returns Qwen3-ASR language name or "None" if detection fails or unsupported.
    """
    if not text or not text.strip():
        return "None"
    
    try:
        from langdetect import detect
        lang_code = detect(text)
        
        # Map to Qwen3-ASR language name
        qwen_lang = LANGDETECT_TO_QWEN.get(lang_code, "None")
        
        return qwen_lang
    except Exception:
        return "None"


def convert_sample(sample: Dict[str, Any], audio_base_path: str = None) -> Dict[str, Any]:
    """
    Convert a single sample from HuggingFace format to Qwen3-ASR format.
    
    Input:
        {
            "messages": [{"content": "<audio>", "role": "user"}, {"content": "text", "role": "assistant"}],
            "audios": ["path/to/audio.wav"]
        }
    
    Output:
        {"audio": "path/to/audio.wav", "text": "language English<asr_text>text", "valid": True}
    """
    messages = sample.get("messages", [])
    audios = sample.get("audios", [])
    
    if not messages or not audios:
        return {"audio": "", "text": "", "valid": False}
    
    # Extract transcription from assistant message
    transcription = None
    for msg in messages:
        if msg.get("role") == "assistant":
            transcription = msg.get("content", "").strip()
            break
    
    if not transcription:
        return {"audio": "", "text": "", "valid": False}
    
    # Get audio path
    audio_path = audios[0] if audios else None
    if not audio_path:
        return {"audio": "", "text": "", "valid": False}
    
    # Join with base path if provided
    if audio_base_path:
        audio_path = os.path.join(audio_base_path, audio_path)
    
    # Detect language
    language = detect_language(transcription)
    
    # Format output text
    text = f"language {language}<asr_text>{transcription}"
    
    return {
        "audio": audio_path,
        "text": text,
        "valid": True
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert ASR dataset to Qwen3-ASR training format"
    )
    parser.add_argument(
        "--input_dataset", 
        required=True, 
        help="Path to HuggingFace dataset"
    )
    parser.add_argument(
        "--output_dir", 
        required=True, 
        help="Output directory for sharded JSONL files"
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
    parser.add_argument(
        "--audio_base_path",
        type=str,
        default=None,
        help="Base path to prepend to audio paths (default: None)"
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to process (default: train)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("QWEN3-ASR DATASET CONVERSION")
    logger.info("=" * 60)
    logger.info(f"Input:       {args.input_dataset}")
    logger.info(f"Output dir:  {args.output_dir}")
    logger.info(f"Num shards:  {args.num_shards}")
    logger.info(f"Num proc:    {args.num_proc}")
    logger.info(f"Audio base:  {args.audio_base_path or '(none)'}")
    logger.info(f"Split:       {args.split}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info("Loading dataset...")
    ds = load_dataset(args.input_dataset, split=args.split, num_proc=args.num_proc)
    total_samples = len(ds)
    logger.info(f"Loaded {total_samples:,} samples")
    
    # Validate audio paths with random 100 samples
    logger.info("Validating audio paths (random 100 samples)...")
    import random
    sample_indices = random.sample(range(total_samples), min(100, total_samples))
    missing_count = 0
    checked_paths = []
    
    for idx in sample_indices:
        sample = ds[idx]
        audios = sample.get("audios", [])
        if audios:
            audio_path = audios[0]
            if args.audio_base_path:
                audio_path = os.path.join(args.audio_base_path, audio_path)
            
            if not os.path.exists(audio_path):
                missing_count += 1
                if len(checked_paths) < 5:  # Show first 5 missing
                    checked_paths.append(audio_path)
    
    if missing_count > 0:
        logger.warning(f"{missing_count}/100 sampled audio files not found!")
        logger.warning("Example missing paths:")
        for p in checked_paths:
            logger.warning(f"  {p}")
        response = input("Continue anyway? [y/N]: ").strip().lower()
        if response != 'y':
            logger.info("Aborted.")
            sys.exit(1)
    else:
        logger.info("All 100 sampled audio files exist. Proceeding...")

    # Convert samples using parallel map
    logger.info("Converting samples (detecting languages)...")
    convert_fn = partial(convert_sample, audio_base_path=args.audio_base_path)
    ds_converted = ds.map(
        convert_fn,
        num_proc=args.num_proc,
        desc="Converting",
        remove_columns=ds.column_names,  # Remove original columns
    )
    
    # Filter valid samples
    logger.info("Filtering valid samples...")
    ds_valid = ds_converted.filter(
        lambda x: x["valid"],
        num_proc=args.num_proc,
        desc="Filtering"
    )
    
    valid_count = len(ds_valid)
    logger.info(f"Valid samples: {valid_count:,} / {total_samples:,}")
    
    # Remove the 'valid' column before saving
    ds_valid = ds_valid.remove_columns(["valid"])
    
    # Calculate shard sizes
    num_shards = min(args.num_shards, valid_count)
    
    logger.info("Shuffling dataset...")
    ds_valid = ds_valid.shuffle(seed=42)
    
    logger.info(f"Writing {num_shards} parquet shards...")
    
    for shard_idx in tqdm(range(num_shards), desc="Writing shards"):
        shard = ds_valid.shard(num_shards=num_shards, index=shard_idx)
        shard_path = output_dir / f"{shard_idx:05d}.parquet"
        shard.to_parquet(str(shard_path))
    
    # Language statistics (parallel)
    logger.info("Computing language statistics...")
    
    def extract_language(sample):
        text = sample["text"]
        if text.startswith("language "):
            lang = text.split("<asr_text>")[0].replace("language ", "").strip()
        else:
            lang = "Unknown"
        return {"lang": lang}
    
    lang_ds = ds_valid.map(
        extract_language,
        num_proc=args.num_proc,
        desc="Extracting languages"
    )
    
    # Count using pandas (fast)
    from collections import Counter
    lang_counts = Counter(lang_ds["lang"])
    
    logger.info("=" * 60)
    logger.info("CONVERSION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total input:     {total_samples:,}")
    logger.info(f"Valid output:    {valid_count:,}")
    logger.info(f"Skipped:         {total_samples - valid_count:,}")
    logger.info(f"Output shards:   {num_shards} parquet files in {args.output_dir}")
    logger.info(f"Samples/shard:   ~{valid_count // num_shards:,}")
    
    # Print language distribution
    logger.info("Language distribution:")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / valid_count
        logger.info(f"  {lang:15s}: {count:>10,} ({pct:5.2f}%)")
    
    # Save undetected (None) language texts to file
    none_count = lang_counts.get("None", 0)
    if none_count > 0:
        none_file = output_dir / "undetected_languages.jsonl"
        logger.info(f"Saving {none_count:,} undetected language samples to {none_file}...")
        
        # Filter samples with "None" language
        none_samples = ds_valid.filter(
            lambda x: x["text"].startswith("language None<asr_text>"),
            num_proc=args.num_proc,
            desc="Filtering undetected"
        )
        
        with open(none_file, 'w', encoding='utf-8') as f:
            for sample in none_samples:
                # Extract just the text part for analysis
                text = sample["text"].replace("language None<asr_text>", "")
                f.write(json.dumps({"audio": sample["audio"], "text": text}, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {none_count:,} undetected samples to {none_file}")
    
    # Print sample output
    logger.info("Sample output format:")
    first_shard = output_dir / "00000.parquet"
    if first_shard.exists():
        import pyarrow.parquet as pq
        table = pq.read_table(str(first_shard))
        df = table.to_pandas()
        for i in range(min(3, len(df))):
            sample = df.iloc[i].to_dict()
            logger.info(f"  {json.dumps(sample, ensure_ascii=False)[:120]}...")


if __name__ == "__main__":
    main()
