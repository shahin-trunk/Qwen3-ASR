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
- Uses lingua-language-detector for automatic language detection
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
from tqdm import tqdm

# Global detector (initialized once per process)
_DETECTOR = None

# Language mapping from lingua to Qwen3-ASR format
LINGUA_TO_QWEN_LANG = {
    "ARABIC": "Arabic",
    "CHINESE": "Chinese",
    "ENGLISH": "English",
    "FRENCH": "French",
    "GERMAN": "German",
    "SPANISH": "Spanish",
    "PORTUGUESE": "Portuguese",
    "INDONESIAN": "Indonesian",
    "ITALIAN": "Italian",
    "KOREAN": "Korean",
    "RUSSIAN": "Russian",
    "THAI": "Thai",
    "VIETNAMESE": "Vietnamese",
    "JAPANESE": "Japanese",
    "TURKISH": "Turkish",
    "HINDI": "Hindi",
    "MALAY": "Malay",
    "DUTCH": "Dutch",
    "SWEDISH": "Swedish",
    "DANISH": "Danish",
    "FINNISH": "Finnish",
    "POLISH": "Polish",
    "CZECH": "Czech",
    "TAGALOG": "Filipino",  # lingua uses TAGALOG
    "PERSIAN": "Persian",
    "GREEK": "Greek",
    "ROMANIAN": "Romanian",
    "HUNGARIAN": "Hungarian",
    "MACEDONIAN": "Macedonian",
    # Cantonese is not directly supported by lingua, will fall back to Chinese
}

# Qwen3-ASR supported languages for validation
SUPPORTED_LANGUAGES = {
    "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish",
    "Portuguese", "Indonesian", "Italian", "Korean", "Russian", "Thai",
    "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch", "Swedish",
    "Danish", "Finnish", "Polish", "Czech", "Filipino", "Persian", "Greek",
    "Romanian", "Hungarian", "Macedonian"
}


def get_language_detector():
    """Get or initialize lingua language detector (cached per process)."""
    global _DETECTOR
    if _DETECTOR is None:
        try:
            from lingua import Language, LanguageDetectorBuilder
            
            # Build detector with all supported languages
            languages = [
                Language.ARABIC, Language.CHINESE, Language.ENGLISH, Language.FRENCH,
                Language.GERMAN, Language.SPANISH, Language.PORTUGUESE, Language.INDONESIAN,
                Language.ITALIAN, Language.KOREAN, Language.RUSSIAN, Language.THAI,
                Language.VIETNAMESE, Language.JAPANESE, Language.TURKISH, Language.HINDI,
                Language.MALAY, Language.DUTCH, Language.SWEDISH, Language.DANISH,
                Language.FINNISH, Language.POLISH, Language.CZECH, Language.TAGALOG,
                Language.PERSIAN, Language.GREEK, Language.ROMANIAN, Language.HUNGARIAN,
                Language.MACEDONIAN,
            ]
            
            _DETECTOR = LanguageDetectorBuilder.from_languages(*languages).build()
        except ImportError:
            print("ERROR: lingua-language-detector not installed.")
            print("Install with: pip install lingua-language-detector")
            sys.exit(1)
    return _DETECTOR


def detect_language(text: str) -> str:
    """
    Detect language from text using lingua.
    
    Returns Qwen3-ASR language name or "None" if detection fails.
    """
    if not text or not text.strip():
        return "None"
    
    try:
        detector = get_language_detector()
        result = detector.detect_language_of(text)
        if result is None:
            return "None"
        
        lang_name = result.name  # e.g., "ARABIC", "ENGLISH"
        qwen_lang = LINGUA_TO_QWEN_LANG.get(lang_name, "None")
        
        # Validate against supported languages
        if qwen_lang not in SUPPORTED_LANGUAGES:
            return "None"
        
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
    
    print("=" * 80)
    print("QWEN3-ASR DATASET CONVERSION")
    print("=" * 80)
    print(f"Input:       {args.input_dataset}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Num shards:  {args.num_shards}")
    print(f"Num proc:    {args.num_proc}")
    print(f"Audio base:  {args.audio_base_path or '(none)'}")
    print(f"Split:       {args.split}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...", flush=True)
    ds = load_dataset(args.input_dataset, split=args.split, num_proc=args.num_proc)
    total_samples = len(ds)
    print(f"Loaded {total_samples:,} samples")
    print()
    
    # Validate audio paths with random 100 samples
    print("Validating audio paths (random 100 samples)...", flush=True)
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
        print(f"WARNING: {missing_count}/100 sampled audio files not found!")
        print("Example missing paths:")
        for p in checked_paths:
            print(f"  {p}")
        print()
        response = input("Continue anyway? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            sys.exit(1)
    else:
        print(f"All 100 sampled audio files exist. Proceeding...")
    print()

    # Convert samples using parallel map
    print("Converting samples (detecting languages)...", flush=True)
    convert_fn = partial(convert_sample, audio_base_path=args.audio_base_path)
    ds_converted = ds.map(
        convert_fn,
        num_proc=args.num_proc,
        desc="Converting",
        remove_columns=ds.column_names,  # Remove original columns
    )
    
    # Filter valid samples
    print("Filtering valid samples...", flush=True)
    ds_valid = ds_converted.filter(
        lambda x: x["valid"],
        num_proc=args.num_proc,
        desc="Filtering"
    )
    
    valid_count = len(ds_valid)
    print(f"Valid samples: {valid_count:,} / {total_samples:,}")
    print()
    
    # Remove the 'valid' column before saving
    ds_valid = ds_valid.remove_columns(["valid"])
    
    # Calculate shard sizes
    num_shards = min(args.num_shards, valid_count)
    
    print(f"Shuffling dataset...")
    ds_valid = ds_valid.shuffle(seed=42)
    
    print(f"Writing {num_shards} parquet shards...")
    
    for shard_idx in tqdm(range(num_shards), desc="Writing shards"):
        shard = ds_valid.shard(num_shards=num_shards, index=shard_idx)
        shard_path = output_dir / f"{shard_idx:05d}.parquet"
        shard.to_parquet(str(shard_path))
    
    # Language statistics
    print()
    print("Computing language statistics...")
    lang_counts = {}
    for sample in tqdm(ds_valid, desc="Counting languages"):
        text = sample["text"]
        # Extract language from "language X<asr_text>..."
        if text.startswith("language "):
            lang = text.split("<asr_text>")[0].replace("language ", "").strip()
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
    
    print()
    print("=" * 80)
    print("CONVERSION COMPLETE")
    print("=" * 80)
    print(f"Total input:     {total_samples:,}")
    print(f"Valid output:    {valid_count:,}")
    print(f"Skipped:         {total_samples - valid_count:,}")
    print(f"Output shards:   {num_shards} parquet files in {args.output_dir}")
    print(f"Samples/shard:   ~{valid_count // num_shards:,}")
    print()
    
    # Print language distribution
    print("Language distribution:")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / valid_count
        print(f"  {lang:15s}: {count:>10,} ({pct:5.2f}%)")
    print()
    
    # Print sample output
    print("Sample output format:")
    first_shard = output_dir / "00000.parquet"
    if first_shard.exists():
        import pyarrow.parquet as pq
        table = pq.read_table(str(first_shard))
        df = table.to_pandas()
        for i in range(min(3, len(df))):
            sample = df.iloc[i].to_dict()
            print(f"  {json.dumps(sample, ensure_ascii=False)[:120]}...")


if __name__ == "__main__":
    main()
