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


def convert_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
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
        "--split",
        default="train",
        help="Dataset split to process (default: train)"
    )
    parser.add_argument(
        "--dataset_num_proc",
        type=int,
        default=96,
        help="Number of processes for loading dataset (default: 96)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("QWEN3-ASR DATASET CONVERSION")
    print("=" * 80)
    print(f"Input:       {args.input_dataset}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Num shards:  {args.num_shards}")
    print(f"Num proc:    {args.num_proc}")
    print(f"Split:       {args.split}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...", flush=True)
    ds = load_dataset(args.input_dataset, split=args.split, num_proc=args.dataset_num_proc)
    total_samples = len(ds)
    print(f"Loaded {total_samples:,} samples")
    print()
    
    # Calculate shard sizes
    num_shards = min(args.num_shards, total_samples)
    samples_per_shard = total_samples // num_shards
    remainder = total_samples % num_shards
    
    # Prepare shard info
    shards = []
    current_idx = 0
    for i in range(num_shards):
        shard_size = samples_per_shard + (1 if i < remainder else 0)
        end_idx = current_idx + shard_size
        
        shard_output = output_dir / f"shard_{i:05d}.jsonl"
        shards.append({
            'shard_id': i,
            'start_idx': current_idx,
            'end_idx': end_idx,
            'output_path': str(shard_output)
        })
        current_idx = end_idx
    
    print(f"Processing {num_shards} shards...")
    print()
    
    # Process shards
    num_workers = min(args.num_proc, num_shards)
    
    total_success = 0
    total_processed = 0
    
    # Progress bar for overall progress
    overall_pbar = tqdm(
        total=num_shards,
        desc="Overall progress",
        position=8,
        leave=True,
        ncols=100
    )
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        
        for shard in shards:
            # Extract samples for this shard
            shard_samples = [
                ds[i] for i in range(shard['start_idx'], shard['end_idx'])
            ]
            
            future = executor.submit(
                process_shard,
                shard_samples,
                shard['shard_id'],
                shard['output_path'],
                num_shards
            )
            futures[future] = shard
        
        for future in as_completed(futures):
            shard = futures[future]
            try:
                shard_id, success_count, processed_count = future.result()
                total_success += success_count
                total_processed += processed_count
                overall_pbar.update(1)
            except Exception as e:
                print(f"\nShard {shard['shard_id']} failed: {e}")
                overall_pbar.update(1)
    
    overall_pbar.close()
    
    # Create manifest file listing all shards
    manifest_path = output_dir / "manifest.txt"
    with open(manifest_path, 'w') as f:
        for shard in shards:
            f.write(shard['output_path'] + '\n')
    
    # Summary
    failed_count = total_processed - total_success
    print()
    print("=" * 80)
    print("CONVERSION COMPLETE")
    print("=" * 80)
    print(f"Total processed: {total_processed:,}")
    print(f"Successful:      {total_success:,}")
    if failed_count > 0:
        print(f"Skipped:         {failed_count:,}")
    print(f"Output shards:   {num_shards} files in {args.output_dir}")
    print(f"Manifest:        {manifest_path}")
    print()
    
    # Print sample output
    print("Sample output format:")
    first_shard = output_dir / "shard_00000.jsonl"
    if first_shard.exists():
        with open(first_shard, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                sample = json.loads(line)
                print(f"  {json.dumps(sample, ensure_ascii=False)[:120]}...")


if __name__ == "__main__":
    main()
