#!/usr/bin/env python3
"""
Verification script for torchaudio and torchcodec installation.
Tests audio format support (wav, mp3, opus, aac) via torchaudio FFmpeg backend.
"""

import sys
import tempfile
import os


def verify_torch_installation():
    """Verify torch, torchaudio, and torchcodec are installed correctly."""
    print("=" * 60)
    print("Verifying PyTorch and torchaudio installation...")
    print("=" * 60)
    
    import torch
    import torchaudio
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"torchaudio version: {torchaudio.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    return True


def verify_torchcodec():
    """Verify torchcodec installation and FFmpeg integration."""
    print("\n" + "=" * 60)
    print("Verifying torchcodec installation...")
    print("=" * 60)
    
    import torchcodec
    print("torchcodec imported successfully")
    
    from torchcodec.decoders import AudioDecoder
    print("AudioDecoder available")
    
    return True


def verify_audio_formats():
    """Test audio format support via torchaudio FFmpeg backend."""
    print("\n" + "=" * 60)
    print("Testing audio format support via torchaudio...")
    print("=" * 60)
    
    import torch
    import torchaudio
    
    # Create a simple test waveform (440 Hz sine wave, 1 second)
    sample_rate = 16000
    duration = 1
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)
    
    # Test formats
    formats = {
        'wav': {'format': 'wav', 'encoding': 'PCM_S', 'bits_per_sample': 16},
        'mp3': {'format': 'mp3'},
        'opus': {'format': 'ogg', 'compression': 10},  # opus in ogg container
    }
    
    results = {}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for fmt_name, kwargs in formats.items():
            filepath = os.path.join(tmpdir, f'test.{fmt_name}')
            try:
                fmt = kwargs.pop('format')
                torchaudio.save(filepath, waveform, sample_rate, format=fmt, **kwargs)
                loaded, sr = torchaudio.load(filepath)
                print(f"  {fmt_name.upper()}: OK (saved and loaded successfully)")
                results[fmt_name] = True
            except Exception as e:
                print(f"  {fmt_name.upper()}: FAILED - {e}")
                results[fmt_name] = False
        
        # Test AAC separately (m4a container)
        try:
            filepath = os.path.join(tmpdir, 'test.m4a')
            torchaudio.save(filepath, waveform, sample_rate, format='mp4')
            loaded, sr = torchaudio.load(filepath)
            print(f"  AAC (m4a): OK (saved and loaded successfully)")
            results['aac'] = True
        except Exception as e:
            print(f"  AAC (m4a): Note - {e}")
            print("  AAC decoding should still work for existing files")
            results['aac'] = False
    
    return results


def main():
    """Run all verification checks."""
    print("\n" + "#" * 60)
    print("# Audio Processing Environment Verification")
    print("#" * 60)
    
    all_passed = True
    
    try:
        verify_torch_installation()
    except Exception as e:
        print(f"ERROR: PyTorch/torchaudio verification failed: {e}")
        all_passed = False
    
    try:
        verify_torchcodec()
    except Exception as e:
        print(f"ERROR: torchcodec verification failed: {e}")
        all_passed = False
    
    try:
        results = verify_audio_formats()
        # wav, mp3, opus are required; aac is optional
        required_formats = ['wav', 'mp3', 'opus']
        for fmt in required_formats:
            if not results.get(fmt, False):
                print(f"ERROR: Required format {fmt} not supported")
                all_passed = False
    except Exception as e:
        print(f"ERROR: Audio format verification failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All verifications PASSED")
        print("=" * 60)
        return 0
    else:
        print("Some verifications FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
