#!/usr/bin/env python3
"""Benchmark language detection libraries for ASR dataset processing.

Tests: lingua, langdetect, gcld3, fasttext
Languages: Arabic, Malayalam, Chinese, English, Hindi

Usage:
    python benchmark_langdetect.py
    python benchmark_langdetect.py --iterations 100
"""

import argparse
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# Test data for 5 languages (short and long samples)
TEST_DATA = {
    "Arabic": [
        "مرحبا",
        "مرحبا بكم في العالم",
        "مرحبا بكم في العالم العربي الكبير والواسع",
        "في إمكانكم بالطبع متابعتها ومتابعة كل ما جاء فيها من مواقع التواصل الاجتماعي",
    ],
    "Malayalam": [
        "നമസ്കാരം",
        "നമസ്കാരം എങ്ങനെ ഉണ്ട്",
        "നമസ്കാരം എങ്ങനെ ഉണ്ട് ഞാൻ നല്ലതാണ്",
        "കേരളം ഇന്ത്യയുടെ തെക്കുപടിഞ്ഞാറൻ തീരത്തുള്ള ഒരു സംസ്ഥാനമാണ്",
    ],
    "Chinese": [
        "你好",
        "你好世界欢迎",
        "你好世界欢迎来到中国北京",
        "甚至出现交易几乎停滞的情况这是一个很长的句子",
    ],
    "English": [
        "Hello",
        "Hello world how are you",
        "Hello world how are you doing today",
        "Years after the accident the situation has improved significantly",
    ],
    "Hindi": [
        "नमस्ते",
        "नमस्ते दुनिया कैसे हो",
        "नमस्ते दुनिया कैसे हो आप कहाँ रहते हो",
        "भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली जाती हैं",
    ],
}

# Expected language codes for each library
LANG_CODES = {
    "lingua": {"Arabic": "ARABIC", "Malayalam": None, "Chinese": "CHINESE", "English": "ENGLISH", "Hindi": "HINDI"},
    "langdetect": {"Arabic": "ar", "Malayalam": "ml", "Chinese": "zh-cn", "English": "en", "Hindi": "hi"},
    "gcld3": {"Arabic": "ar", "Malayalam": "ml", "Chinese": "zh", "English": "en", "Hindi": "hi"},
    "fasttext": {"Arabic": "ar", "Malayalam": "ml", "Chinese": "zh", "English": "en", "Hindi": "hi"},
}


@dataclass
class DetectionResult:
    lang_code: str
    confidence: float
    time_ms: float
    success: bool


@dataclass
class LibraryBenchmark:
    name: str
    available: bool
    supports_malayalam: bool
    total_correct: int
    total_tests: int
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    results: Dict[str, List[DetectionResult]]
    error: Optional[str] = None


def init_lingua():
    """Initialize lingua detector."""
    from lingua import Language, LanguageDetectorBuilder
    
    # lingua doesn't support Malayalam
    languages = [
        Language.ARABIC,
        Language.CHINESE,
        Language.ENGLISH,
        Language.HINDI,
    ]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    
    def detect(text: str) -> Tuple[str, float]:
        result = detector.detect_language_of(text)
        if result is None:
            return "unknown", 0.0
        return result.name, 1.0  # lingua doesn't provide confidence
    
    return detect


def init_langdetect():
    """Initialize langdetect."""
    from langdetect import detect, detect_langs
    
    def detect_fn(text: str) -> Tuple[str, float]:
        try:
            lang = detect(text)
            langs = detect_langs(text)
            conf = langs[0].prob if langs else 0.0
            return lang, conf
        except:
            return "unknown", 0.0
    
    return detect_fn


def init_gcld3():
    """Initialize gcld3."""
    import gcld3
    
    detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)
    
    def detect(text: str) -> Tuple[str, float]:
        result = detector.FindLanguage(text=text)
        return result.language, result.probability
    
    return detect


def init_fasttext():
    """Initialize fasttext."""
    import fasttext
    import os
    
    # Download model if not exists
    model_path = "/tmp/lid.176.ftz"
    if not os.path.exists(model_path):
        import urllib.request
        print("Downloading fasttext model...")
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
            model_path
        )
    
    model = fasttext.load_model(model_path)
    
    def detect(text: str) -> Tuple[str, float]:
        text = text.replace("\n", " ")
        predictions = model.predict(text, k=1)
        lang = predictions[0][0].replace("__label__", "")
        conf = float(predictions[1][0])
        return lang, conf
    
    return detect


def benchmark_library(
    name: str,
    init_fn: Callable,
    iterations: int = 10
) -> LibraryBenchmark:
    """Benchmark a single library."""
    
    # Try to initialize
    try:
        detect_fn = init_fn()
    except Exception as e:
        return LibraryBenchmark(
            name=name,
            available=False,
            supports_malayalam=False,
            total_correct=0,
            total_tests=0,
            avg_time_ms=0,
            min_time_ms=0,
            max_time_ms=0,
            results={},
            error=str(e)
        )
    
    expected_codes = LANG_CODES.get(name, {})
    results: Dict[str, List[DetectionResult]] = {}
    all_times = []
    total_correct = 0
    total_tests = 0
    supports_malayalam = expected_codes.get("Malayalam") is not None
    
    for lang_name, texts in TEST_DATA.items():
        results[lang_name] = []
        expected = expected_codes.get(lang_name)
        
        for text in texts:
            # Skip if library doesn't support this language
            if expected is None:
                results[lang_name].append(DetectionResult(
                    lang_code="unsupported",
                    confidence=0.0,
                    time_ms=0.0,
                    success=False
                ))
                continue
            
            # Run multiple iterations for timing
            times = []
            detected_lang = ""
            detected_conf = 0.0
            
            for _ in range(iterations):
                start = time.perf_counter()
                try:
                    detected_lang, detected_conf = detect_fn(text)
                except Exception as e:
                    detected_lang = f"error: {e}"
                    detected_conf = 0.0
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            
            avg_time = sum(times) / len(times)
            all_times.extend(times)
            
            # Check if correct (handle zh-cn vs zh)
            is_correct = (
                detected_lang == expected or
                detected_lang.startswith(expected.split("-")[0]) or
                expected.startswith(detected_lang.split("-")[0])
            )
            
            if is_correct:
                total_correct += 1
            total_tests += 1
            
            results[lang_name].append(DetectionResult(
                lang_code=detected_lang,
                confidence=detected_conf,
                time_ms=avg_time,
                success=is_correct
            ))
    
    return LibraryBenchmark(
        name=name,
        available=True,
        supports_malayalam=supports_malayalam,
        total_correct=total_correct,
        total_tests=total_tests,
        avg_time_ms=sum(all_times) / len(all_times) if all_times else 0,
        min_time_ms=min(all_times) if all_times else 0,
        max_time_ms=max(all_times) if all_times else 0,
        results=results
    )


def print_results(benchmarks: List[LibraryBenchmark]):
    """Print benchmark results."""
    
    print("\n" + "=" * 90)
    print("LANGUAGE DETECTION BENCHMARK RESULTS")
    print("=" * 90)
    
    # Summary table
    print("\n## Summary\n")
    print(f"{'Library':<15} {'Available':<10} {'Malayalam':<10} {'Accuracy':<12} {'Avg Time':<12} {'Min Time':<12}")
    print("-" * 80)
    
    for b in benchmarks:
        if b.available:
            accuracy = f"{b.total_correct}/{b.total_tests} ({100*b.total_correct/b.total_tests:.1f}%)" if b.total_tests > 0 else "N/A"
            print(f"{b.name:<15} {'Yes':<10} {'Yes' if b.supports_malayalam else 'No':<10} {accuracy:<12} {b.avg_time_ms:.3f}ms{'':<5} {b.min_time_ms:.3f}ms")
        else:
            print(f"{b.name:<15} {'No':<10} {'-':<10} {'-':<12} {'-':<12} {'-':<12}")
            print(f"    Error: {b.error}")
    
    # Detailed results per language
    print("\n## Detailed Results by Language\n")
    
    for lang_name in TEST_DATA.keys():
        print(f"\n### {lang_name}\n")
        print(f"{'Library':<15} {'Text Sample':<35} {'Detected':<10} {'Conf':<8} {'Time':<10} {'Status'}")
        print("-" * 90)
        
        for b in benchmarks:
            if not b.available:
                continue
            
            lang_results = b.results.get(lang_name, [])
            for i, (text, result) in enumerate(zip(TEST_DATA[lang_name], lang_results)):
                text_short = text[:30] + "..." if len(text) > 30 else text
                status = "✓" if result.success else ("N/A" if result.lang_code == "unsupported" else "✗")
                conf_str = f"{result.confidence:.2f}" if result.confidence > 0 else "-"
                time_str = f"{result.time_ms:.3f}ms" if result.time_ms > 0 else "-"
                
                if i == 0:
                    print(f"{b.name:<15} {text_short:<35} {result.lang_code:<10} {conf_str:<8} {time_str:<10} {status}")
                else:
                    print(f"{'':<15} {text_short:<35} {result.lang_code:<10} {conf_str:<8} {time_str:<10} {status}")
    
    # Recommendations
    print("\n## Recommendations\n")
    
    available_with_malayalam = [b for b in benchmarks if b.available and b.supports_malayalam]
    if available_with_malayalam:
        fastest = min(available_with_malayalam, key=lambda x: x.avg_time_ms)
        most_accurate = max(available_with_malayalam, key=lambda x: x.total_correct / x.total_tests if x.total_tests > 0 else 0)
        
        print(f"- **Fastest with Malayalam support**: {fastest.name} ({fastest.avg_time_ms:.3f}ms avg)")
        print(f"- **Most accurate with Malayalam**: {most_accurate.name} ({100*most_accurate.total_correct/most_accurate.total_tests:.1f}% accuracy)")
        
        if fastest.name == most_accurate.name:
            print(f"\n**Recommendation**: Use **{fastest.name}** - it's both fastest and most accurate!")
        else:
            print(f"\n**Recommendation**: Use **{fastest.name}** for speed, or **{most_accurate.name}** for accuracy.")
    else:
        print("No library with Malayalam support is available.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark language detection libraries")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations per test (default: 10)")
    args = parser.parse_args()
    
    print("=" * 90)
    print("LANGUAGE DETECTION LIBRARY BENCHMARK")
    print("=" * 90)
    print(f"Languages: Arabic, Malayalam, Chinese, English, Hindi")
    print(f"Iterations per test: {args.iterations}")
    print()
    
    libraries = [
        ("lingua", init_lingua),
        ("langdetect", init_langdetect),
        ("gcld3", init_gcld3),
        ("fasttext", init_fasttext),
    ]
    
    benchmarks = []
    for name, init_fn in libraries:
        print(f"Benchmarking {name}...", end=" ", flush=True)
        result = benchmark_library(name, init_fn, iterations=args.iterations)
        if result.available:
            print(f"done ({result.avg_time_ms:.3f}ms avg)")
        else:
            print(f"FAILED: {result.error[:50]}...")
        benchmarks.append(result)
    
    print_results(benchmarks)


if __name__ == "__main__":
    main()
