#!/usr/bin/env python
"""Generate a TTS sample, transcribe it, and score basic smoke metrics.

This is a debugging harness for Voxtral TTS bring-up on Apple Silicon.
It is intentionally heuristic: ASR + repetition/duration metrics are used
as a smoke signal, not as a perfect proxy for human listening quality.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from mlx_audio.audio_io import read as audio_read
from mlx_audio.audio_io import write as audio_write
from mlx_audio.stt import load as load_stt
from mlx_audio.tts import load as load_tts


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def jaccard_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    a_set = set(a)
    b_set = set(b)
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def longest_same_word_run(tokens: list[str]) -> int:
    if not tokens:
        return 0
    best = 1
    current = 1
    for prev, cur in zip(tokens, tokens[1:]):
        if cur == prev:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def repeated_bigram_fraction(tokens: list[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    counts = Counter(bigrams)
    most_common = counts.most_common(1)[0][1]
    return most_common / len(bigrams)


@dataclass
class SmokeMetrics:
    prompt_text: str
    transcript_text: str
    duration_seconds: float
    processing_seconds: float
    sample_rate: int
    samples: int
    prompt_word_count: int
    transcript_word_count: int
    sequence_ratio: float
    word_jaccard: float
    common_word_fraction: float
    repeated_bigram_fraction: float
    longest_same_word_run: int
    transcript_words_per_second: float
    looks_suspicious: bool
    suspicious_reasons: list[str]


def score_metrics(prompt_text: str, transcript_text: str, duration_seconds: float, processing_seconds: float, sample_rate: int, samples: int) -> SmokeMetrics:
    prompt_norm = normalize_text(prompt_text)
    transcript_norm = normalize_text(transcript_text)
    prompt_tokens = tokenize(prompt_text)
    transcript_tokens = tokenize(transcript_text)

    sequence_ratio = SequenceMatcher(None, prompt_norm, transcript_norm).ratio()
    word_jaccard = jaccard_overlap(prompt_tokens, transcript_tokens)

    common_word_fraction = 0.0
    if transcript_tokens:
        common_word_fraction = Counter(transcript_tokens).most_common(1)[0][1] / len(transcript_tokens)

    longest_run = longest_same_word_run(transcript_tokens)
    bigram_fraction = repeated_bigram_fraction(transcript_tokens)
    transcript_wps = len(transcript_tokens) / duration_seconds if duration_seconds > 0 else 0.0

    reasons: list[str] = []
    if not transcript_tokens:
        reasons.append("empty_transcript")
    if sequence_ratio < 0.18:
        reasons.append("low_sequence_similarity")
    if word_jaccard < 0.12:
        reasons.append("low_word_overlap")
    if common_word_fraction > 0.35 and len(transcript_tokens) >= 8:
        reasons.append("dominant_repeated_word")
    if bigram_fraction > 0.20 and len(transcript_tokens) >= 8:
        reasons.append("dominant_repeated_bigram")
    if longest_run >= 4:
        reasons.append("long_same_word_run")
    if transcript_wps < 0.5 and len(transcript_tokens) >= 4:
        reasons.append("very_low_transcript_density")
    if duration_seconds > 25 and len(prompt_tokens) <= 25:
        reasons.append("runaway_duration_for_short_prompt")

    return SmokeMetrics(
        prompt_text=prompt_text,
        transcript_text=transcript_text,
        duration_seconds=duration_seconds,
        processing_seconds=processing_seconds,
        sample_rate=sample_rate,
        samples=samples,
        prompt_word_count=len(prompt_tokens),
        transcript_word_count=len(transcript_tokens),
        sequence_ratio=sequence_ratio,
        word_jaccard=word_jaccard,
        common_word_fraction=common_word_fraction,
        repeated_bigram_fraction=bigram_fraction,
        longest_same_word_run=longest_run,
        transcript_words_per_second=transcript_wps,
        looks_suspicious=bool(reasons),
        suspicious_reasons=reasons,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Voxtral sample and score it with ASR-backed smoke metrics."
    )
    parser.add_argument("--tts-model", default="mlx-community/Voxtral-4B-TTS-2603-mlx-4bit")
    parser.add_argument("--asr-model", default="mlx-community/whisper-tiny-asr-fp16")
    parser.add_argument("--text", required=True, help="Text to synthesize and then validate.")
    parser.add_argument("--voice", default="casual_male")
    parser.add_argument("--lang-code", default="en")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--output-dir", default="smoke_outputs")
    parser.add_argument("--file-prefix", default="voxtral_smoke")
    parser.add_argument(
        "--delete-audio",
        action="store_true",
        help="Delete the generated WAV after scoring. JSON is always kept.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"{args.file_prefix}.wav"
    json_path = output_dir / f"{args.file_prefix}.json"

    print(f"Loading TTS model: {args.tts_model}")
    tts_model = load_tts(args.tts_model)
    print(f"Loading ASR model: {args.asr_model}")
    stt_model = load_stt(args.asr_model)

    print(f"Generating audio for: {args.text}")
    start = perf_counter()
    results = list(
        tts_model.generate(
            text=args.text,
            voice=args.voice,
            lang_code=args.lang_code,
            max_tokens=args.max_tokens,
        )
    )
    processing_seconds = perf_counter() - start
    if not results:
        raise RuntimeError("No TTS result was generated")

    audio = np.array(results[0].audio, dtype=np.float32).reshape(-1)
    sample_rate = int(results[0].sample_rate)
    audio_write(str(audio_path), audio, sample_rate, format="wav")

    # Re-read through the shared IO path so duration/sample accounting matches
    # what the downstream STT path will see on disk.
    audio_disk, sr_disk = audio_read(str(audio_path), always_2d=True)
    samples = int(audio_disk.shape[0])
    duration_seconds = samples / sr_disk if sr_disk > 0 else 0.0

    print(f"Transcribing generated audio: {audio_path}")
    stt_result = stt_model.generate(str(audio_path), language="en")
    transcript_text = getattr(stt_result, "text", str(stt_result)).strip()

    metrics = score_metrics(
        prompt_text=args.text,
        transcript_text=transcript_text,
        duration_seconds=duration_seconds,
        processing_seconds=processing_seconds,
        sample_rate=int(sr_disk),
        samples=samples,
    )

    json_path.write_text(json.dumps(asdict(metrics), indent=2) + "\n")

    print()
    print("Smoke summary")
    print(f"  Audio: {audio_path}")
    print(f"  JSON: {json_path}")
    print(f"  Duration: {metrics.duration_seconds:.2f}s")
    print(f"  Sequence ratio: {metrics.sequence_ratio:.3f}")
    print(f"  Word jaccard: {metrics.word_jaccard:.3f}")
    print(f"  Common word fraction: {metrics.common_word_fraction:.3f}")
    print(f"  Repeated bigram fraction: {metrics.repeated_bigram_fraction:.3f}")
    print(f"  Longest same-word run: {metrics.longest_same_word_run}")
    print(f"  Transcript words/sec: {metrics.transcript_words_per_second:.3f}")
    print(f"  Suspicious: {metrics.looks_suspicious}")
    print(f"  Reasons: {', '.join(metrics.suspicious_reasons) if metrics.suspicious_reasons else 'none'}")
    print()
    print("Transcript")
    print(metrics.transcript_text or "<empty>")

    if args.delete_audio:
        audio_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
