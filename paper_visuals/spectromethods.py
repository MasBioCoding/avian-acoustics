#!/usr/bin/env python3
"""
Generate a viridis mel spectrogram for a selected time window in a single audio file.

Typical usage:
    python paper_visuals/spectromethods.py /Volumes/Z Slim/zslim_birdcluster/xc_downloads/emberiza_citrinella/emberiza_citrinella_511818.mp3 --start 10 --end 120 \
        --output /Users/masjansma/Desktop/scriptie/method_visuals/longspectro_2.png
    python paper_visuals/spectromethods.py /path/to/recording.wav --start 0 --end 8 \
        --output paper_visuals/output/example.png
"""


from __future__ import annotations

import argparse
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "birdnet_paper_visuals_cache"
for _cache_dir in (
    _CACHE_ROOT / "matplotlib",
    _CACHE_ROOT / "xdg",
    _CACHE_ROOT / "numba",
):
    _cache_dir.mkdir(parents=True, exist_ok=True)

# Keep plotting/cache side effects inside a writable temp directory.
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(_CACHE_ROOT / "numba"))

import audioread
import matplotlib
import numpy as np
import soundfile as sf
from scipy import signal

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


@dataclass(frozen=True)
class SpectrogramRequest:
    """Inputs needed to render a single mel spectrogram image."""

    audio_path: Path
    output_path: Path | None
    start: float
    end: float
    sample_rate: int
    n_fft: int
    hop_length: int
    n_mels: int
    max_frequency: float
    top_db: float
    dpi: int
    width: float
    height: float


@dataclass(frozen=True)
class LoadedAudio:
    """Decoded audio window plus metadata used for plotting."""

    samples: np.ndarray
    sample_rate: int
    duration: float
    actual_start: float
    actual_end: float


def format_seconds(value: float) -> str:
    """Return a compact human-readable seconds string."""

    return f"{value:.2f}".rstrip("0").rstrip(".")


def format_seconds_for_filename(value: float) -> str:
    """Return a filename-safe seconds token."""

    return format_seconds(value).replace(".", "p")


def join_path_parts(parts: str | list[str]) -> Path:
    """Rebuild a path from one or more CLI tokens."""

    if isinstance(parts, str):
        return Path(parts)
    return Path(" ".join(parts))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a viridis mel spectrogram for a single audio file over a "
            "selected time range. Audio is resampled to 32 kHz and capped at "
            "16 kHz by default."
        )
    )
    parser.add_argument(
        "audio_path",
        nargs="+",
        help=(
            "Path to an input audio file such as .mp3, .wav, or .flac. "
            "Paths containing spaces are supported."
        ),
    )
    parser.add_argument(
        "--start",
        type=float,
        default=10.0,
        help="Window start time in seconds (default: 10).",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=40.0,
        help="Window end time in seconds before clamping to the file duration (default: 40).",
    )
    parser.add_argument(
        "--output",
        nargs="+",
        help=(
            "Output image path. Defaults to "
            "paper_visuals/output/<stem>_<start>s_<end>s_viridis.png."
        ),
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=32000,
        help="Resample rate in Hz (default: 32000).",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=2048,
        help="FFT window size used for the spectrogram.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="Hop length between STFT frames.",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=128,
        help="Number of mel bands to render (default: 128).",
    )
    parser.add_argument(
        "--max-frequency",
        type=float,
        default=16000.0,
        help="Upper frequency limit in Hz for the mel spectrogram (default: 16000).",
    )
    parser.add_argument(
        "--top-db",
        type=float,
        default=80.0,
        help="Dynamic range used for decibel conversion.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=10.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=4.0,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution in dots per inch.",
    )
    return parser.parse_args()


def build_request(args: argparse.Namespace) -> SpectrogramRequest:
    """Validate CLI arguments and convert them into a typed request."""

    audio_path = join_path_parts(args.audio_path).expanduser()
    output_path = join_path_parts(args.output).expanduser() if args.output else None

    if not audio_path.is_file():
        raise SystemExit(f"Audio file not found: {audio_path}")
    if args.start < 0:
        raise SystemExit("--start must be greater than or equal to 0.")
    if args.end <= args.start:
        raise SystemExit("--end must be greater than --start.")
    if args.n_fft <= 0:
        raise SystemExit("--n-fft must be a positive integer.")
    if args.hop_length <= 0:
        raise SystemExit("--hop-length must be a positive integer.")
    if args.n_mels <= 0:
        raise SystemExit("--n-mels must be a positive integer.")
    if args.max_frequency <= 0:
        raise SystemExit("--max-frequency must be greater than 0.")
    if args.dpi <= 0:
        raise SystemExit("--dpi must be a positive integer.")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive numbers.")
    if args.sample_rate <= 0:
        raise SystemExit("--sample-rate must be a positive integer.")

    return SpectrogramRequest(
        audio_path=audio_path,
        output_path=output_path,
        start=float(args.start),
        end=float(args.end),
        sample_rate=int(args.sample_rate),
        n_fft=int(args.n_fft),
        hop_length=int(args.hop_length),
        n_mels=int(args.n_mels),
        max_frequency=float(args.max_frequency),
        top_db=float(args.top_db),
        dpi=int(args.dpi),
        width=float(args.width),
        height=float(args.height),
    )


def resolve_time_window(duration: float, start: float, end: float) -> tuple[float, float, float]:
    """Return the file duration plus a valid start/end window."""

    if duration <= 0:
        raise SystemExit("Audio file appears to be empty or unreadable.")
    if start >= duration:
        raise SystemExit(
            "Requested start time "
            f"({format_seconds(start)}s) is outside the file duration "
            f"({format_seconds(duration)}s)."
        )

    actual_end = min(end, duration)
    if actual_end <= start:
        raise SystemExit(
            "Requested window is empty after clamping to the file duration: "
            f"{format_seconds(start)}s to {format_seconds(actual_end)}s."
        )

    return duration, start, actual_end


def resample_audio(samples: np.ndarray, original_sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    """Resample mono audio to the requested sample rate."""

    if target_sr == original_sr:
        return samples.astype(np.float32, copy=False), int(original_sr)

    ratio = math.gcd(original_sr, target_sr)
    resampled = signal.resample_poly(samples, target_sr // ratio, original_sr // ratio)
    return resampled.astype(np.float32, copy=False), int(target_sr)


def hz_to_mel(frequencies_hz: np.ndarray | float) -> np.ndarray:
    """Convert frequencies in Hz to the mel scale."""

    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + (frequencies / 700.0))


def mel_to_hz(mel_values: np.ndarray | float) -> np.ndarray:
    """Convert mel-scaled values back to Hz."""

    mel = np.asarray(mel_values, dtype=np.float64)
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filter_bank(
    frequencies: np.ndarray,
    sample_rate: int,
    n_mels: int,
    max_frequency: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a triangular mel filter bank aligned to FFT frequency bins."""

    nyquist = sample_rate / 2.0
    capped_max_frequency = min(max_frequency, nyquist)
    if capped_max_frequency <= 0:
        raise SystemExit("The mel frequency limit must be above 0 Hz.")

    mel_points = np.linspace(
        hz_to_mel(0.0),
        hz_to_mel(capped_max_frequency),
        num=n_mels + 2,
        dtype=np.float64,
    )
    hz_points = mel_to_hz(mel_points)
    filter_bank = np.zeros((n_mels, frequencies.size), dtype=np.float32)
    eps = np.finfo(np.float32).eps

    for index in range(n_mels):
        left = hz_points[index]
        center = hz_points[index + 1]
        right = hz_points[index + 2]

        lower_slope = (frequencies - left) / max(center - left, eps)
        upper_slope = (right - frequencies) / max(right - center, eps)
        filter_bank[index] = np.maximum(0.0, np.minimum(lower_slope, upper_slope))

    normalization = 2.0 / np.maximum(hz_points[2:] - hz_points[:-2], eps)
    filter_bank *= normalization[:, np.newaxis]

    mel_plot_edges = mel_to_hz(
        np.linspace(
            hz_to_mel(0.0),
            hz_to_mel(capped_max_frequency),
            num=n_mels + 1,
            dtype=np.float64,
        )
    )
    return filter_bank, mel_plot_edges.astype(np.float32), float(capped_max_frequency)


def centers_to_edges(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Convert monotonic center coordinates into plotting edges."""

    centers = np.asarray(values, dtype=np.float64)
    if centers.size == 0:
        return np.asarray([lower, upper], dtype=np.float64)
    if centers.size == 1:
        return np.asarray([lower, upper], dtype=np.float64)

    midpoints = (centers[:-1] + centers[1:]) / 2.0
    return np.concatenate(([lower], midpoints, [upper]))


def load_audio_with_soundfile(request: SpectrogramRequest) -> LoadedAudio:
    """Load the requested audio window via libsndfile when supported."""

    info = sf.info(str(request.audio_path))
    duration, actual_start, actual_end = resolve_time_window(
        float(info.frames) / float(info.samplerate),
        request.start,
        request.end,
    )

    start_frame = max(int(math.floor(actual_start * info.samplerate)), 0)
    end_frame = max(int(math.ceil(actual_end * info.samplerate)), start_frame + 1)
    audio, native_sr = sf.read(
        str(request.audio_path),
        start=start_frame,
        stop=end_frame,
        dtype="float32",
        always_2d=True,
    )
    if audio.size == 0:
        raise RuntimeError("SoundFile returned no samples for the requested window.")

    mono = audio.mean(axis=1)
    samples, sample_rate = resample_audio(mono, int(native_sr), request.sample_rate)
    return LoadedAudio(
        samples=samples,
        sample_rate=sample_rate,
        duration=duration,
        actual_start=actual_start,
        actual_end=actual_end,
    )


def load_audio_with_audioread(request: SpectrogramRequest) -> LoadedAudio:
    """Load the requested audio window through audioread as a broader fallback."""

    with audioread.audio_open(str(request.audio_path)) as input_file:
        duration, actual_start, actual_end = resolve_time_window(
            float(input_file.duration),
            request.start,
            request.end,
        )
        sample_rate = int(input_file.samplerate)
        channels = int(input_file.channels)
        start_frame = max(int(math.floor(actual_start * sample_rate)), 0)
        end_frame = max(int(math.ceil(actual_end * sample_rate)), start_frame + 1)

        current_frame = 0
        chunks: list[np.ndarray] = []

        for chunk in input_file:
            buffer = np.frombuffer(chunk, dtype="<i2")
            if buffer.size == 0:
                continue

            remainder = buffer.size % channels
            if remainder:
                buffer = buffer[:-remainder]
            if buffer.size == 0:
                continue

            frames = buffer.reshape(-1, channels)
            next_frame = current_frame + frames.shape[0]

            overlap_start = max(start_frame, current_frame)
            overlap_end = min(end_frame, next_frame)
            if overlap_start < overlap_end:
                local_start = overlap_start - current_frame
                local_end = overlap_end - current_frame
                chunks.append(frames[local_start:local_end])

            current_frame = next_frame
            if current_frame >= end_frame:
                break

    if not chunks:
        raise RuntimeError("audioread returned no samples for the requested window.")

    audio = np.concatenate(chunks, axis=0).astype(np.float32) / 32768.0
    mono = audio.mean(axis=1)
    samples, output_sr = resample_audio(mono, sample_rate, request.sample_rate)
    return LoadedAudio(
        samples=samples,
        sample_rate=output_sr,
        duration=duration,
        actual_start=actual_start,
        actual_end=actual_end,
    )


def load_audio_window(request: SpectrogramRequest) -> LoadedAudio:
    """Load audio samples using SoundFile first, then fall back to audioread."""

    try:
        return load_audio_with_soundfile(request)
    except Exception:
        try:
            return load_audio_with_audioread(request)
        except Exception as exc:
            raise SystemExit(
                f"Could not decode audio file '{request.audio_path}'. "
                "Install support for the file format or try converting it to WAV."
            ) from exc


def resolve_output_path(
    request: SpectrogramRequest,
    actual_start: float,
    actual_end: float,
) -> Path:
    """Return the final image path and ensure its parent directory exists."""

    if request.output_path is not None:
        output_path = request.output_path
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".png")
    else:
        output_path = DEFAULT_OUTPUT_DIR / (
            f"{request.audio_path.stem}_"
            f"{format_seconds_for_filename(actual_start)}s_"
            f"{format_seconds_for_filename(actual_end)}s_viridis.png"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def render_spectrogram(
    request: SpectrogramRequest,
) -> tuple[Path, float, float]:
    """Render a spectrogram image and return output path plus timing metadata."""

    loaded = load_audio_window(request)
    output_path = resolve_output_path(request, loaded.actual_start, loaded.actual_end)

    if loaded.samples.size < 2:
        raise SystemExit(
            "No audio samples were loaded for the requested window: "
            f"{format_seconds(loaded.actual_start)}s to {format_seconds(loaded.actual_end)}s."
        )

    window_size = min(request.n_fft, int(loaded.samples.size))
    if window_size < 2:
        raise SystemExit("The selected audio window is too short to compute a spectrogram.")

    overlap = min(max(window_size - request.hop_length, 0), window_size - 1)
    frequencies, times, spectrum = signal.spectrogram(
        loaded.samples,
        fs=loaded.sample_rate,
        window="hann",
        nperseg=window_size,
        noverlap=overlap,
        detrend=False,
        scaling="spectrum",
        mode="psd",
    )
    mel_filter_bank, mel_edges_hz, capped_max_frequency = build_mel_filter_bank(
        frequencies=frequencies,
        sample_rate=loaded.sample_rate,
        n_mels=request.n_mels,
        max_frequency=request.max_frequency,
    )
    mel_power = np.maximum(mel_filter_bank @ spectrum, 1e-12)
    spectrogram_db = 10.0 * np.log10(mel_power)
    spectrogram_db -= float(np.max(spectrogram_db))
    spectrogram_db = np.maximum(spectrogram_db, -request.top_db)

    times = times + loaded.actual_start
    time_edges = centers_to_edges(times, lower=loaded.actual_start, upper=loaded.actual_end)

    fig, ax = plt.subplots(figsize=(request.width, request.height))
    image = ax.pcolormesh(
        time_edges,
        mel_edges_hz,
        spectrogram_db,
        shading="auto",
        cmap="viridis",
        vmin=-request.top_db,
        vmax=0.0,
    )
    ax.set_title(
        f"{request.audio_path.name} ({format_seconds(loaded.actual_start)}s to "
        f"{format_seconds(loaded.actual_end)}s)"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz, mel scale)")
    ax.set_ylim(0.0, capped_max_frequency)

    colorbar = fig.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label("Relative intensity (dB)")

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=request.dpi,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)

    return output_path, loaded.duration, loaded.actual_end


def main() -> None:
    """Run the CLI."""

    request = build_request(parse_args())
    output_path, duration, actual_end = render_spectrogram(request)
    clamped_note = " (end clamped to file duration)" if actual_end < request.end else ""

    print(
        f"Saved spectrogram to {output_path}. "
        f"Window: {format_seconds(request.start)}s to {format_seconds(actual_end)}s"
        f"{clamped_note}. "
        f"File duration: {format_seconds(duration)}s."
    )


if __name__ == "__main__":
    main()
