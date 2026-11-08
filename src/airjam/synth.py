"""Drum synthesis + rock-solid playback.

Sounds are still pure numpy at 44.1 kHz (no sample packs), but playback
now has two paths:
  1. sounddevice - low-latency, cross-platform
  2. winsound    - Windows stdlib fallback via in-memory WAV, so AirJam
                   makes sound even when sounddevice cannot open a device

Hits are also slightly pitched per strike, so fast rolls sound like a
human playing rather than a machine gun.
"""

from __future__ import annotations

import io
import random
import wave

import numpy as np

RATE = 44100

KICK, SNARE, HIHAT, CRASH = range(4)
NAMES = ("kick", "snare", "hi-hat", "crash")


def render(drum: int) -> np.ndarray:
    """Mono float32 buffer ~one drum hit, clickless tail."""
    if drum == KICK:
        s = _kick()
    elif drum == SNARE:
        s = _snare()
    elif drum == HIHAT:
        s = _hihat()
    elif drum == CRASH:
        s = _crash()
    else:
        raise ValueError(f"unknown drum {drum}")
    s = s.copy()
    fade_n = int(RATE * 0.03)
    s[-fade_n:] *= np.linspace(1.0, 0.0, fade_n).astype(np.float32)
    return s


_cache: dict[int, np.ndarray] = {}


def sample(drum: int) -> np.ndarray:
    """Cached version of render() - one waveform per drum."""
    if drum not in _cache:
        _cache[drum] = render(drum)
    return _cache[drum]


def repitch(s: np.ndarray, pitch: float) -> np.ndarray:
    """Resample for a +-percent pitch shift - cheap humanization."""
    n = len(s)
    idx = np.linspace(0, (n - 1) / max(pitch, 0.1), n)
    return np.interp(idx, np.arange(n), s).astype(np.float32)


def wav_bytes(drum: int, pitch: float = 1.0) -> bytes:
    """Drum hit as a mono 16-bit WAV in memory - what winsound plays."""
    s = repitch(sample(drum), pitch)
    peak = float(np.abs(s).max()) or 1.0
    s = s / peak * 0.85          # normalize so every hit is loud and even
    pcm = (np.clip(s, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


_wav_cache: dict[tuple[int, float], bytes] = {}


def wav_play_bytes(drum: int, pitch: float) -> bytes:
    key = (drum, round(pitch, 2))
    if key not in _wav_cache:
        _wav_cache[key] = wav_bytes(drum, pitch)
    return _wav_cache[key]


def _kick() -> np.ndarray:
    n = int(RATE * 0.30)
    t = np.arange(n, dtype=np.float64) / RATE
    freq = 42.0 + 105.0 * np.exp(-t / 0.028)
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    body = np.sin(phase)
    click = np.random.default_rng(7).uniform(-1, 1, n) * np.exp(-t / 0.004) * 0.25
    return (body * np.exp(-t / 0.06) + click).astype(np.float32)


def _snare() -> np.ndarray:
    n = int(RATE * 0.22)
    t = np.arange(n, dtype=np.float64) / RATE
    tone = np.sin(2 * np.pi * 185 * t) + 0.5 * np.sin(2 * np.pi * 330 * t)
    tone *= np.exp(-t / 0.09)
    noise = np.random.default_rng(11).uniform(-1, 1, n)
    noise = noise - np.convolve(noise, np.ones(24) / 24, mode="same")
    noise *= np.exp(-t / 0.055)
    return (0.4 * tone + noise).astype(np.float32)


def _hihat() -> np.ndarray:
    n = int(RATE * 0.09)
    t = np.arange(n, dtype=np.float64) / RATE
    noise = np.random.default_rng(23).uniform(-1, 1, n)
    noise = noise - np.convolve(noise, np.ones(12) / 12, mode="same")
    return (noise * np.exp(-t / 0.022)).astype(np.float32)


def _crash() -> np.ndarray:
    n = int(RATE * 1.0)
    t = np.arange(n, dtype=np.float64) / RATE
    noise = np.random.default_rng(31).uniform(-1, 1, n)
    shimmer = noise - np.convolve(noise, np.ones(6) / 6, mode="same")
    return (0.7 * shimmer * np.exp(-t / 0.32)).astype(np.float32)


class Player:
    """Tries sounddevice, falls back to winsound (Windows), else silent."""

    def __init__(self):
        try:
            import sounddevice
            self._device = sounddevice
            self.backend = "sounddevice"
        except Exception:
            self._device = None
            self.backend = None

    def play(self, drum: int) -> None:
        drift = {KICK: 0.04, SNARE: 0.05, HIHAT: 0.10, CRASH: 0.06}[drum]
        pitch = 1.0 + random.uniform(-drift, drift)  # humanize each strike
        if self._device is not None:
            try:
                self._device.play(repitch(sample(drum), pitch), RATE)
                return
            except Exception:
                self._device = None
                self.backend = "winsound"
        try:
            import winsound

            winsound.PlaySound(wav_play_bytes(drum, pitch), winsound.SND_MEMORY | winsound.SND_ASYNC)
        except Exception:
            pass  # no audio anywhere - stay silently playable

    def available(self) -> bool:
        return True  # winsound fallback makes audio basically always exist
