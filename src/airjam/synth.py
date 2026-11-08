"""Drum synthesis - kick, snare, hi-hat and crash as plain numpy waves.

Hand-rolled synthesis keeps AirJam dependency-free: no sample packs, no
audio files, just physics-flavored envelopes rendered at 44100 Hz. Each
drum is cached, so a repeated hit costs nothing.
"""

from __future__ import annotations

import numpy as np

RATE = 44100

KICK, SNARE, HIHAT, CRASH = range(4)
NAMES = ("kick", "snare", "hi-hat", "crash")


def render(drum: int) -> np.ndarray:
    """Mono float32 buffer for one drum hit, with a short tail fade."""
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
    fade_n = int(RATE * 0.03)                          # 30 ms clickless ending
    s[-fade_n:] *= np.linspace(1.0, 0.0, fade_n).astype(np.float32)
    return s


_cache: dict[int, np.ndarray] = {}


def sample(drum: int) -> np.ndarray:
    """Cached version of render() - one waveform per drum, forever."""
    if drum not in _cache:
        _cache[drum] = render(drum)
    return _cache[drum]


def _kick() -> np.ndarray:
    n = int(RATE * 0.30)
    t = np.arange(n, dtype=np.float64) / RATE
    freq = 42.0 + 105.0 * np.exp(-t / 0.028)          # pitch drops with time
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
    noise = noise - np.convolve(noise, np.ones(24) / 24, mode="same")  # crude high-pass
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
    """Plays cached buffers through the default output device.

    sounddevice is imported lazily; on machines without audio the play
    call silently does nothing, and AirJam stays fully playable as a
    (silent) visual experience.
    """

    def __init__(self):
        self._player = None

    def play(self, drum: int) -> None:
        if self._player is None:
            try:
                import sounddevice
            except Exception:
                self._player = False
        if not self._player:
            return
        self._player.play(sample(drum), RATE)

    def available(self) -> bool:
        return self._player is not False and self._player is not None
