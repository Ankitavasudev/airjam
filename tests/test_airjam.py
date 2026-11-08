"""Headless tests: no webcam, no mediapipe, no audio device."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from airjam.app import COOLDOWN, LANE_COLORS, OFF, ON, Session, pad_center, pinch_test, run_over_frame
from airjam.synth import CRASH, HIHAT, KICK, SNARE, NAMES, Player, RATE, sample
from airjam.tracker import HandState, HandTracker


# --- synth ----------------------------------------------------------------

def test_samples_finite_and_nonempty():
    for drum in (KICK, SNARE, HIHAT, CRASH):
        s = sample(drum)
        assert s.dtype == np.float32
        assert len(s) > 0
        assert np.isfinite(s).all()


def test_kick_has_energy_at_start_and_silence_at_end():
    s = sample(KICK)
    assert abs(s[:100]).mean() > 0.01
    assert abs(s[-2000:]).max() < 0.02  # decayed


def test_hihat_shorter_than_crash():
    assert len(sample(HIHAT)) < len(sample(CRASH))


def test_sample_cache_returns_same_object():
    assert sample(SNARE) is sample(SNARE)


def test_unknown_drum_raises():
    with pytest.raises(ValueError):
        sample(9)


def test_render_rate_constant():
    # every buffer length is whole frames at RATE
    assert len(sample(KICK)) < 5 * RATE


# --- pinch detection ------------------------------------------------------

def make_session():
    return Session(width=960, height=540)


def test_close_pinch_triggers_lane():
    s = make_session()
    pinch_test(s, [0.3, 0.9, 0.9, 0.9])
    assert s.hits == 1
    assert s.streak == 1


def test_hysteresis_prevents_double_fire_without_release():
    s = make_session()
    pinch_test(s, [0.3, 0.9, 0.9, 0.9])   # close -> fires
    n = s.hits
    pinch_test(s, [0.45, 0.9, 0.9, 0.9])  # inside hysteresis band -> no fire
    pinch_test(s, [0.1, 0.9, 0.9, 0.9])   # still closed -> no fire
    assert s.hits == n


def test_release_then_retrigger_respects_cooldown():
    s = make_session()
    pinch_test(s, [0.3, 0.9, 0.9, 0.9])     # fire #1
    pinch_test(s, [0.8, 0.9, 0.9, 0.9])     # release -> cooldown keeps counting
    pinch_test(s, [0.3, 0.9, 0.9, 0.9])     # re-close inside cooldown -> held
    for _ in range(COOLDOWN - 2):
        pinch_test(s, [0.3, 0.9, 0.9, 0.9])  # still cooling, still held
    assert s.hits == 1
    pinch_test(s, [0.8, 0.9, 0.9, 0.9])     # release between hits
    pinch_test(s, [0.3, 0.9, 0.9, 0.9])     # fresh pinch -> fire #2
    assert s.hits == 2
    assert s.best == 2


def test_each_finger_maps_to_its_own_lane():
    s = make_session()
    pinch = [0.9, 0.3, 0.9, 0.9]  # index 1 = middle finger -> snare
    pinch_test(s, pinch)
    assert s.last[0] == SNARE
    assert s.hits == 1


def test_streak_resets_never_happens_in_session():  # streak only grows here
    s = make_session()
    for i in range(3):
        pinch_test(s, [0.9, 0.9, 0.3, 0.9])
        pinch_test(s, [0.9, 0.9, 0.8, 0.9])
        for _ in range(COOLDOWN + 1):
            pinch_test(s, [0.9, 0.9, 0.8, 0.9])
    assert s.streak == 3
    assert s.best == 3


# --- renderer / pipeline ---------------------------------------------------

def hand_state(left: bool = True):
    tips = [[80, 240], [120, 180], [140, 160], [150, 190], [170, 210]]
    if not left:
        tips = [[880 - p[0] + 80, p[1]] for p in tips]
    return HandState(tips=tips, pinch=[0.9, 0.9, 0.9, 0.9], x=tips[1][0], y=tips[1][1])


def test_run_over_frame_returns_same_shape():
    s = make_session()
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    out = run_over_frame(frame.copy(), s, None)
    assert out.shape == frame.shape
    assert out.sum() > 0  # pads were drawn


def test_pipeline_triggers_and_draws():
    s = make_session()
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    hs = HandState(tips=[[80, 240], [120, 180], [140, 160], [150, 190], [170, 210]],
                   pinch=[0.3, 0.9, 0.9, 0.9], x=120, y=180)
    out = run_over_frame(frame.copy(), s, hs)
    assert s.hits == 1
    assert len(s.rings) == 1
    assert len(s.sparks) == 16
    # ring + sparks actually put pixels on screen
    assert out.sum() > frame.sum()
    assert s.floaters and "kick" in s.floaters[0][2]


def test_run_over_frame_resizes_input():
    s = make_session()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    out = run_over_frame(frame, s, None)
    assert out.shape == (s.height, s.width, 3)


def test_no_hand_does_not_crash():
    s = make_session()
    frame = np.full((540, 960, 3), 90, dtype=np.uint8)
    out = run_over_frame(frame.copy(), s, None)
    assert out.shape == frame.shape


def test_pad_centers_inside_frame():
    s = make_session()
    for lane in range(4):
        x, y = pad_center(s, lane)
        assert 0 <= x < s.width
        assert 0 <= y < s.height


def test_lane_colors_distinct():
    assert len(set(LANE_COLORS)) == 4


# --- tracker (import surface only - mediapipe is lazy) ---------------------

def test_hand_state_dataclass():
    hs = hand_state()
    assert hs.pinch == [0.9, 0.9, 0.9, 0.9]
    assert len(hs.tips) == 5


def test_tracker_lazy_import():
    t = HandTracker()
    assert t._mode is None  # nothing loaded until first process()


def test_names_match_drum_count():
    assert len(NAMES) == 4


def test_player_silent_when_no_device():
    p = Player()
    p.play(KICK)  # must not raise even without audio device


# --- wav / playback bytes ---------------------------------------------------

def test_wav_bytes_is_real_wav():
    from airjam.synth import wav_bytes
    data = wav_bytes(KICK)
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WAVE"
    assert len(data) > 1000


def test_wav_cache_is_stable_per_pitch():
    from airjam.synth import wav_play_bytes
    assert wav_play_bytes(HIHAT, 1.0) is wav_play_bytes(HIHAT, 1.0)


def test_repitch_shifts_length_position():
    from airjam.synth import repitch, sample, CRASH
    s = sample(CRASH)
    up = repitch(s, 1.25)
    assert up.shape == s.shape
    assert abs(up[len(up) // 4]) != abs(s[len(s) // 4])  # actually shifted


def test_combo_text_after_streak():
    s = make_session()
    for i in range(4):
        pinch_test(s, [0.9, 0.3, 0.9, 0.9])   # snare fires
        pinch_test(s, [0.9, 0.8, 0.9, 0.9])   # release
        for _ in range(COOLDOWN):
            pinch_test(s, [0.9, 0.8, 0.9, 0.9])
    assert s.streak == 4
    assert "x4" in s.floaters[-1][2]


def test_screen_shake_clears():
    s = make_session()
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    pinch_test(s, [0.3, 0.9, 0.9, 0.9])
    assert s.shake > 0
    for _ in range(20):
        run_over_frame(frame.copy(), s, None)  # frames pass, shake settles
    assert s.shake == 0


# --- v3: stage + beat --------------------------------------------------------

def test_beat_toggles():
    s = make_session()
    assert s.beat is False
    s.toggle_beat()
    assert s.beat is True
    s.toggle_beat()
    assert s.beat is False


def test_beat_plays_pattern_rings():
    from airjam.app import BEAT_TICKS
    s = make_session()
    s.toggle_beat()
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    for _ in range(BEAT_TICKS * 2 + 2):
        run_over_frame(frame.copy(), s, None)
    assert s.beat_hits >= 2          # the groove actually played
    assert s.hits == 0               # beat doesn't steal your score


def test_stage_is_session_sized():
    s = make_session()
    assert s.stage.shape == (540, 960, 3)


def test_camera_card_draws():
    s = make_session()
    frame = np.full((540, 960, 3), 60, dtype=np.uint8)
    out = run_over_frame(frame, s, None)
    corner = out[-40:, -40:]
    assert corner.sum() > 0  # card content ended up bottom-right
