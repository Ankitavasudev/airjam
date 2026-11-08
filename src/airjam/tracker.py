"""Hand tracker: fingertip positions + pinch strength off a webcam frame.

One module, two MediaPipe generations:
  * mediapipe >= 1.0 - the tasks API (HandLandmarker, VIDEO mode)
  * mediapipe 0.10.x - the legacy solutions API

MediaPipe imports lazily so the headless test suite never needs it.

Pinch pairs use fingertip distance normalized by hand scale
(wrist -> middle-finger MCP), so it works at any camera distance:
  lanes = (index, middle, ring, pinky) pinched against the thumb.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# landmark ids: wrist, thumb tip, index/middle/ring/pinky tips, middle MCP
WRIST, TIPS = 0, (4, 8, 12, 16, 20)
MIDDLE_MCP = 9


@dataclass
class HandState:
    """Fingertip screen positions + normalized pinch distances per finger."""

    tips: list[list[int]]          # [(x, y)] x5: thumb, index, middle, ring, pinky
    pinch: list[float]             # thumb-tip distance / hand scale, x4 (index..pinky)
    x: int                         # index fingertip x (for FX)
    y: int


class HandTracker:
    """One hand at a time. process() returns None when no hand is visible."""

    def __init__(self, max_hands: int = 1):
        self._mode: str | None = None
        self._landmarker = None
        self._hands = None
        self._max_hands = max_hands
        self._ts_ms = 0

    def _ensure(self):
        if self._landmarker is None and self._hands is None:
            import mediapipe as mp

            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self._mode = "legacy"
                self._hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self._max_hands,
                    min_detection_confidence=0.6,
                )
            else:
                self._mode = "tasks"
                self._landmarker = self._create_tasks(mp)
        return self._landmarker or self._hands

    def _create_tasks(self, mp):
        from pathlib import Path

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            import urllib.request

            print("downloading hand_landmarker.task (one-time)...")
            urllib.request.urlretrieve(MODEL_URL, model_path)

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self._max_hands,
            min_hand_detection_confidence=0.6,
        )
        return vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray) -> HandState | None:
        self._ts_ms += 33
        self._ensure()
        if self._mode == "tasks":
            lm = self._detect_tasks(frame_bgr)
        else:
            lm = self._detect_legacy(frame_bgr)
        if lm is None:
            return None
        return _state_from(lm, *frame_bgr.shape[:2][::-1])

    def _detect_tasks(self, frame_bgr: np.ndarray):
        import mediapipe as mp

        rgb = frame_bgr[:, :, ::-1].copy()  # BGR -> RGB
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, self._ts_ms)
        if not result.hand_landmarks:
            return None
        return result.hand_landmarks[0]

    def _detect_legacy(self, frame_bgr: np.ndarray):
        result = self._hands.process(frame_bgr)
        if not result.multi_hand_landmarks:
            return None
        return result.multi_hand_landmarks[0].landmark

    def close(self) -> None:
        self._landmarker = None
        self._hands = None
        self._mode = None


def _state_from(lm, w: int, h: int) -> HandState:
    scale = max(_dist(lm, MIDDLE_MCP, WRIST), 1e-6)
    tips = [[int(lm[i].x * w), int(lm[i].y * h)] for i in TIPS]

    # thumb-tip to each fingertip, normalized by hand size
    pinch = [_dist(lm, TIPS[k], TIPS[0]) / scale for k in (1, 2, 3, 4)]
    return HandState(tips=tips, pinch=pinch, x=tips[1][0], y=tips[1][1])


def _dist(lm, a: int, b: int) -> float:
    return float((lm[a].x - lm[b].x) ** 2 + (lm[a].y - lm[b].y) ** 2) ** 0.5
