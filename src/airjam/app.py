"""Main AirJam loop: camera in, pinch detection out, drums + FX on screen.

The pipeline is one function, run_over_frame(frame, state), so headless
tests (no webcam, no mediapipe) can drive the exact same code path.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from .synth import CRASH, HIHAT, KICK, SNARE, NAMES, Player
from .tracker import HandState, HandTracker

LANE_COLORS = (        # bgr
    (60, 80, 255),     # kick - red
    (110, 255, 120),   # snare - green
    (255, 200, 90),    # hi-hat - sky
    (30, 200, 255),    # crash - orange
)
LANE_KEYS = ("q", "w", "e", "r")
ON, OFF = 0.42, 0.55   # pinch distance thresholds with hysteresis
COOLDOWN = 8           # frames between re-triggers of the same lane


class Session:
    """State threaded between frames: lanes, streak, FX."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.closed = [False] * 4      # hysteresis pinch state per lane
        self.cooling = [0] * 4
        self.hits = 0
        self.streak = 0
        self.best = 0
        self.last: tuple[int, int, int] | None = None  # (lane, x, y) latest hit
        self.rings: list[tuple[int, int, int, int]] = []  # (lane, x, y, age)
        self.sparks: list[list[int]] = []  # [x, y, vx, vy, age, lane]
        self.flash = [0] * 4
        self.tick = 0
        self.player = Player()

    def trigger(self, lane: int, x: int, y: int) -> None:
        self.hits += 1
        self.streak += 1
        self.best = max(self.best, self.streak)
        self.last = (lane, x, y)
        self.rings.append((lane, x, y, 0))
        rng = np.random.default_rng(self.tick * 4 + lane)
        for _ in range(14):
            angle = rng.uniform(0, 2 * math.pi)
            speed = rng.uniform(2.0, 7.0)
            self.sparks.append([
                x, y,
                int(math.cos(angle) * speed),
                int(math.cos(angle) * speed * 0.6) - 2,
                0, lane,
            ])
        self.flash[lane] = 12
        self.player.play(lane)


def pinch_test(session: Session, pinch: list[float]) -> None:
    """Hysteresis around raw pinch distances; state updates always, firing
    only when the lane is off cooldown, so one pinch is one hit."""
    for lane in range(4):
        closed = session.closed[lane]
        if not closed and pinch[lane] < ON:
            session.closed[lane] = True
            if session.cooling[lane] == 0:
                _fire(session, lane)  # arms the cooldown
        elif closed and pinch[lane] > OFF:
            session.closed[lane] = False
    for lane in range(4):
        if session.cooling[lane] > 0:
            session.cooling[lane] -= 1


def _fire(session: Session, lane: int) -> None:
    # pad centers are fixed; the FX burst is where the pinch is
    px, py = pad_center(session, lane)
    session.trigger(lane, px, py)
    session.cooling[lane] = COOLDOWN


def pad_center(session: Session, lane: int) -> tuple[int, int]:
    x = int(session.width * (lane + 1.6) / 5.2)
    y = int(session.height * 0.44)
    padding = PAD_RADIUS(session.height) + 6
    return min(x, session.width - padding), max(y, padding)


def PAD_RADIUS(height: int) -> int:
    return max(34, height // 7)


def run_over_frame(frame_bgr: np.ndarray, session: Session, hand: HandState | None) -> np.ndarray:
    """Advance one frame; returns the annotated display frame."""
    import cv2

    session.tick += 1
    if hand is not None:
        pinch_test(session, hand.pinch)
    _step_fx(session)

    frame = frame_bgr.copy()
    _fade(frame, 0.55)
    _draw_pads(frame, session)
    _draw_fx(frame, session)

    if hand is not None:
        for tip in hand.tips:
            cv2.circle(frame, tuple(tip), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(tip), 6, LANE_COLORS[2], 1, cv2.LINE_AA)
    _hud(frame, session)
    return frame


def _step_fx(session: Session) -> None:
    for i, (lane, x, y, age) in enumerate(session.rings):
        session.rings[i] = (lane, x, y, age + 1)
    session.rings[:] = [r for r in session.rings if r[3] < 26]
    for s in session.sparks:
        s[0] += s[2]
        s[1] += s[3]
        s[3] += 1
        s[4] += 1
    session.sparks[:] = [s for s in session.sparks if s[4] < 42]
    for lane in range(4):
        if session.flash[lane] > 0:
            session.flash[lane] -= 1


def _fade(frame: np.ndarray, amount: float) -> None:
    frame[...] = (frame.astype(np.float32) * amount + 30).astype(np.uint8)


def _draw_pads(frame: np.ndarray, session: Session) -> None:
    import cv2

    r = PAD_RADIUS(session.height)
    for lane in range(4):
        x, y = pad_center(session, lane)
        color = LANE_COLORS[lane]
        boost = session.flash[lane] * 4
        cv2.circle(frame, (x, y), r + boost, color, 2, cv2.LINE_AA)
        bob = int(6 * math.sin((session.tick + lane * 17) / 22.0))
        cv2.circle(frame, (x, min(y + bob, session.height - 1)), max(6, r - 18 + boost),
                   color, -1, cv2.LINE_AA)
        cv2.putText(frame, NAMES[lane], (x - 30, y + r - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _draw_fx(frame: np.ndarray, session: Session) -> None:
    import cv2

    for lane, x, y, age in session.rings:
        t = age / 26.0
        radius = int(20 + 130 * t)
        alpha = 1.0 - t
        color = tuple(int(c * alpha) for c in LANE_COLORS[lane])
        cv2.circle(frame, (x, y), radius, color, max(1, int(4 * (1 - t))), cv2.LINE_AA)
    for x, y, _, _, age, lane in session.sparks:
        c = tuple(int(c * (1.0 - age / 42.0)) for c in LANE_COLORS[lane])
        cv2.circle(frame, (x, y), 3, c, -1)
    top_flash = max(session.flash)
    if top_flash:
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1),
                      (top_flash * 8,) * 3, 6)


def _hud(frame: np.ndarray, session: Session) -> None:
    import cv2

    cv2.putText(frame, "AirJam - pinch to play", (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"hits {session.hits}   streak {session.streak}   best {session.best}",
                (14, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 200, 255), 1, cv2.LINE_AA)
    if session.streak >= 4:
        glow = 40 + int(150 * ((session.tick % 30) / 30))
        cv2.putText(frame, f"ON FIRE x{session.streak}", (frame.shape[1] - 320, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (glow, glow, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "thumb+index=kick  +middle=snare  +ring=hihat  +pinky=crash  |  q=quit",
                (14, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (150, 150, 150), 1, cv2.LINE_AA)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="airjam", description="Play drums with your fingers - webcam only.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args(argv)

    try:
        import mediapipe  # noqa: F401 - fail early, before the camera opens
    except ImportError:
        print("AirJam needs mediapipe: pip install mediapipe")
        return 2

    try:
        import cv2
    except ImportError:
        print("AirJam needs opencv-python: pip install opencv-python")
        return 2

    session = Session(width=args.width, height=args.height)
    session.player = Player()
    tracker = HandTracker()
    if sys.platform == "win32":
        cam = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not cam.isOpened():
            cam = cv2.VideoCapture(args.camera)
    else:
        cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        print("could not open camera %d" % args.camera)
        return 2
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("AirJam - pinch thumb to a finger to hit that drum. q to quit.")
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            hand = tracker.process(frame)
            out = run_over_frame(frame, session, hand)
            cv2.imshow("AirJam - air drum kit", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback

        traceback.print_exc()
        return 1
    finally:
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
