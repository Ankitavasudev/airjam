"""Main AirJam v3: clean studio UI.

  * deep navy stage (camera no longer floods the screen - it lives in a
    small FaceTime-style card in the corner, which also hides the webcam's
    low resolution)
  * drum pads with a radial pinch-progress arc per lane - you SEE the hit
    charging up as thumb and finger approach
  * fingertip cursors + comet trail
  * SPACE toggles a backing beat (kick/hi-hat/snare groove) so even solo
    pinching sounds like music
  * 'f' toggles fullscreen window; q / ESC quits

Still one pipeline function, run_over_frame(frame, session, hand), so
headless tests drive the exact same code path.
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
BEAT_PATTERN = (KICK, HIHAT, SNARE, HIHAT, KICK, KICK, SNARE, HIHAT)
BEAT_TICKS = 11        # frames per beat step

ON, OFF = 0.45, 0.58   # pinch hysteresis thresholds
COOLDOWN = 6


class Session:
    """State threaded between frames."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.closed = [False] * 4
        self.cooling = [0] * 4
        self.hits = 0
        self.streak = 0
        self.best = 0
        self.last: tuple[int, int, int] | None = None
        self.rings: list[tuple[int, int, int, int]] = []
        self.sparks: list[list[int]] = []
        self.floaters: list[list] = []   # [x, y, text, age, lane]
        self.trail: list[tuple[int, int]] = []
        self.flash = [0] * 4
        self.shake = 0
        self.tick = 0
        self.player = Player()
        self.beat = False
        self.beat_step = -1             # frame tick of the next step
        self.beat_hits = 0
        self.stage = self._make_stage()

    def _make_stage(self) -> np.ndarray:
        """Deep navy vertical gradient - drawn once, copied every frame."""
        yy = np.linspace(0, 1, self.height)[:, None]
        base = np.zeros((self.height, self.width, 3), dtype=np.float32)
        base[..., 0] = (52 * (1 - yy) + 30)          # B
        base[..., 1] = (18 * (1 - yy) + 10)          # G
        base[..., 2] = (30 * (1 - yy) + 16)          # R
        return base.astype(np.uint8)

    def toggle_beat(self) -> None:
        self.beat = not self.beat
        self.beat_step = self.tick

    def trigger(self, lane: int) -> None:
        self.hits += 1
        self.streak += 1
        self.best = max(self.best, self.streak)
        self.last = (lane, timestamp_of(self.tick))
        self.rings.append((lane, 0, 0, 0))
        x, y = pad_center(self, lane)
        self.floaters.append([x, y - PAD_RADIUS(self.height) - 34, self._combo_text(), 0, lane])
        rng = np.random.default_rng(self.tick * 4 + lane)
        for _ in range(16):
            angle = rng.uniform(0, 2 * math.pi)
            speed = rng.uniform(2.0, 8.0)
            self.sparks.append([
                x, y,
                int(math.cos(angle) * speed),
                int(math.sin(angle) * speed * 0.7) - 2,
                0, lane,
            ])
        self.flash[lane] = 12
        self.shake = 7 if lane in (KICK, CRASH) else 4
        self.player.play(lane)

    def _combo_text(self) -> str:
        if self.last is None:
            return ""
        if self.streak >= 3:
            return "COMBO x%d" % self.streak
        return NAMES[self.last[0]]


def timestamp_of(tick: int) -> int:
    return tick * 33 % 100000


def pinch_test(session: Session, pinch: list[float]) -> None:
    """Hysteresis: state always updates, firing only when off cooldown."""
    for lane in range(4):
        closed = session.closed[lane]
        if not closed and pinch[lane] < ON:
            session.closed[lane] = True
            if session.cooling[lane] == 0:
                session.trigger(lane)
                session.cooling[lane] = COOLDOWN
        elif closed and pinch[lane] > OFF:
            session.closed[lane] = False
    for lane in range(4):
        if session.cooling[lane] > 0:
            session.cooling[lane] -= 1


def pad_center(session: Session, lane: int) -> tuple[int, int]:
    x = int((lane + 1) * session.width / 5.0)
    y = int(session.height * 0.47)
    r = PAD_RADIUS(session.height)
    return min(max(x, r + 16), session.width - r - 16), y


def PAD_RADIUS(height: int) -> int:
    return max(64, height // 8)


def run_over_frame(frame_bgr: np.ndarray, session: Session, hand: HandState | None) -> np.ndarray:
    """Advance one frame; returns the annotated display frame."""
    import cv2

    session.tick += 1
    if hand is not None:
        pinch_test(session, hand.pinch)
        session.trail.append((int(hand.x), int(hand.y)))
        if len(session.trail) > 26:
            del session.trail[:6]
    _beat_step(session)
    _step_fx(session, hand is not None)

    scene = session.stage.copy()
    _draw_beat_dot(scene, session)
    _draw_pads(scene, session)
    if hand is not None:
        _draw_pinch_arcs(scene, session, hand)
        _draw_cursor(scene, session, hand)
    _draw_fx(scene, session)
    _draw_camera_card(scene, session, frame_bgr)
    _hud(scene, session)

    if session.shake > 0:
        dx = int(np.random.default_rng(session.tick).uniform(-2.0, 2.0))
        dy = int(np.random.default_rng(session.tick + 1).uniform(-1.2, 1.2))
        mat = np.float32([[1, 0, dx], [0, 1, dy]])
        scene = cv2.warpAffine(scene, mat, (scene.shape[1], scene.shape[0]),
                               borderMode=cv2.BORDER_REPLICATE)
    return scene


def _beat_step(session: Session) -> None:
    if not session.beat:
        return
    while session.tick - session.beat_step >= BEAT_TICKS:
        session.beat_step += BEAT_TICKS
        drum = BEAT_PATTERN[session.beat_hits % len(BEAT_PATTERN)]
        session.beat_hits += 1
        session.player.play(drum)
        session.flash[drum] = max(session.flash[drum], 6)
        x, y = pad_center(session, drum)
        session.rings.append((drum, x, y, 0))


def _step_fx(session: Session, hand_present: bool) -> None:
    for i, (lane, x, y, age) in enumerate(session.rings):
        session.rings[i] = (lane, x, y, age + 1)
    session.rings[:] = [r for r in session.rings if r[3] < 28]
    for s in session.sparks:
        s[0] += s[2]
        s[1] += s[3]
        s[3] += 1
        s[4] += 1
    session.sparks[:] = [s for s in session.sparks if s[4] < 46]
    for f in session.floaters:
        f[1] -= 2
        f[3] += 1
    session.floaters[:] = [f for f in session.floaters if f[3] < 44]
    for lane in range(4):
        if session.flash[lane] > 0:
            session.flash[lane] -= 1
    if session.shake > 0:
        session.shake -= 1
    if not hand_present and session.trail:   # trail fades with the hand
        del session.trail[:3]


def _draw_beat_dot(frame: np.ndarray, session: Session) -> None:
    import cv2

    x, y = 46, session.height - 44
    if session.beat:
        phase = ((session.tick - session.beat_step) % BEAT_TICKS) / BEAT_TICKS
        r = int(8 + 10 * math.sin(phase * math.pi))
        cv2.circle(frame, (x, y), r, LANE_COLORS[2], -1, cv2.LINE_AA)
        cv2.putText(frame, "beat", (x + 16, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 220, 255), 1, cv2.LINE_AA)
    else:
        cv2.circle(frame, (x, y), 6, (90, 90, 90), -1, cv2.LINE_AA)


def _draw_pads(frame: np.ndarray, session: Session) -> None:
    import cv2

    r = PAD_RADIUS(session.height)
    for lane in range(4):
        x, y = pad_center(session, lane)
        color = LANE_COLORS[lane]
        boost = session.flash[lane] * 5
        bob = int(6 * math.sin((session.tick + lane * 19) / 26.0))
        yy = y + bob
        dark = tuple(int(c * 0.3) for c in color)
        cv2.circle(frame, (x + 4, yy + 7), r, dark, -1, cv2.LINE_AA)            # shadow
        faint = tuple(int(c * 0.30) for c in color)
        cv2.circle(frame, (x, yy), r + 16 + boost, faint, 2, cv2.LINE_AA)       # aura
        cv2.circle(frame, (x, yy), r + boost, color, 3, cv2.LINE_AA)            # rim
        core = tuple(min(255, int(c * 0.92)) for c in color)
        cv2.circle(frame, (x, yy), max(16, r - 26 + boost), core, -1, cv2.LINE_AA)
        cv2.circle(frame, (x - r // 5, yy - r // 4), max(5, r - 56), (250, 250, 250),
                   -1, cv2.LINE_AA)                                             # gloss
        scale = 0.85 if r < 80 else 1.05
        (tw, _), _ = cv2.getTextSize(NAMES[lane], cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(frame, NAMES[lane], (x - tw // 2, yy + r - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_pinch_arcs(frame: np.ndarray, session: Session, hand: HandState) -> None:
    """Radial charge arc per pad: distance thumb->fingertip -> visual %."""
    import cv2

    r = PAD_RADIUS(session.height)
    for lane in range(4):
        d = hand.pinch[lane]
        t = max(0.0, min(1.0, (1.05 - d) / 0.62))
        if t <= 0.02:
            continue
        x, y = pad_center(session, lane)
        angle = -90 + int(t * 295)
        color = LANE_COLORS[lane] if t >= 0.8 else tuple(int(c * 0.65) for c in LANE_COLORS[lane])
        cv2.ellipse(frame, (x, y), (r + 26, r + 26), 0, 0, angle, color, 6, cv2.LINE_AA)
        if t >= 0.8:
            cv2.putText(frame, "HIT", (x - 22, y - r - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)


def _draw_cursor(frame: np.ndarray, session: Session, hand: HandState) -> None:
    import cv2

    pts = session.trail[-22:]
    for i in range(1, len(pts)):
        t = i / len(pts)
        cv2.line(frame, pts[i - 1], pts[i], (int(210 * t), int(210 * t), int(255 * t)),
                 max(1, int(6 * t)), cv2.LINE_AA)
    for tip in hand.tips[1:]:
        cv2.circle(frame, tuple(tip), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(tip), 10, (170, 220, 255), 1, cv2.LINE_AA)
    tx, ty = hand.tips[0]
    cv2.circle(frame, (tx, ty), 7, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (tx, ty), 12, (210, 200, 130), 2, cv2.LINE_AA)


def _draw_fx(frame: np.ndarray, session: Session) -> None:
    import cv2

    for lane, x, y, age in session.rings:
        px, py = pad_center(session, lane)
        t = age / 28.0
        radius = int(PAD_RADIUS(session.height) * (1.0 + 1.35 * t))
        alpha = 1.0 - t
        color = tuple(int(c * alpha) for c in LANE_COLORS[lane])
        cv2.circle(frame, (px, py), radius, color, max(1, int(7 * (1 - t))), cv2.LINE_AA)
    for x, y, _, _, age, lane in session.sparks:
        fade = 1.0 - age / 46.0
        c = tuple(int(cc * fade) for cc in LANE_COLORS[lane])
        cv2.circle(frame, (x, y), 3, c, -1)
    for f in session.floaters:
        x, y, text, age, lane = f
        alpha = max(0.0, 1.0 - age / 44.0)
        color = tuple(int(c * alpha) + int((1 - alpha) * 60) for c in LANE_COLORS[lane])
        cv2.putText(frame, text, (x - 60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)


def _draw_camera_card(frame: np.ndarray, session: Session, frame_bgr: np.ndarray) -> None:
    """The webcam feed, small, bordered - FaceTime style bottom-right."""
    import cv2

    card_w = max(200, session.width // 5)
    h, w = frame_bgr.shape[:2]
    card_h = int(card_w * h / w)
    x0 = session.width - card_w - 26
    y0 = session.height - card_h - 26
    card = cv2.resize(frame_bgr, (card_w, card_h), interpolation=cv2.INTER_AREA)
    sharpened = cv2.addWeighted(card, 1.25, cv2.GaussianBlur(card, (0, 0), 3), -0.25, 0)
    cv2.rectangle(frame, (x0 + 5, y0 + 6), (x0 + card_w + 5, y0 + card_h + 6), (6, 6, 6), -1)  # shadow
    frame[y0:y0 + card_h, x0:x0 + card_w] = sharpened
    cv2.rectangle(frame, (x0 - 2, y0 - 2), (x0 + card_w + 2, y0 + card_h + 2), (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(frame, "camera", (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)


def _hud(frame: np.ndarray, session: Session) -> None:
    import cv2

    cv2.putText(frame, "AirJam", (26, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "air drum kit", (30, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 180, 210), 1, cv2.LINE_AA)
    texts = f"hits {session.hits}   streak {session.streak}   best {session.best}"
    (tw, _), _ = cv2.getTextSize(texts, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, texts, (frame.shape[1] - tw - 24, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (180, 220, 255), 2, cv2.LINE_AA)
    if session.streak >= 4:
        pulse = int(120 + 130 * abs(math.sin(session.tick / 8)))
        cv2.putText(frame, f"ON FIRE x{session.streak}", (frame.shape[1] // 2 - 110, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (pulse, pulse, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "pinch: thumb+index=kick  +middle=snare  +ring=hihat  +pinky=crash",
                (26, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "SPACE = beat   f = window   ESC = quit",
                (26, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="airjam", description="Play drums with your fingers - webcam only.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
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
    print("audio backend:", session.player.backend or "silent")
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

    print("AirJam 3.0 - pinch to play, SPACE for the beat, ESC to quit.")
    win = "AirJam - air drum kit"
    fullscreen = True
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(1)
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            hand = tracker.process(frame)
            out = run_over_frame(frame, session, hand)
            cv2.imshow(win, out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord(" "):
                session.toggle_beat()
            elif key == ord("f"):
                if fullscreen:
                    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    session.width, session.height = 960, 540
                    session.stage = session._make_stage()
                else:
                    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    session.width, session.height = args.width, args.height
                    session.stage = session._make_stage()
                fullscreen = not fullscreen
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
