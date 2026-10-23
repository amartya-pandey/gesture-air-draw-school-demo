import time
import av
import cv2
import numpy as np
import mediapipe as mp
import mediapipe as mp
try:
    from mediapipe import solutions as mp_solutions
except ImportError:
    from mediapipe.python import solutions as mp_solutions
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="Gesture Air Draw", page_icon="🎨", layout="wide")
st.title("🎨 Demo 2: Gesture Magic & Air Drawing")

st.markdown("""
**Controls:** ☝️ **1 Finger (Index):** Draw | ✌️ **2 Fingers (Peace):** Hover & Select Tool on Top Bar | 🖐️ **5 Fingers (Open Palm):** Hold 1 sec to Clear Canvas
""")

W, H = 1280, 720

PALETTE = [
    {"name": "Red",    "bgr": (0, 0, 255)},
    {"name": "Green",  "bgr": (0, 255, 0)},
    {"name": "Cyan",   "bgr": (255, 255, 0)},
    {"name": "Yellow", "bgr": (0, 230, 255)},
    {"name": "Purple", "bgr": (255, 0, 200)},
    {"name": "Eraser", "bgr": (0, 0, 0)},
    {"name": "Clear",  "bgr": (60, 60, 60)},
]

TIPS = [4, 8, 12, 16, 20]

def get_fingers_up(lm, handedness_label="Right"):
    fingers = []
    if handedness_label == "Right":
        thumb_up = 1 if lm[4].x < lm[3].x else 0
    else:
        thumb_up = 1 if lm[4].x > lm[3].x else 0
    fingers.append(thumb_up)

    pip_indices = [6, 10, 14, 18]
    for tip_idx, pip_idx in zip(TIPS[1:], pip_indices):
        fingers.append(1 if lm[tip_idx].y < lm[pip_idx].y else 0)
    return fingers

def draw_top_toolbar(img, active_color_idx):
    num_items = len(PALETTE)
    btn_w = W // num_items
    btn_h = 75

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (W, btn_h), (20, 24, 32), -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    for i, item in enumerate(PALETTE):
        bx1 = i * btn_w + 8
        by1 = 10
        bx2 = (i + 1) * btn_w - 8
        by2 = btn_h - 10

        is_selected = (i == active_color_idx)
        b_color = item["bgr"]

        if item["name"] == "Eraser":
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (45, 45, 55), -1)
            cv2.putText(img, "ERASER", (bx1 + 18, by1 + 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)
        elif item["name"] == "Clear":
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (60, 30, 30), -1)
            cv2.putText(img, "CLEAR ALL", (bx1 + 10, by1 + 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 120, 120), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (bx1, by1), (bx2, by2), b_color, -1)
            text_color = (0, 0, 0) if item["name"] in ["Yellow", "Green", "Cyan"] else (255, 255, 255)
            cv2.putText(img, item["name"].upper(), (bx1 + 20, by1 + 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, text_color, 2, cv2.LINE_AA)

        if is_selected:
            cv2.rectangle(img, (bx1 - 3, by1 - 3), (bx2 + 3, by2 + 3), (255, 255, 255), 3, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (90, 100, 115), 1, cv2.LINE_AA)
class AirDrawProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp_solutions.hands
        self.mp_draw = mp_solutions.drawing_utils
        self.mp_styles = mp_solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65
        )
        self.canvas = np.zeros((H, W, 3), dtype=np.uint8)
        self.active_color_idx = 0
        self.brush_thickness = 10
        self.eraser_thickness = 55
        self.prev_x, self.prev_y = None, None
        self.palm_start_time = None
        self.CLEAR_HOLD_SECS = 0.9
        self.last_clear_time = 0
        self.clear_requested = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        frame_bgr = cv2.resize(cv2.flip(img, 1), (W, H))

        if self.clear_requested:
            self.canvas[:] = 0
            self.clear_requested = False

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        res = self.hands.process(rgb_frame)

        finger_count = 0
        mode_text = "Show Hand"
        mode_color = (180, 180, 180)
        curr_tool = PALETTE[self.active_color_idx]["name"]
        is_eraser = (curr_tool == "Eraser")
        active_bgr = PALETTE[self.active_color_idx]["bgr"]

        if res.multi_hand_landmarks:
            hl = res.multi_hand_landmarks[0]
            handedness = "Right"
            if res.multi_handedness:
                raw_label = res.multi_handedness[0].classification[0].label
                handedness = "Right" if raw_label == "Left" else "Left"

            self.mp_draw.draw_landmarks(
                frame_bgr, hl, self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style()
            )

            lm = hl.landmark
            fingers = get_fingers_up(lm, handedness)
            finger_count = sum(fingers)

            raw_tip_x = int(lm[8].x * W)
            raw_tip_y = int(lm[8].y * H)

            if self.prev_x is None:
                smooth_x, smooth_y = raw_tip_x, raw_tip_y
            else:
                smooth_x = int(self.prev_x * 0.35 + raw_tip_x * 0.65)
                smooth_y = int(self.prev_y * 0.35 + raw_tip_y * 0.65)

            if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
                mode_text = "TOOL SELECTION / HOVER"
                mode_color = (255, 255, 0)
                self.prev_x, self.prev_y = None, None
                self.palm_start_time = None

                mid_x = int((lm[8].x + lm[12].x) * 0.5 * W)
                mid_y = int((lm[8].y + lm[12].y) * 0.5 * H)
                cv2.circle(frame_bgr, (mid_x, mid_y), 15, (255, 255, 255), 2, cv2.LINE_AA)

                if mid_y < 75:
                    btn_w = W // len(PALETTE)
                    selected_idx = mid_x // btn_w
                    if 0 <= selected_idx < len(PALETTE):
                        if PALETTE[selected_idx]["name"] == "Clear":
                            self.canvas[:] = 0
                            mode_text = "CANVAS CLEARED!"
                            mode_color = (0, 0, 255)
                        else:
                            self.active_color_idx = selected_idx

            elif fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                self.palm_start_time = None
                if is_eraser:
                    mode_text = "ERASING"
                    mode_color = (120, 120, 255)
                    if self.prev_x is not None and self.prev_y is not None:
                        cv2.line(self.canvas, (self.prev_x, self.prev_y), (smooth_x, smooth_y), (0, 0, 0), self.eraser_thickness)
                    cv2.circle(frame_bgr, (smooth_x, smooth_y), self.eraser_thickness // 2, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    mode_text = "DRAWING"
                    mode_color = (0, 255, 120)
                    if self.prev_x is not None and self.prev_y is not None:
                        cv2.line(self.canvas, (self.prev_x, self.prev_y), (smooth_x, smooth_y), active_bgr, self.brush_thickness, cv2.LINE_AA)
                    cv2.circle(frame_bgr, (smooth_x, smooth_y), self.brush_thickness + 2, active_bgr, -1, cv2.LINE_AA)

                self.prev_x, self.prev_y = smooth_x, smooth_y

            elif fingers[1] and fingers[2] and fingers[3] and fingers[4]:
                self.prev_x, self.prev_y = None, None
                mode_text = "HOLD TO CLEAR..."
                mode_color = (0, 140, 255)

                if self.palm_start_time is None:
                    self.palm_start_time = time.time()

                elapsed = time.time() - self.palm_start_time
                progress = min(1.0, elapsed / self.CLEAR_HOLD_SECS)

                wrist_x, wrist_y = int(lm[0].x * W), int(lm[0].y * H)
                cv2.circle(frame_bgr, (wrist_x, wrist_y), 32, (50, 50, 50), -1, cv2.LINE_AA)
                sweep = int(progress * 360)
                cv2.ellipse(frame_bgr, (wrist_x, wrist_y), (32, 32), 0, -90, -90 + sweep, (0, 0, 255), 4, cv2.LINE_AA)

                if elapsed >= self.CLEAR_HOLD_SECS and (time.time() - self.last_clear_time > 1.0):
                    self.canvas[:] = 0
                    mode_text = "CANVAS CLEARED!"
                    self.last_clear_time = time.time()
                    self.palm_start_time = None
            else:
                mode_text = "READY"
                mode_color = (200, 200, 200)
                self.prev_x, self.prev_y = None, None
                self.palm_start_time = None
        else:
            self.prev_x, self.prev_y = None, None
            self.palm_start_time = None

        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame_bgr, frame_bgr, mask=inv_mask)
        canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        frame_bgr = cv2.add(frame_bg, canvas_fg)

        draw_top_toolbar(frame_bgr, self.active_color_idx)

        hud_y = H - 55
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (0, hud_y), (W, H), (15, 18, 25), -1)
        cv2.addWeighted(overlay, 0.88, frame_bgr, 0.12, 0, frame_bgr)
        cv2.line(frame_bgr, (0, hud_y), (W, hud_y), (60, 70, 85), 1)

        cv2.putText(frame_bgr, f"Fingers: {finger_count}", (24, H - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 240, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"Mode: {mode_text}", (230, H - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, mode_color, 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"Tool: {curr_tool}", (W - 350, H - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
        if not is_eraser:
            cv2.circle(frame_bgr, (W - 180, H - 25), 14, active_bgr, -1, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")

ctx = webrtc_streamer(
    key="air-draw",
    video_processor_factory=AirDrawProcessor,
    media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

st.sidebar.header("🎨 Toolbar Controls")
tool_names = [item["name"] for item in PALETTE[:-1]]
selected_tool = st.sidebar.radio("Select Active Tool:", tool_names, index=0)
clear_btn = st.sidebar.button("🗑️ Clear Canvas")

if ctx.video_processor:
    ctx.video_processor.active_color_idx = tool_names.index(selected_tool)
    if clear_btn:
        ctx.video_processor.clear_requested = True
