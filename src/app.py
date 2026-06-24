"""
Main Gesture Controller App
Run:  python src/app.py

Requires:
  - models/gesture_model.pkl   (from train.py)
  - models/label_encoder.pkl
"""

import customtkinter as ctk
import cv2
import mediapipe as mp
import joblib
import pyautogui
import numpy as np
import threading
import time
import json
import os
import sys
from PIL import Image, ImageTk, ImageDraw, ImageFilter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features   import extract_features, get_index_fingertip
from actions    import perform_action, AVAILABLE_ACTIONS, ACTION_LABELS

# ─── Theme ──────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG        = "#0d0f14"
PANEL     = "#13161e"
CARD      = "#1a1e2a"
ACCENT    = "#4f8ef7"
ACCENT2   = "#7c5cfc"
GREEN     = "#3ecf8e"
RED       = "#f75e5e"
TEXT      = "#e8eaf0"
SUBTEXT   = "#6b7280"
BORDER    = "#252a38"

GESTURE_COLORS = {
    0: "#4f8ef7",
    1: "#f75e5e",
    2: "#3ecf8e",
    3: "#f7c948",
    4: "#a78bfa",
}

CONFIG_PATH = "data/gesture_config.json"

# ─── Default gesture → action mapping ───────────────────────────────────────
DEFAULT_MAPPING = {
    "open_hand":  "move_mouse",
    "fist":       "left_click",
    "peace":      "scroll_up",
    "thumbs_up":  "volume_up",
    "pinch":      "right_click",
}


def load_config(gesture_names):
    if os.path.exists(CONFIG_PATH):
        try:
            return json.load(open(CONFIG_PATH))
        except Exception:
            pass
    return {g: DEFAULT_MAPPING.get(g, "nothing") for g in gesture_names}


def save_config(mapping):
    os.makedirs("data", exist_ok=True)
    json.dump(mapping, open(CONFIG_PATH, "w"), indent=2)


# ════════════════════════════════════════════════════════════════════════════
class GestureControllerApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Gesture OS Controller")
        self.geometry("1100x680")
        self.minsize(1000, 640)
        self.configure(fg_color=BG)

        # State
        self.active         = False
        self.cap            = None
        self.model          = None
        self.le             = None
        self.gesture_names  = []
        self.gesture_map    = {}
        self.current_gesture = ctk.StringVar(value="--")
        self.confidence_var  = ctk.DoubleVar(value=0.0)
        self.fps_var         = ctk.StringVar(value="0 fps")
        self.status_var      = ctk.StringVar(value="Ready")
        self.cam_image_ref   = None
        self._thread         = None

        self._load_model()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Model Loading ────────────────────────────────────────────────────────
    def _load_model(self):
        try:
            self.model = joblib.load("models/gesture_model.pkl")
            self.le    = joblib.load("models/label_encoder.pkl")
            self.gesture_names = list(self.le.classes_)
            self.gesture_map   = load_config(self.gesture_names)
            print(f"Model loaded. Gestures: {self.gesture_names}")
        except FileNotFoundError:
            self.gesture_names = list(DEFAULT_MAPPING.keys())
            self.gesture_map   = {g: DEFAULT_MAPPING.get(g, "nothing")
                                  for g in self.gesture_names}
            print("WARNING: No trained model found. Run collect_data.py then train.py first.")

    # ── UI Construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        self._build_left()
        self._build_right()

    def _build_left(self):
        left = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=16)
        left.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="nsew")
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        # Header
        hdr = ctk.CTkFrame(left, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 0))

        ctk.CTkLabel(hdr, text="◈  Gesture OS Controller",
                     font=ctk.CTkFont("Helvetica", 18, "bold"),
                     text_color=TEXT).pack(side="left")

        self.status_badge = ctk.CTkLabel(hdr, text="● Inactive",
                                          font=ctk.CTkFont("Helvetica", 12),
                                          text_color=SUBTEXT)
        self.status_badge.pack(side="right")

        # Camera feed
        cam_frame = ctk.CTkFrame(left, fg_color=CARD, corner_radius=12)
        cam_frame.grid(row=1, column=0, padx=20, pady=16, sticky="nsew")

        self.cam_label = ctk.CTkLabel(cam_frame, text="",
                                       fg_color=CARD, corner_radius=12)
        self.cam_label.pack(fill="both", expand=True, padx=2, pady=2)

        # Placeholder when inactive
        self._show_cam_placeholder()

        # Bottom controls
        ctrl = ctk.CTkFrame(left, fg_color="transparent")
        ctrl.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="ew")
        ctrl.grid_columnconfigure(0, weight=1)

        self.activate_btn = ctk.CTkButton(
            ctrl,
            text="▶   Activate Hand Control",
            font=ctk.CTkFont("Helvetica", 14, "bold"),
            fg_color=ACCENT, hover_color="#3a75e0",
            height=48, corner_radius=10,
            command=self._toggle
        )
        self.activate_btn.grid(row=0, column=0, sticky="ew")

        info = ctk.CTkFrame(ctrl, fg_color="transparent")
        info.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        self.gesture_badge = ctk.CTkLabel(info,
            text="Gesture: --",
            font=ctk.CTkFont("Helvetica", 13),
            text_color=ACCENT)
        self.gesture_badge.pack(side="left")

        self.fps_label = ctk.CTkLabel(info,
            textvariable=self.fps_var,
            font=ctk.CTkFont("Helvetica", 12),
            text_color=SUBTEXT)
        self.fps_label.pack(side="right")

    def _build_right(self):
        right = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=16)
        right.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="nsew")
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(right, text="Gesture Settings",
                     font=ctk.CTkFont("Helvetica", 16, "bold"),
                     text_color=TEXT).grid(row=0, column=0,
                                            padx=20, pady=(20, 4), sticky="w")
        ctk.CTkLabel(right, text="Remap any gesture to any action",
                     font=ctk.CTkFont("Helvetica", 12),
                     text_color=SUBTEXT).grid(row=1, column=0,
                                               padx=20, pady=(0, 12), sticky="w")

        # Scrollable gesture mapping area
        scroll = ctk.CTkScrollableFrame(right, fg_color="transparent",
                                         corner_radius=0)
        scroll.grid(row=2, column=0, padx=12, pady=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)

        self.mapping_widgets = {}   # gesture → CTkOptionMenu widget

        for i, gesture in enumerate(self.gesture_names):
            card = ctk.CTkFrame(scroll, fg_color=CARD, corner_radius=10)
            card.grid(row=i, column=0, padx=4, pady=6, sticky="ew")
            card.grid_columnconfigure(1, weight=1)

            # Colour dot
            color = GESTURE_COLORS.get(i % len(GESTURE_COLORS), ACCENT)
            dot = ctk.CTkLabel(card, text="●", text_color=color,
                                font=ctk.CTkFont("Helvetica", 18))
            dot.grid(row=0, column=0, padx=(14, 8), pady=14)

            # Gesture name
            ctk.CTkLabel(card, text=gesture.replace("_", " ").title(),
                         font=ctk.CTkFont("Helvetica", 13, "bold"),
                         text_color=TEXT, anchor="w").grid(
                         row=0, column=1, sticky="w")

            # Action dropdown
            action_labels = [ACTION_LABELS[a] for a in AVAILABLE_ACTIONS]
            current_action = self.gesture_map.get(gesture, "nothing")
            current_label  = ACTION_LABELS.get(current_action, "Do Nothing")

            var = ctk.StringVar(value=current_label)
            menu = ctk.CTkOptionMenu(
                card,
                variable=var,
                values=action_labels,
                width=160,
                fg_color=PANEL,
                button_color=ACCENT2,
                button_hover_color="#6a4de0",
                dropdown_fg_color=CARD,
                font=ctk.CTkFont("Helvetica", 12),
                command=lambda lbl, g=gesture: self._on_remap(g, lbl)
            )
            menu.grid(row=0, column=2, padx=12, pady=10)
            self.mapping_widgets[gesture] = (var, menu)

        # Live gesture indicator (bottom of right panel)
        live_card = ctk.CTkFrame(right, fg_color=CARD, corner_radius=10, height=90)
        live_card.grid(row=3, column=0, padx=12, pady=12, sticky="ew")
        live_card.grid_propagate(False)
        live_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(live_card, text="Live Detection",
                     font=ctk.CTkFont("Helvetica", 11),
                     text_color=SUBTEXT).grid(row=0, column=0, pady=(10, 0))

        self.live_gesture_label = ctk.CTkLabel(live_card, text="--",
                                                font=ctk.CTkFont("Helvetica", 20, "bold"),
                                                text_color=ACCENT)
        self.live_gesture_label.grid(row=1, column=0)

        self.conf_bar = ctk.CTkProgressBar(live_card, width=200,
                                            progress_color=ACCENT,
                                            fg_color=BORDER)
        self.conf_bar.set(0)
        self.conf_bar.grid(row=2, column=0, padx=20, pady=(4, 12))

        # Save reminder
        ctk.CTkLabel(right, text="Changes save automatically",
                     font=ctk.CTkFont("Helvetica", 11),
                     text_color=SUBTEXT).grid(row=4, column=0, pady=(0, 16))

    # ── Placeholder camera frame ─────────────────────────────────────────────
    def _show_cam_placeholder(self):
        w, h = 620, 400
        img = Image.new("RGB", (w, h), color=(26, 30, 42))
        draw = ImageDraw.Draw(img)

        # Draw crosshair / camera icon suggestion
        cx, cy = w // 2, h // 2
        r = 60
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), outline=(79, 142, 247, 80), width=2)
        draw.line((cx-r-20, cy, cx+r+20, cy), fill=(40, 50, 70), width=1)
        draw.line((cx, cy-r-20, cx, cy+r+20), fill=(40, 50, 70), width=1)

        for i in range(8):
            angle = i * 45
            import math
            rad = math.radians(angle)
            x1 = cx + (r+12) * math.cos(rad)
            y1 = cy + (r+12) * math.sin(rad)
            x2 = cx + (r+24) * math.cos(rad)
            y2 = cy + (r+24) * math.sin(rad)
            draw.line((x1, y1, x2, y2), fill=(50, 65, 95), width=1)

        img_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
        self.cam_label.configure(image=img_ctk, text="Click Activate to start",
                                  text_color=SUBTEXT,
                                  font=ctk.CTkFont("Helvetica", 14),
                                  compound="bottom")
        self.cam_label.image = img_ctk

    # ── Gesture remap ────────────────────────────────────────────────────────
    def _on_remap(self, gesture, label_str):
        # Find action key from label
        action = next((k for k, v in ACTION_LABELS.items() if v == label_str), "nothing")
        self.gesture_map[gesture] = action
        save_config(self.gesture_map)
        print(f"Remapped: {gesture} → {action}")

    # ── Toggle activation ────────────────────────────────────────────────────
    def _toggle(self):
        if not self.active:
            if self.model is None:
                self._show_error("No model found!\nRun collect_data.py then train.py first.")
                return
            self.active = True
            self.activate_btn.configure(text="⏹   Deactivate", fg_color=RED,
                                         hover_color="#d94040")
            self.status_badge.configure(text="● Active", text_color=GREEN)
            self.cam_label.configure(text="")
            self._thread = threading.Thread(target=self._camera_loop, daemon=True)
            self._thread.start()
        else:
            self.active = False
            self.activate_btn.configure(text="▶   Activate Hand Control",
                                         fg_color=ACCENT, hover_color="#3a75e0")
            self.status_badge.configure(text="● Inactive", text_color=SUBTEXT)
            self.live_gesture_label.configure(text="--", text_color=ACCENT)
            self.gesture_badge.configure(text="Gesture: --")
            self.conf_bar.set(0)
            self.fps_var.set("0 fps")
            self._show_cam_placeholder()

    # ── Camera + detection loop (runs in background thread) ─────────────────
    def _camera_loop(self):
        try:
            self._camera_loop_inner()
        except Exception as e:
            import traceback
            print("\n=== CAMERA THREAD ERROR ===")
            traceback.print_exc()
            print("===========================\n")
            # Reflect the failure back to the UI thread so it doesn't
            # silently sit at "Active" forever
            self.active = False

    def _camera_loop_inner(self):
        print("Opening camera...")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open camera at index 0. "
                "Try changing cv2.VideoCapture(0) to cv2.VideoCapture(1), "
                "or check if another program is using the camera."
            )
        print("Camera opened successfully.")

        mp_hands = mp.solutions.hands
        mp_draw  = mp.solutions.drawing_utils

        screen_w, screen_h = pyautogui.size()
        COOLDOWN    = 0.6
        last_action = 0
        last_gesture = None
        fps_counter  = 0
        fps_timer    = time.time()

        # Smooth mouse movement
        smooth_x, smooth_y = screen_w // 2, screen_h // 2
        SMOOTH = 0.25       # lower = smoother but more lag

        with mp_hands.Hands(max_num_hands=1,
                            min_detection_confidence=0.75,
                            min_tracking_confidence=0.75) as hands:

            while self.active:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                detected_gesture = None
                confidence       = 0.0

                if result.multi_hand_landmarks:
                    for hand_lm in result.multi_hand_landmarks:

                        # Draw landmarks with custom style
                        mp_draw.draw_landmarks(
                            frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                            mp_draw.DrawingSpec(color=(79, 142, 247),
                                                thickness=2, circle_radius=4),
                            mp_draw.DrawingSpec(color=(180, 180, 200),
                                                thickness=1))

                        feats = extract_features(hand_lm)
                        probs = self.model.predict_proba([feats])[0]
                        idx   = int(np.argmax(probs))
                        detected_gesture = self.le.classes_[idx]
                        confidence       = float(probs[idx])

                        # ── Mouse movement (always from fingertip) ──────────
                        action = self.gesture_map.get(detected_gesture, "nothing")
                        fx, fy = get_index_fingertip(hand_lm)

                        # Map camera coords → screen coords with margin
                        margin = 0.1
                        fx = max(margin, min(1 - margin, fx))
                        fy = max(margin, min(1 - margin, fy))
                        target_x = int((fx - margin) / (1 - 2*margin) * screen_w)
                        target_y = int((fy - margin) / (1 - 2*margin) * screen_h)

                        # Smooth
                        smooth_x = smooth_x + SMOOTH * (target_x - smooth_x)
                        smooth_y = smooth_y + SMOOTH * (target_y - smooth_y)
                        pyautogui.moveTo(int(smooth_x), int(smooth_y), duration=0)

                        # ── Gesture action with cooldown ────────────────────
                        now = time.time()
                        if (action != "move_mouse" and action != "nothing"
                                and now - last_action > COOLDOWN
                                and detected_gesture != last_gesture):
                            perform_action(action)
                            last_action  = now
                            last_gesture = detected_gesture

                        # Draw gesture label on frame
                        color_idx = list(self.le.classes_).index(detected_gesture) if detected_gesture else 0
                        color_hex = GESTURE_COLORS.get(color_idx % len(GESTURE_COLORS), "#4f8ef7")
                        r_c = int(color_hex[1:3], 16)
                        g_c = int(color_hex[3:5], 16)
                        b_c = int(color_hex[5:7], 16)
                        bgr  = (b_c, g_c, r_c)

                        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50),
                                      (13, 16, 26), -1)
                        cv2.putText(frame,
                                    f"{detected_gesture.replace('_',' ').upper()}  "
                                    f"{int(confidence*100)}%",
                                    (14, 34), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, bgr, 2)

                        mapped = ACTION_LABELS.get(action, action)
                        cv2.putText(frame, f"→ {mapped}",
                                    (frame.shape[1] - 200, 34),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (150, 150, 170), 1)

                # ── FPS counter ─────────────────────────────────────────────
                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    self.fps_var.set(f"{fps_counter} fps")
                    fps_counter = 0
                    fps_timer   = time.time()

                # ── Send frame to UI ────────────────────────────────────────
                self._update_ui_frame(frame, detected_gesture, confidence)

        if self.cap:
            self.cap.release()

    # ── Update UI from camera thread (thread-safe) ───────────────────────────
    def _update_ui_frame(self, frame_bgr, gesture, confidence):
        try:
            rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil  = Image.fromarray(rgb).resize((620, 400), Image.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil,
                                    size=(620, 400))
            self.cam_label.configure(image=ctk_img, text="")
            self.cam_label.image = ctk_img   # keep reference

            if gesture:
                display = gesture.replace("_", " ").title()
                self.gesture_badge.configure(text=f"Gesture: {display}")
                self.live_gesture_label.configure(text=display)
                self.conf_bar.set(confidence)

                # Color based on confidence
                color = GREEN if confidence > 0.85 else (ACCENT if confidence > 0.6 else RED)
                self.live_gesture_label.configure(text_color=color)
        except Exception:
            pass

    # ── Error dialog ──────────────────────────────────────────────────────────
    def _show_error(self, msg):
        dlg = ctk.CTkToplevel(self)
        dlg.title("Error")
        dlg.geometry("380x180")
        dlg.configure(fg_color=CARD)
        dlg.grab_set()
        ctk.CTkLabel(dlg, text=msg, font=ctk.CTkFont("Helvetica", 13),
                     text_color=TEXT, wraplength=320).pack(pady=30)
        ctk.CTkButton(dlg, text="OK", command=dlg.destroy,
                      fg_color=ACCENT).pack()

    # ── Clean shutdown ────────────────────────────────────────────────────────
    def _on_close(self):
        self.active = False
        time.sleep(0.2)
        if self.cap:
            self.cap.release()
        self.destroy()


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = GestureControllerApp()
    app.mainloop()