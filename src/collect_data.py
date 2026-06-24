"""
Run this script to collect your gesture training data.
You can run this MULTIPLE TIMES across multiple sessions —
new samples get APPENDED to your existing data, nothing gets lost.

Each time you press a number key to start a new gesture recording,
that's tagged as a new "session_id". This lets train.py use grouped
cross-validation later, which gives a more honest accuracy estimate
(since frames from the same continuous recording look almost
identical to each other and shouldn't be split across train/test).

Usage:
    python src/collect_data.py

Controls:
    Press 1-5 to select which gesture you're recording
    Press Q   to save and quit (safe to run again later to add more)
"""

import cv2
import mediapipe as mp
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import extract_features

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

CSV_PATH    = 'data/gestures.csv'
N_FEATURES  = 20   # output size of extract_features()

# --- Define your 5 gestures here ---
GESTURES = {
    '1': 'open_hand',
    '2': 'fist',
    '3': 'peace',
    '4': 'thumbs_up',
    '5': 'pinch',
}

COLORS = {
    'open_hand':  (100, 220, 100),
    'fist':       (220, 100, 100),
    'peace':      (100, 180, 220),
    'thumbs_up':  (220, 200, 100),
    'pinch':      (200, 100, 220),
}

FEATURE_COLS = [f'f{i}' for i in range(N_FEATURES)]
ALL_COLS     = FEATURE_COLS + ['label', 'session_id']

records       = []
current_label = None
cap           = cv2.VideoCapture(0)

os.makedirs('data', exist_ok=True)

# ── Load existing data if it exists, so we can show running totals ──────────
existing_counts = {}
session_offset  = 0
if os.path.exists(CSV_PATH):
    try:
        existing_df = pd.read_csv(CSV_PATH)
        existing_counts = existing_df['label'].value_counts().to_dict()
        if 'session_id' in existing_df.columns and len(existing_df) > 0:
            session_offset = int(existing_df['session_id'].max()) + 1
        print(f"Found existing data: {len(existing_df)} samples already saved")
        for g, c in existing_counts.items():
            print(f"  {g}: {c} existing samples")
    except Exception as e:
        print(f"Could not read existing CSV ({e}), starting fresh")
else:
    print("No existing data found, starting fresh")

session_id = session_offset

print("\n=== GESTURE DATA COLLECTOR ===")
print("Press 1-5 to select gesture, hold pose, move hand slightly for variation")
print("Aim for 250+ TOTAL samples per gesture (existing + new)")
print("Press Q to save and quit — safe to run multiple times, data appends\n")
for k, v in GESTURES.items():
    print(f"  [{k}] {v}")
print()

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif chr(key) in GESTURES:
            current_label = GESTURES[chr(key)]
            session_id += 1   # new key press = new session, even for same gesture
            print(f"Recording: {current_label} (session {session_id})")

        if result.multi_hand_landmarks and current_label:
            for hand_lm in result.multi_hand_landmarks:
                color = COLORS.get(current_label, (255, 255, 255))
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=4),
                    mp_draw.DrawingSpec(color=(200, 200, 200), thickness=1))
                feats = extract_features(hand_lm)
                records.append(list(feats) + [current_label, session_id])

        # HUD overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 56), (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        label_text    = current_label if current_label else "-- press 1-5 --"
        color         = COLORS.get(current_label, (180, 180, 180))
        new_count     = len([r for r in records if r[-2] == current_label]) if current_label else 0
        existing_count = existing_counts.get(current_label, 0) if current_label else 0
        cv2.putText(frame, f"Gesture: {label_text}", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"New: {new_count}  |  Total will be: {new_count + existing_count}",
                    (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # Legend on right
        for i, (k, v) in enumerate(GESTURES.items()):
            c = COLORS.get(v, (200, 200, 200))
            active = (v == current_label)
            existing_n = existing_counts.get(v, 0)
            cv2.putText(frame, f"[{k}] {v} ({existing_n})", (w - 220, 30 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        c if active else (120, 120, 120), 2 if active else 1)

        cv2.imshow("Gesture Collector  [Q = save & quit]", frame)

cap.release()
cv2.destroyAllWindows()

if records:
    new_df = pd.DataFrame(records, columns=ALL_COLS)

    # ── APPEND to existing CSV instead of overwriting ──────────────────────
    if os.path.exists(CSV_PATH):
        old_df = pd.read_csv(CSV_PATH)
        if 'session_id' not in old_df.columns:
            # Old data recorded before this fix had no session_id — backfill
            # with a single placeholder session so concat doesn't break.
            old_df['session_id'] = -1
        combined_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_csv(CSV_PATH, index=False)

    print(f"\nSaved {len(records)} new samples (appended)")
    print(f"Total dataset now: {len(combined_df)} samples\n")

    final_counts = combined_df['label'].value_counts().to_dict()
    for g in GESTURES.values():
        count = final_counts.get(g, 0)
        print(f"  {g}: {count} total samples {'✓' if count >= 200 else '⚠ need more'}")

    n_old_sessions = (combined_df['session_id'] == -1).sum()
    if n_old_sessions > 0:
        print(f"\nNote: {n_old_sessions} samples from before session tracking was added")
        print("have session_id = -1 (treated as one big session in grouped CV).")
else:
    print("No new data collected.")