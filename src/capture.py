"""
STEP 1 — Test that your camera and MediaPipe hand detection work.

Run this FIRST, before anything else.
This does NOT do any gesture recognition or OS control yet —
it just proves your camera + MediaPipe pipeline is working.

Usage:
    python src/capture.py

Controls:
    Press Q to quit
"""

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils


def main():
    print("Opening camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        print("Try changing cv2.VideoCapture(0) to cv2.VideoCapture(1)")
        return

    print("Camera opened. Show your hand to the camera.")
    print("Press Q in the window to quit.\n")

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame from camera.")
                break

            # Mirror the frame so movement feels natural
            frame = cv2.flip(frame, 1)

            # MediaPipe expects RGB, OpenCV gives BGR by default
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # If a hand is detected, draw the 21 landmarks + connections
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(79, 142, 247), thickness=2, circle_radius=4),
                        mp_draw.DrawingSpec(color=(200, 200, 200), thickness=1)
                    )

                # Simple on-screen confirmation
                cv2.putText(frame, "Hand detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
            else:
                cv2.putText(frame, "No hand detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)

            cv2.imshow("Step 1: Hand Detection Test  (press Q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera released. Test complete.")


if __name__ == "__main__":
    main()