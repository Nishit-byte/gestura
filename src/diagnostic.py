import cv2
import time

print("Script started", flush=True)

cap = cv2.VideoCapture(0)
print(f"isOpened: {cap.isOpened()}", flush=True)

time.sleep(1)  # give camera a moment to initialize

ret, frame = cap.read()
print(f"First read - ret: {ret}", flush=True)

if frame is not None:
    print(f"Frame shape: {frame.shape}", flush=True)
else:
    print("Frame is None", flush=True)

frame_count = 0
print("Entering loop, press Ctrl+C to stop", flush=True)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {frame_count}: read failed", flush=True)
            break
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: ok, shape={frame.shape}", flush=True)
        cv2.imshow("Diagnostic Test - press Q", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Q pressed, exiting", flush=True)
            break
except KeyboardInterrupt:
    print("Interrupted by user", flush=True)

cap.release()
cv2.destroyAllWindows()
print(f"Done. Total frames read: {frame_count}", flush=True)