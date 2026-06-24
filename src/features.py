"""
STEP 2 — Feature extraction.

Converts MediaPipe's 21 raw hand landmarks (each with x, y, z) into a
feature vector that's stable regardless of:
  - where your hand is in the frame (position invariant)
  - how close your hand is to the camera (scale invariant)

This is the part that turns "21 dots" into something a classifier
can actually learn gesture shapes from.
"""

import numpy as np


def extract_features(hand_landmarks):
    """
    Takes a MediaPipe hand_landmarks object (21 landmarks).
    Returns a 20-dimensional numpy array of normalized distances.

    Why distance-from-wrist instead of raw (x, y)?
    -----------------------------------------------
    Landmark 0 is always the wrist. Every other landmark's distance
    FROM the wrist describes the hand's shape (open, closed, pointing)
    without caring where the wrist itself is in the camera frame.

    Why divide by the max distance?
    --------------------------------
    A hand close to the camera produces bigger raw distances than the
    same hand far from the camera. Dividing every value by the largest
    one in that frame rescales everything to a 0-1 range, so "fist up
    close" and "fist far away" produce nearly identical feature vectors.
    """
    lm = hand_landmarks.landmark
    wrist = np.array([lm[0].x, lm[0].y])

    distances = []
    for i in range(1, 21):
        point = np.array([lm[i].x, lm[i].y])
        dist = np.linalg.norm(point - wrist)
        distances.append(dist)

    distances = np.array(distances)
    distances = distances / (distances.max() + 1e-6)  # +1e-6 avoids divide-by-zero

    return distances  # shape: (20,)


def get_index_fingertip(hand_landmarks):
    """
    Returns the normalized (x, y) position of the index fingertip
    (landmark 8). Used separately for mouse cursor control —
    this one we DON'T normalize by wrist/distance, because for
    cursor movement we want the actual screen position, not a
    shape descriptor.
    """
    tip = hand_landmarks.landmark[8]
    return tip.x, tip.y


if __name__ == "__main__":
    class FakeLandmark:
        def __init__(self, x, y, z=0):
            self.x, self.y, self.z = x, y, z

    class FakeHandLandmarks:
        def __init__(self, points):
            self.landmark = points

    fake_points = [FakeLandmark(0.5, 0.5)]
    for i in range(20):
        fake_points.append(FakeLandmark(0.5 + 0.02 * i, 0.5 - 0.01 * i))

    fake_hand = FakeHandLandmarks(fake_points)
    features = extract_features(fake_hand)

    print("Feature vector shape:", features.shape)
    print("Feature vector values:", features)
    print("Min:", features.min(), "Max:", features.max())
    print("\nIf max is 1.0 and values are between 0-1, the math is correct.")