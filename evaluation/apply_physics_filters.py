import os
import cv2
from physics_filters.orientation import dominant_orientation
from physics_filters.continuity import crack_length
from physics_filters.mask_utils import generate_binary_mask

FP_DIR = "evaluation/false_positives"

REJECTED = 0
ACCEPTED = 0

for fname in os.listdir(FP_DIR):
    path = os.path.join(FP_DIR, fname)
    img = cv2.imread(path)

    if img is None:
        continue

    mask = generate_binary_mask(img)

    angle = dominant_orientation(mask)
    length = crack_length(mask)

    # --- Physics Rules ---
    if angle is None:
        REJECTED += 1
        continue

    if length < 120:  # pixel threshold (tunable)
        REJECTED += 1
        continue

    ACCEPTED += 1

print("Physics Filter Results")
print(f"Accepted (likely cracks): {ACCEPTED}")
print(f"Rejected (false positives): {REJECTED}")
