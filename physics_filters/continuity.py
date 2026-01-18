import cv2
import numpy as np

def crack_length(binary_mask):
    """
    Estimates crack length via contour analysis.
    Returns length in pixels.
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0

    return max(cv2.arcLength(cnt, False) for cnt in contours)
