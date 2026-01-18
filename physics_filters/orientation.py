import cv2
import numpy as np

def dominant_orientation(binary_mask):
    """
    Computes dominant line orientation using Hough Transform.
    Returns angle in degrees or None.
    """
    edges = cv2.Canny(binary_mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is None:
        return None

    angles = []
    for rho, theta in lines[:, 0]:
        angles.append(theta * 180 / np.pi)

    return np.median(angles)
