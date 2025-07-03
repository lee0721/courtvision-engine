"""
A utility module providing functions for drawing shapes on video frames.

This module includes functions to draw triangles, ellipses, and rounded rectangles,
which are used to represent various annotations such as player IDs, ball pointers,
or team statistics in sports video analysis.
"""
import cv2 
import numpy as np
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

def draw_traingle(frame, bbox, color):
    """
    Draws a filled triangle (used for ball or ball possessor) on the given frame.

    Args:
        frame (np.ndarray): The target image to draw on.
        bbox (tuple): Bounding box (x, y, w, h) of the object.
        color (tuple): Triangle color in BGR.
    """
    y = int(bbox[1])
    x, _ = get_center_of_bbox(bbox)

    triangle_points = np.array([
        [x, y],
        [x - 10, y - 20],
        [x + 10, y - 20],
    ])
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
    return frame

def draw_ellipse(frame, bbox, color, track_id=None):
    """
    Draws an ellipse under a player to represent presence and optionally show their ID.

    Args:
        frame (np.ndarray): Frame to draw on.
        bbox (tuple): Bounding box of player (x1, y1, x2, y2).
        color (tuple): BGR color of the ellipse.
        track_id (int, optional): If provided, draws a label box with this ID.
    """
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)

    width = max(int(width), 1)
    height = max(int(0.35 * width), 1)

    try:
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(width, height),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )
    except cv2.error as e:
        print(f"[OpenCV 錯誤] ellipse 繪圖失敗：track_id={track_id} bbox={bbox} 寬度={width} 錯誤={e}")

    # Track ID label
    if track_id is not None:
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        cv2.rectangle(frame,
                      (int(x1_rect), int(y1_rect)),
                      (int(x2_rect), int(y2_rect)),
                      color,
                      cv2.FILLED)

        # Adjust text position based on number of digits
        x1_text = x1_rect + 12
        if int(track_id) > 99:
            x1_text -= 10

        cv2.putText(
            frame,
            f"{track_id}",
            (int(x1_text), int(y1_rect + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    return frame

def draw_rounded_rectangle(img, top_left, bottom_right, radius, color, alpha=1.0):
    """
    Draws a filled rounded rectangle with transparency.

    Args:
        img (np.ndarray): Frame to draw on.
        top_left (tuple): Top-left corner (x1, y1).
        bottom_right (tuple): Bottom-right corner (x2, y2).
        radius (int): Radius for rounded corners.
        color (tuple): Fill color in BGR.
        alpha (float): Transparency factor (1 = opaque).
    """
    overlay = img.copy()
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Body
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)

    # Corners
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)

    # Blend overlay with original image
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img