"""
A utility module providing functions for drawing shapes on video frames.

This module includes functions to draw triangles and ellipses on frames, which can be used
to represent various annotations such as player positions or ball locations in sports analysis.
"""

import cv2 
import numpy as np
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

def draw_traingle(frame,bbox,color):
    """
    Draws a filled triangle on the given frame at the specified bounding box location.

    Args:
        frame (numpy.ndarray): The frame on which to draw the triangle.
        bbox (tuple): A tuple representing the bounding box (x, y, width, height).
        color (tuple): The color of the triangle in BGR format.

    Returns:
        numpy.ndarray: The frame with the triangle drawn on it.
    """
    y= int(bbox[1])
    x,_ = get_center_of_bbox(bbox)

    triangle_points = np.array([
        [x,y],
        [x-10,y-20],
        [x+10,y-20],
    ])
    cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

    return frame

def draw_ellipse(frame,bbox,color,track_id=None):
    """
    Draws an ellipse and an optional rectangle with a track ID on the given frame at the specified bounding box location.

    Args:
        frame (numpy.ndarray): The frame on which to draw the ellipse.
        bbox (tuple): A tuple representing the bounding box (x, y, width, height).
        color (tuple): The color of the ellipse in BGR format.
        track_id (int, optional): The track ID to display inside a rectangle. Defaults to None.

    Returns:
        numpy.ndarray: The frame with the ellipse and optional track ID drawn on it.
    """
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)

    cv2.ellipse(
        frame,
        center=(x_center,y2),
        axes=(int(width), int(0.35*width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color = color,
        thickness=2,
        lineType=cv2.LINE_4
    )

    rectangle_width = 40
    rectangle_height=20
    x1_rect = x_center - rectangle_width//2
    x2_rect = x_center + rectangle_width//2
    y1_rect = (y2- rectangle_height//2) +15
    y2_rect = (y2+ rectangle_height//2) +15

    if track_id is not None:
        cv2.rectangle(frame,
                        (int(x1_rect),int(y1_rect) ),
                        (int(x2_rect),int(y2_rect)),
                        color,
                        cv2.FILLED)
        
        x1_text = x1_rect+12
        if track_id > 99:
            x1_text -=10
        
        cv2.putText(
            frame,
            f"{track_id}",
            (int(x1_text),int(y1_rect+15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,0),
            2
        )

    return frame

def draw_rounded_rectangle(img, top_left, bottom_right, radius, color, alpha=1.0):
    """
    Draws a filled rounded rectangle on the image with optional transparency.

    Args:
        img (np.ndarray): The target image to draw on.
        top_left (tuple): (x1, y1) of rectangle.
        bottom_right (tuple): (x2, y2) of rectangle.
        radius (int): Radius of corner circles.
        color (tuple): Fill color in BGR (e.g., (255,255,255)).
        alpha (float): Transparency (1.0 = solid, 0 = invisible).

    Returns:
        np.ndarray: The image with the rounded rectangle drawn.
    """
    overlay = img.copy()
    x1, y1 = top_left
    x2, y2 = bottom_right

    # 中間矩形
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)

    # 四個圓角
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)

    # 套用透明度
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    return img