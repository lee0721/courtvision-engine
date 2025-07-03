import numpy as np
import cv2 

class Homography:
    """
    A class to compute and apply a homography transformation between two 2D point sets.

    This is typically used for transforming points from one perspective (e.g., camera view)
    to another (e.g., top-down tactical view).
    """

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        """
        Initializes the Homography transformation using source and target keypoints.

        Args:
            source (np.ndarray): Array of 2D source points (N x 2).
            target (np.ndarray): Array of 2D target points (N x 2).

        Raises:
            ValueError: If source and target do not have the same shape or invalid dimensions.
        """
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")
        
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        
        self.m, _ = cv2.findHomography(source, target)
        if self.m is None:
            raise ValueError("Homography matrix could not be calculated.")
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Applies the homography transformation to a set of 2D points.

        Args:
            points (np.ndarray): Array of 2D points to transform (N x 2).

        Returns:
            np.ndarray: Transformed points (N x 2) in the target coordinate system.

        Raises:
            ValueError: If the input points are not 2D.
        """
        if points.size == 0:
            return points
        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")
        
        points = points.reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.m)
        return points.reshape(-1, 2).astype(np.float32)