import numpy as np
import cv2

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.M = None
        self.H = None
        self.perspectiveTransform(source, target)
        self.homographyTransform(source, target)

    def perspectiveTransform(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.M = cv2.getPerspectiveTransform(source, target)

    def homographyTransform(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.H, _ = cv2.findHomography(source, target, cv2.RANSAC, 1.0)

    def transformPointsPerspective(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0 or self.M is None:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.M)
        return transformed_points.reshape(-1, 2)

    def transformPointsHomography(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0 or self.H is None:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.H)
        return transformed_points.reshape(-1, 2)
