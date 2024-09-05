import cv2
import numpy as np
import math

def calculate_euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def draw_distance_line(scene, point1, point2, distance):
    point1 = tuple(np.round(point1).astype(int))
    point2 = tuple(np.round(point2).astype(int))
    cv2.line(scene, point1, point2, (0, 255, 0), 2)
    text_position = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
    cv2.putText(scene, f"{distance:.2f} m", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 4)
    return scene

def calculate_ema(previous_ema, current_value, alpha):
    return alpha * current_value + (1 - alpha) * previous_ema


