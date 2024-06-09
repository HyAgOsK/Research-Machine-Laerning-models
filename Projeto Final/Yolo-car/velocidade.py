# velocidade.py
import cv2
import math
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from supervision.assets import VideoAssets, download_assets
from collections import defaultdict, deque

CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
MODEL_NAME = "./models/best.pt"
MODEL_RESOLUTION = 640

SOURCE = np.array([
    [520, 540],
    [830, 540],
    [1100, 630],
    [300, 630]
])

TARGET = np.array([
    [0, 0],
    [6.2, 0],
    [4.5, 29],
    [0, 29]
])

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def calculate_euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def draw_distance_line(scene, point1, point2, distance):
    point1 = tuple(np.round(point1).astype(int))
    point2 = tuple(np.round(point2).astype(int))
    cv2.line(scene, point1, point2, (0, 255, 0), 2)
    text_position = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
    cv2.putText(scene, f"{distance:.2f} m", text_position, cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 1)
    return scene

view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
model = YOLO(MODEL_NAME)
coordinates = defaultdict(lambda: deque(maxlen=30))  # Ajustar conforme necessário

def process_frame(frame, video_info, byte_track, bounding_box_annotator, label_annotator):
    result = model(frame, imgsz=MODEL_RESOLUTION, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
    detections = detections[detections.class_id != 0]
    detections = detections.with_nms(IOU_THRESHOLD)
    detections = byte_track.update_with_detections(detections=detections)

    points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_points = view_transformer.transform_points(points=points).astype(int)

    for tracker_id, [_, y] in zip(detections.tracker_id, transformed_points):
        coordinates[tracker_id].append(y)

    labels = []
    for tracker_id in detections.tracker_id:
        if len(coordinates[tracker_id]) < video_info.fps / 2:
            labels.append(f"#{tracker_id}")
        else:
            coordinate_start = coordinates[tracker_id][-1]
            coordinate_end = coordinates[tracker_id][0]
            distance = abs(coordinate_start - coordinate_end)
            time = len(coordinates[tracker_id]) / video_info.fps
            speed = distance / time * 3.6
            labels.append(f"#{tracker_id} {int(speed)} km/h")

    annotated_frame = frame.copy()
    annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    for i in range(len(transformed_points) - 1):
        for j in range(i + 1, len(transformed_points)):
            point1 = tuple(points[i])
            point2 = tuple(points[j])
            distance = calculate_euclidean_distance(transformed_points[i], transformed_points[j])
            annotated_frame = draw_distance_line(annotated_frame, point1, point2, distance)

    return annotated_frame
