import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict, deque
from tqdm import tqdm
import math
import time
import matplotlib.pyplot as plt
from utils.helper import send_email
from utils.constants import *
from deepsparse import Pipeline
from deepsparse.yolo.schemas import YOLOInput
import os
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from inference_sdk import InferenceHTTPClient
from utils.constants import (SENDER_ADDRESS, PORT, SMTP_SERVER_ADDRESS, SENDER_PASSWORD)

# Configurar o modelo OCR
ocr_model = ocr_predictor(pretrained=True)

st.set_page_config(layout="wide")

SOURCE_VIDEO_PATH = "testetesteteste.mp4"
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
model_extension = 'pt'
MODEL_NAME = "yolov8n.pt"
MODEL_RESOLUTION = 1280
ALPHA = 0.5
SPEED_THRESHOLD = 1005  # Speed threshold in km/h to save frames

model_plate = YOLO('./models/license_plate_detector.pt')
distance = 0
SOURCE_MATRIX = np.array([
    [578, 589],
    [931, 589],
    [1484, 895],
    [200, 895]
])

TARGET_WIDTH = 7.60
TARGET_HEIGHT = 31

TARGET_MATRIX = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])

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

st.title("Rastreador de tráfego em instituições de ensino")
st.markdown("### Velocidade (km/h) e Distância (m)")

uploaded_model_file = st.sidebar.file_uploader("Faça o upload do modelo YOLOv8 (.pt ou .tflite)", type=["pt", "tflite", "onnx"])

if uploaded_model_file is not None:
    model_extension = uploaded_model_file.name.split('.')[-1]
    if model_extension == 'pt':
        with open("uploaded_model.pt", "wb") as f:
            f.write(uploaded_model_file.getbuffer())
        MODEL_NAME = "uploaded_model.pt"
        model = YOLO(MODEL_NAME)
    elif model_extension == 'tflite':
        MODEL_RESOLUTION = 1280
        with open("uploaded_model.tflite", "wb") as f:
            f.write(uploaded_model_file.getbuffer())
        MODEL_NAME = "uploaded_model.tflite"
        model = YOLO(MODEL_NAME)
    elif model_extension == 'onnx':
        MODEL_RESOLUTION = 224
        with open("uploaded_model.onnx", "wb") as f:
            f.write(uploaded_model_file.getbuffer())
        MODEL_NAME = "uploaded_model.onnx"
        model = Pipeline.create(task='yolov8', model_path="uploaded_model.onnx",)
else:
    model = YOLO(MODEL_NAME)

CONFIDENCE_THRESHOLD = st.sidebar.slider("Ajuste a confiança do modelo", 0.0, 1.0, 0.3)
previous_ema_dist_obj = None
source_type = st.sidebar.selectbox("Escolha a fonte do vídeo", ["vídeo padrão", "Upload de vídeo", "Webcam"])

if source_type == "vídeo padrão":
    video_source = SOURCE_VIDEO_PATH
    video_info = sv.VideoInfo.from_video_path(video_path=video_source)
    frame_generator = sv.get_video_frames_generator(source_path=video_source)
elif source_type == "Upload de vídeo":
    uploaded_file = st.sidebar.file_uploader("Faça o upload do seu vídeo", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        video_source = uploaded_file.name
        with open(video_source, "wb") as f:
            f.write(uploaded_file.getbuffer())
        video_info = sv.VideoInfo.from_video_path(video_path=video_source)
        frame_generator = sv.get_video_frames_generator(source_path=video_source)
    else:
        st.stop()
if source_type == "Webcam":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Erro ao acessar a webcam.")
        st.stop()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_info = sv.VideoInfo(fps=fps, width=width, height=height)
    frame_generator = (cap.read()[1] for _ in iter(int, 1))
else:
    video_info = sv.VideoInfo.from_video_path(video_path=video_source)
    frame_generator = sv.get_video_frames_generator(source_path=video_source)

view_transformer = ViewTransformer(source=SOURCE_MATRIX, target=TARGET_MATRIX)
previous_ema = None
byte_track = sv.ByteTrack(
    frame_rate=video_info.fps, track_activation_threshold=CONFIDENCE_THRESHOLD
)
text_scale = min(video_info.resolution_wh) * 1e-3
bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(
    text_scale=text_scale,
    text_thickness=2,
    text_position=sv.Position.BOTTOM_CENTER
)
polygon_zone = sv.PolygonZone(polygon=SOURCE_MATRIX)
coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

left_column, right_column = st.columns(2)
video_placeholder = left_column.empty()
chart_placeholder = right_column.empty()

timestamps = []
object_counts = []
speed_view_counts = []

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,2))
line, = ax1.plot([], [], label="Contagem de Objetos")
ax1.set_xlabel("Tempo (minutos)")
ax1.set_ylabel("Número de Objetos")
ax1.set_title("Fluxo de Objetos ao Longo do Tempo")
ax1.legend()

line2, = ax2.plot([], [], label="Velocidade dos Objetos (Km/h)")
ax2.set_xlabel("Tempo (minutos)")
ax2.set_ylabel("Velocidade dos Objetos")
ax2.set_title("Velocidade vs Tempo (km/h)")
ax2.legend()

chart_placeholder = st.empty()

def update_chart():
    if len(timestamps) == len(object_counts) and len(timestamps) == len(speed_view_counts):
        line.set_data(timestamps, object_counts)
        line2.set_data(timestamps, speed_view_counts)
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        chart_placeholder.pyplot(fig)
    else:
        pass
time_window = 1.0
object_count_window = deque(maxlen=int(video_info.fps * time_window))
object_count_speed = deque(maxlen=int(video_info.fps * time_window))
time_start_window = time.time()
frame_count = 0
start_time = time.time()
previous_ema_dist = None
speed_view = [] 

show_video = st.sidebar.checkbox("Mostrar vídeo", value=True)

for frame in tqdm(frame_generator, total=video_info.total_frames):
    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    if model_extension == 'onnx':
        input_data = YOLOInput(images=[frame])
        pipeline_outputs = model(images=[frame])
        detections = sv.Detections.from_deepsparse(pipeline_outputs)
    else:
        result = model(frame, imgsz=MODEL_RESOLUTION, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

    detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
    detections = detections[polygon_zone.trigger(detections)]
    detections = detections.with_nms(IOU_THRESHOLD)
    detections = byte_track.update_with_detections(detections=detections)
    points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_points = view_transformer.transformPointsHomography(points=points)

    for tracker_id, [_, y] in zip(detections.tracker_id, transformed_points):
        coordinates[tracker_id].append(y)

    labels = []
    for tracker_id in detections.tracker_id:
        if tracker_id not in coordinates or len(coordinates[tracker_id]) < video_info.fps / 2:
            labels.append(f"#{tracker_id}")
            speed_view.append(0)
        else:
            coordinate_start = coordinates[tracker_id][0]
            coordinate_end = coordinates[tracker_id][-1]
            distance_obj = abs(coordinate_end - coordinate_start)
            time_interval = len(coordinates[tracker_id]) / video_info.fps

            if previous_ema_dist is None:
                previous_ema_dist = distance_obj
                ema_speed_d = distance_obj
            else:
                ema_speed_d = calculate_ema(previous_ema_dist, distance_obj, 1)
                previous_ema_dist = ema_speed_d

            speed = previous_ema_dist / time_interval * 3.6
            
            if previous_ema is None:
                previous_ema = speed
                ema_speed = speed
                speed_view.append(0)
            else:
                ema_speed = calculate_ema(previous_ema, speed, ALPHA)
                previous_ema = ema_speed
                speed_view.append(speed)
                for tracker_id, bbox in zip(detections.tracker_id, detections.xyxy):
                    if previous_ema >= SPEED_THRESHOLD:
                        x1, y1, x2, y2 = map(int, bbox)
                        vehicle_crop = frame[y1:y2, x1:x2]
                        
                        image_path = f"id_{tracker_id}.jpg"
                        cv2.imwrite(image_path, vehicle_crop)
                        
      
                        single_img_doc = DocumentFile.from_images(image_path)
                        ocr_result = ocr_model(single_img_doc)


                        placa = ''
                        for i, block in enumerate(ocr_result.pages[0].blocks):
                            for j, line in enumerate(block.lines):
                                for k, word in enumerate(line.words):
                                    placa += word.value

                        send_email(
                            sender=SENDER_ADDRESS,
                            password=SENDER_PASSWORD,
                            receiver="hyago.silva@mtel.inatel.br",
                            smtp_server=SMTP_SERVER_ADDRESS,
                            smtp_port=PORT,
                            email_message=f"""Prezado,
                                            Este e-mail é uma notificação!

                                            Um veículo foi detectado a {int(speed)} km/h
                                            
                                            A distância desse veículo para os demais objetos ao redor é de {float(distance)}
                                            A placa do carro é provavelmente:
                                            
                                            {placa}.
                                            """,
                            subject="Excesso de velocidade",
                            attachment_path=image_path
                        )
                    placa = ''
                    if previous_ema_dist <= 0.9 and time_interval <= 1:
                        labels.append(f"#{tracker_id} {0} km/h")
                        speed_view.append(0)
                    else:
                        labels.append(f"#{tracker_id} {int(ema_speed)} km/h")
                        speed_view.append(ema_speed)

    num_detections = len(detections)
    num_labels = len(labels)
    if num_detections != num_labels:
        continue

    annotated_frame = frame.copy()
    annotated_frame = bounding_box_annotator.annotate(
        scene=annotated_frame, detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )

    for i in range(len(transformed_points) - 1):
        for j in range(i + 1, len(transformed_points)):
            point1 = tuple(points[i])
            point2 = tuple(points[j])
            distance = calculate_euclidean_distance(transformed_points[i], transformed_points[j])
            if previous_ema_dist_obj is None:
                previous_ema_dist_obj = distance
                ema_speed_dist_obj = distance
            else:
                ema_speed_dist_obj = calculate_ema(previous_ema_dist_obj, distance, ALPHA)
                previous_ema_dist_obj = ema_speed_dist_obj

            annotated_frame = draw_distance_line(annotated_frame, point1, point2, previous_ema_dist_obj)

    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if show_video:
        video_placeholder.image(annotated_frame, channels="BGR")

    object_count_window.append(len(detections))
    current_time = time.time()
    if current_time - time_start_window >= time_window:
        avg_object_count = np.mean(object_count_window)
        timestamps.append((current_time - start_time) / 60)
        object_counts.append(avg_object_count)

        print('view speed',  speed_view[-1])
        speed_view_counts.append(speed_view[-1])
        update_chart()
        time_start_window = current_time
        object_count_window.clear()

st.write("Processamento de vídeo concluído!")
