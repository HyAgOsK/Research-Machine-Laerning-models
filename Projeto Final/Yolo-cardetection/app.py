import streamlit as st
import cv2
import numpy as np
import torch
import os
import time
from collections import defaultdict, deque
from ultralytics import YOLO
from PIL import Image
import glob
import wget
import math

st.set_page_config(layout="wide")

cfg_model_path = 'models/yolov8n.pt'
model = None
confidence = .25
coordinates = defaultdict(lambda: deque(maxlen=30))
object_speeds = defaultdict(float)

SOURCE_MATRIX = np.array([
    [578, 589],
    [931, 589],
    [1484, 895],
    [200, 895]
])
TARGET_WIDTH = 5.60
TARGET_HEIGHT = 30
TARGET_MATRIX = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1]
])
ALPHA = 0.5

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.M = cv2.getPerspectiveTransform(source.astype(np.float32), target.astype(np.float32))
        self.H, _ = cv2.findHomography(source.astype(np.float32), target.astype(np.float32))

    def transformPointsHomography(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0 or self.H is None:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.H)
        return transformed_points.reshape(-1, 2)

def calculate_euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def draw_info(scene, point1, point2, distance, object_id, speed, fps):
    point1 = tuple(np.round(point1).astype(int))
    point2 = tuple(np.round(point2).astype(int))
    cv2.line(scene, point1, point2, (0, 255, 0), 2)
    text_position = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
    cv2.putText(scene, f"ID: {object_id} Speed: {speed:.2f} m/s Dist: {distance:.2f} m", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
    cv2.putText(scene, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return scene

def calculate_ema(previous_ema, current_value, alpha):
    return alpha * current_value + (1 - alpha) * previous_ema

view_transformer = ViewTransformer(source=SOURCE_MATRIX, target=TARGET_MATRIX)
model = YOLO(cfg_model_path)

@st.cache_resource
def load_model(path, device):
    model_ = YOLO(path)
    model_.to(device)
    return model_

@st.cache_resource
def download_model(url):
    model_file = wget.download(url, out="models")
    return model_file

def get_user_model():
    model_src = st.sidebar.radio("Tipo", ["Enviar arquivo", "URL"])
    model_file = None
    if model_src == "Enviar arquivo":
        model_bytes = st.sidebar.file_uploader("Enviar modelo", type=['pt'])
        if model_bytes:
            model_file = "models/uploaded_" + model_bytes.name
            with open(model_file, 'wb') as out:
                out.write(model_bytes.read())
    else:
        url = st.sidebar.text_input("model url")
        if url:
            model_file_ = download_model(url)
            if model_file_.split(".")[-1] == "pt":
                model_file = model_file_
    return model_file

def main():
    global model, confidence, cfg_model_path

    st.title("Dashboard de Detecção de Velocidade e Distância")

    st.sidebar.title("Configurações")

    model_src = st.sidebar.radio("Selecione o tipo de arquivo de pesos do YOLO", ["Modelo padrão YOLOv8", "Faça upload do seu modelo"])

    if model_src == "Faça upload do seu modelo":
        user_model_path = get_user_model()
        if user_model_path:
            cfg_model_path = user_model_path
        st.sidebar.text(cfg_model_path.split("/")[-1])
        st.sidebar.markdown("---")

    if not os.path.isfile(cfg_model_path):
        st.warning("Modelo não disponível!! por favor faça o upload do seu modelo.", icon="⚠️")
    else:
        if torch.cuda.is_available():
            device_option = st.sidebar.radio("Selecione o dispositivo para inferência", ['cpu', 'cuda'], disabled=False, index=0)
        else:
            device_option = st.sidebar.radio("Selecione o dispositivo para inferência", ['cpu', 'cuda'], disabled=True, index=0)

        model = load_model(cfg_model_path, device_option)

        confidence = st.sidebar.slider('Confiança', min_value=0.1, max_value=1.0, value=.45)

        if st.sidebar.checkbox("Classes customizadas"):
            model_names = list(model.names.values())
            assigned_class = st.sidebar.multiselect("Selecione as classes", model_names, default=[model_names[0]])
            classes = [model_names.index(name) for name in assigned_class]
            model.classes = classes
        else:
            model.classes = list(model.names.keys())

        st.sidebar.markdown("---")

        input_option = st.sidebar.radio("Selecione o tipo de entrada de dados: ", ['Imagem', 'Vídeo', 'Webcam'])

        data_src = st.sidebar.radio("Adicione seu arquivo: ", ['Dado de amostra', 'Envie seu arquivo'])

        if input_option == 'Imagem':
            image_input(data_src)
        elif input_option == 'Vídeo':
            video_input(data_src)
        else:
            camera_input(0)

def image_input(data_src):
    st.markdown("### Detecção de objetos em imagens")

    img_file = None
    if data_src == 'Dado de amostra':
        img_path = glob.glob('data/sample_images/*')
        img_slider = st.slider("Escolha a imagem", min_value=1, max_value=len(img_path), step=1)
        img_file = img_path[img_slider - 1]
    else:
        img_bytes = st.sidebar.file_uploader("Fazer upload de imagem", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
            Image.open(img_bytes).save(img_file)

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Imagem selecionada")
        with col2:
            img, class_name, id = infer_image(img_file, 30, 0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img, caption="Predição da imagem")
            if class_name == '':
                st.markdown(f'### Objetos detectados:\n #### Nenhum objeto detectado')

def video_input(data_src):
    st.markdown("### Detecção de objetos em vídeo")

    vid_file = None
    if data_src == 'Dado de amostra':
        vid_file = "data/sample_videos/testdistancia.mp4"
    else:
        vid_bytes = st.sidebar.file_uploader("Fazer upload de um vídeo", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            vid_file = "data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        custom_size = st.sidebar.checkbox("Tamanho customizável")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("**Tamanho original do vídeo**")
            st.text(f"Largura: {width}px")
            st.text(f"Altura: {height}px")
        with st2:
            st.markdown("**FPS**")
            st.text(f"FPS: {fps}")
        with st3:
            st.markdown("**Tamanho do vídeo customizado**")
            st.text(f"Largura: {width}px")
            st.text(f"Altura: {height}px")

        FRAME_WINDOW = st.image([])

        stop_processing = st.checkbox('Parar a inferência')
        prev_time = 0

        while cap.isOpened() and not stop_processing:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame, class_name, id = infer_image(frame, 30, 1)
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            FRAME_WINDOW.image(frame)

        st.write("Parando a inferência")
        cap.release()

def camera_input(camera_id):
    run = st.checkbox('Rodar câmera')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(camera_id)
    prev_time = 0

    while run:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame, class_name, id = infer_image(frame, 30, 1)
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        FRAME_WINDOW.image(frame)

    cap.release()

def infer_image(image_path_or_array, max_det, vid_idx, fps=0):
    prev_ema = 0.0
    detection_scene = None
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array)
    else:
        image = image_path_or_array

    scene = image.copy()
    results = model.predict(source=scene, conf=confidence, device=model.device, classes=model.classes, save=False, save_txt=False, max_det=max_det)
    annotated_frame = results[0].plot()

    for result in results:
        detection_scene = annotated_frame
        boxes = result.boxes
        classes = result.names
        class_name = ''
        if vid_idx == 0:
            coordinates[0].clear()
        if boxes:
            for box in boxes:
                c = int(box.cls[0])
                class_name += ' ' + str(classes[c])
                p1 = tuple((box.xyxy[0][0].item(), box.xyxy[0][1].item()))
                p2 = tuple((box.xyxy[0][2].item(), box.xyxy[0][3].item()))
                center = (int(p1[0] + (p2[0] - p1[0]) / 2), int(p2[1]))
                coordinates[c].appendleft(center)
                if len(coordinates[c]) >= 2:
                    point1 = view_transformer.transformPointsHomography(np.array([coordinates[c][0]]))[0]
                    point2 = view_transformer.transformPointsHomography(np.array([coordinates[c][1]]))[0]
                    distance = calculate_euclidean_distance(point1, point2)
                    smoothed_distance = calculate_ema(prev_ema, distance, ALPHA)
                    speed = smoothed_distance / (1/30)  # Assuming 30 FPS for speed calculation
                    object_speeds[c] = speed
                    detection_scene = draw_info(detection_scene, coordinates[c][0], coordinates[c][1], smoothed_distance, c, speed, fps)
                    prev_ema = smoothed_distance

    return detection_scene, class_name, coordinates

if __name__ == "__main__":
    main()
