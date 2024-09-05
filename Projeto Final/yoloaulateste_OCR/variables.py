import numpy as np
from ultralytics import YOLO

SOURCE_VIDEO_PATH = "testetesteteste.mp4"
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
model_extension = 'pt'
MODEL_NAME = "uploaded_model.onnx"
MODEL_RESOLUTION = 640  
ALPHA = 0.5
SPEED_THRESHOLD = 5  # Speed threshold in km/h to save frames

ocr_frame_interval = 0  

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
