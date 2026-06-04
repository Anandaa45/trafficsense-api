from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

from services.congestion import analyze_congestion
from services.counting import TrafficCounter
from utils.custom_layers import FocalLoss, SpatialAttention

load_model = tf.keras.models.load_model

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

print("Memuat model AI (YOLO & CNN)...")
yolo_model = YOLO(str(MODEL_DIR / "best.pt"))
cnn_model = load_model(
    str(MODEL_DIR / "TrafficSense_CNN.keras"),
    custom_objects={"SpatialAttention": SpatialAttention, "FocalLoss": FocalLoss},
)
print("Model siap!")

kamus_kelas = {
    0: "Motorcycle",
    1: "Car",
    2: "Bus",
    3: "Truck",
}


def classify_crop(crop):
    crop_resized = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (160, 160))
    inp = np.expand_dims(crop_resized / 255.0, axis=0)
    preds = cnn_model.predict(inp, verbose=0)[0]
    kelas_id = int(np.argmax(preds))
    return kamus_kelas[kelas_id]


def process_traffic_video(video_path, custom_line_y=None):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width == 0 or height == 0:
        cap.release()
        raise ValueError("Video tidak dapat dibaca.")

    garis_y = int(custom_line_y) if custom_line_y else int(height * 0.8)
    garis_virtual = ((0, garis_y), (width, garis_y))

    counter = TrafficCounter(garis_virtual)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model.track(frame, persist=True, conf=0.35, verbose=False)[0]

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                nama_kelas = classify_crop(crop)
                titik_tengah = ((x1 + x2) // 2, (y1 + y2) // 2)

                counter.update_count(track_id, titik_tengah, nama_kelas)

    cap.release()
    congestion_data = analyze_congestion(counter.count_per_kelas)

    return {
        "total_kendaraan": counter.total_count,
        "rincian": counter.count_per_kelas,
        "kemacetan": congestion_data,
        "garis_y_dipakai": garis_y,
        "tipe_file": "video",
    }


def process_traffic_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Gambar tidak dapat dibaca.")

    height, width = image.shape[:2]
    results = yolo_model.predict(image, conf=0.35, verbose=False)[0]

    count_per_kelas = {"Motorcycle": 0, "Car": 0, "Bus": 0, "Truck": 0}
    total_count = 0

    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            nama_kelas = classify_crop(crop)
            count_per_kelas[nama_kelas] += 1
            total_count += 1

    congestion_data = analyze_congestion(count_per_kelas)

    return {
        "total_kendaraan": total_count,
        "rincian": count_per_kelas,
        "kemacetan": congestion_data,
        "garis_y_dipakai": None,
        "tipe_file": "image",
    }
