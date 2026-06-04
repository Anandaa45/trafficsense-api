import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from services.congestion import analyze_congestion

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
NIGHT_CONFIDENCE = float(os.getenv("NIGHT_CONFIDENCE", "0.18"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.50"))
YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "960"))
YOLO_AUGMENT = os.getenv("YOLO_AUGMENT", "false").lower() == "true"
MIN_BOX_AREA_RATIO = float(os.getenv("MIN_BOX_AREA_RATIO", "0.00012"))
VIDEO_SAMPLE_LIMIT = int(os.getenv("VIDEO_SAMPLE_LIMIT", "20"))
VIDEO_TILE_SAMPLE_LIMIT = int(os.getenv("VIDEO_TILE_SAMPLE_LIMIT", "6"))
TILE_DETECTION = os.getenv("TILE_DETECTION", "true").lower() == "true"
TILE_SIZE = int(os.getenv("TILE_SIZE", "640"))
TILE_OVERLAP = float(os.getenv("TILE_OVERLAP", "0.25"))
TILE_MIN_SIDE = int(os.getenv("TILE_MIN_SIDE", "720"))
MAX_TILES = int(os.getenv("MAX_TILES", "12"))
MERGE_IOU = float(os.getenv("MERGE_IOU", "0.45"))
VIDEO_COUNT_PERCENTILE = int(os.getenv("VIDEO_COUNT_PERCENTILE", "85"))
NIGHT_ENHANCEMENT = os.getenv("NIGHT_ENHANCEMENT", "true").lower() == "true"
LOW_LIGHT_LUMA = float(os.getenv("LOW_LIGHT_LUMA", "105"))

VEHICLE_CLASSES = ("Motorcycle", "Car", "Bus", "Truck")
DEFAULT_CLASS_MAP = {
    0: "Motorcycle",
    1: "Car",
    2: "Bus",
    3: "Truck",
}

print("Memuat model AI YOLO...")
yolo_model = YOLO(str(MODEL_DIR / "best.pt"))
print("Model YOLO siap!")


def normalize_class_name(class_id):
    model_name = str(getattr(yolo_model, "names", {}).get(int(class_id), "")).lower()

    if any(word in model_name for word in ("motorcycle", "motorbike", "motor", "sepeda motor")):
        return "Motorcycle"
    if any(word in model_name for word in ("car", "mobil", "vehicle")):
        return "Car"
    if "bus" in model_name:
        return "Bus"
    if any(word in model_name for word in ("truck", "truk", "lorry")):
        return "Truck"

    return DEFAULT_CLASS_MAP.get(int(class_id))


def empty_counts():
    return {name: 0 for name in VEHICLE_CLASSES}


def luma_stats(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)), float(np.median(gray))


def is_low_light(frame):
    mean_luma, median_luma = luma_stats(frame)
    return mean_luma < LOW_LIGHT_LUMA or median_luma < LOW_LIGHT_LUMA + 12


def enhance_night_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.18, beta=16)

    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    return cv2.addWeighted(enhanced, 1.35, blur, -0.35, 0)


def percentile_counts(frame_counts, percentile=75):
    if not frame_counts:
        return empty_counts()

    result = {}
    for vehicle_class in VEHICLE_CLASSES:
        values = [item.get(vehicle_class, 0) for item in frame_counts]
        result[vehicle_class] = int(round(np.percentile(values, percentile)))

    return result


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union else 0


def merge_detections(detections):
    merged = []

    for class_name in VEHICLE_CLASSES:
        same_class = [item for item in detections if item["class"] == class_name]
        same_class.sort(key=lambda item: item["score"], reverse=True)

        while same_class:
            current = same_class.pop(0)
            merged.append(current)
            same_class = [
                item for item in same_class
                if calculate_iou(current["xyxy"], item["xyxy"]) < MERGE_IOU
            ]

    return merged


def tile_positions(length, tile_size, overlap):
    if length <= tile_size:
        return [0]

    stride = max(1, int(tile_size * (1 - overlap)))
    positions = list(range(0, max(1, length - tile_size + 1), stride))
    last = max(0, length - tile_size)

    if positions[-1] != last:
        positions.append(last)

    return positions


def generate_tiles(width, height):
    tile_size = min(TILE_SIZE, max(width, height))
    x_positions = tile_positions(width, tile_size, TILE_OVERLAP)
    y_positions = tile_positions(height, tile_size, TILE_OVERLAP)
    tiles = []

    for y in y_positions:
        for x in x_positions:
            x2 = min(width, x + tile_size)
            y2 = min(height, y + tile_size)
            tiles.append((x, y, x2, y2))

    if len(tiles) <= MAX_TILES:
        return tiles

    step = max(1, int(np.ceil(len(tiles) / MAX_TILES)))
    return tiles[::step][:MAX_TILES]


def predict_detections(frame, offset_x=0, offset_y=0, confidence=None):
    height, width = frame.shape[:2]
    detections = []

    result = yolo_model.predict(
        frame,
        conf=confidence if confidence is not None else YOLO_CONFIDENCE,
        iou=YOLO_IOU,
        imgsz=YOLO_IMAGE_SIZE,
        augment=YOLO_AUGMENT,
        verbose=False,
    )[0]

    if result.boxes is None:
        return detections

    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()

    for box, class_id, score in zip(boxes, class_ids, scores):
        class_name = normalize_class_name(class_id)
        if class_name not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = map(float, box)
        box_area_ratio = max(0, (x2 - x1) * (y2 - y1)) / max(1, width * height)

        # Tiny boxes can be real on CCTV, but below this ratio they are usually noise.
        if box_area_ratio < MIN_BOX_AREA_RATIO:
            continue

        detections.append({
            "class": class_name,
            "score": float(score),
            "xyxy": [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y],
        })

    return detections


def detect_frame(frame, use_tiles=True):
    height, width = frame.shape[:2]
    counts = empty_counts()
    low_light = NIGHT_ENHANCEMENT and is_low_light(frame)
    detections = predict_detections(frame)

    if low_light:
        enhanced_frame = enhance_night_frame(frame)
        detections.extend(predict_detections(enhanced_frame, confidence=NIGHT_CONFIDENCE))
    else:
        enhanced_frame = None

    if TILE_DETECTION and use_tiles and max(width, height) >= TILE_MIN_SIDE:
        for x1, y1, x2, y2 in generate_tiles(width, height):
            tile = frame[y1:y2, x1:x2]
            if tile.size == 0:
                continue

            detections.extend(predict_detections(tile, offset_x=x1, offset_y=y1))

            if low_light and enhanced_frame is not None:
                enhanced_tile = enhanced_frame[y1:y2, x1:x2]
                detections.extend(
                    predict_detections(
                        enhanced_tile,
                        offset_x=x1,
                        offset_y=y1,
                        confidence=NIGHT_CONFIDENCE,
                    )
                )

    detections = merge_detections(detections)
    confidences = [item["score"] for item in detections]
    normalized_detections = []

    for item in detections:
        class_name = item["class"]
        counts[class_name] += 1
        x1, y1, x2, y2 = item["xyxy"]
        normalized_detections.append({
            "class": class_name,
            "confidence": round(item["score"] * 100, 2),
            "box": [
                round(max(0, x1) / width, 4),
                round(max(0, y1) / height, 4),
                round(min(width, x2) / width, 4),
                round(min(height, y2) / height, 4),
            ],
        })

    return counts, confidences, normalized_detections


def build_response(counts, confidences, file_type, sampled_frames=1, detections=None):
    total_count = int(sum(counts.values()))
    congestion_data = analyze_congestion(counts)
    avg_confidence = round(float(np.mean(confidences) * 100), 1) if confidences else 0.0
    min_confidence = round(float(np.min(confidences) * 100), 1) if confidences else 0.0
    max_confidence = round(float(np.max(confidences) * 100), 1) if confidences else 0.0

    return {
        "total_kendaraan": total_count,
        "rincian": counts,
        "kemacetan": congestion_data,
        "garis_y_dipakai": None,
        "tipe_file": file_type,
        "confidence": avg_confidence,
        "avg_confidence": avg_confidence,
        "min_confidence": min_confidence,
        "max_confidence": max_confidence,
        "confidence_threshold": round(YOLO_CONFIDENCE * 100, 1),
        "sampled_frames": sampled_frames,
        "detection_count": len(confidences),
        "detections": detections or [],
        "model_mode": "yolo_vehicle_detection",
    }


def process_traffic_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Gambar tidak dapat dibaca.")

    counts, confidences, detections = detect_frame(image)
    return build_response(
        counts=counts,
        confidences=confidences,
        file_type="image",
        sampled_frames=1,
        detections=detections,
    )


def process_traffic_video(video_path, custom_line_y=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width == 0 or height == 0:
        cap.release()
        raise ValueError("Video tidak dapat dibaca.")

    frame_counts = []
    all_confidences = []
    best_detections = []
    best_total = -1
    sampled_frames = 0

    if total_frames > 0:
        sample_count = min(total_frames, VIDEO_SAMPLE_LIMIT)
        sample_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
    else:
        sample_indices = []

    for frame_index in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ret, frame = cap.read()
        if not ret:
            continue

        use_tiles = sampled_frames < VIDEO_TILE_SAMPLE_LIMIT
        counts, confidences, detections = detect_frame(frame, use_tiles=use_tiles)
        frame_counts.append(counts)
        all_confidences.extend(confidences)
        sampled_frames += 1

        total = sum(counts.values())
        if total > best_total:
            best_total = total
            best_detections = detections

    if sampled_frames == 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_index = 0

        while cap.isOpened() and sampled_frames < VIDEO_SAMPLE_LIMIT:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % 25 == 0:
                use_tiles = sampled_frames < VIDEO_TILE_SAMPLE_LIMIT
                counts, confidences, detections = detect_frame(frame, use_tiles=use_tiles)
                frame_counts.append(counts)
                all_confidences.extend(confidences)
                sampled_frames += 1

                total = sum(counts.values())
                if total > best_total:
                    best_total = total
                    best_detections = detections

            frame_index += 1

    cap.release()

    if sampled_frames == 0:
        raise ValueError("Tidak ada frame video yang bisa dianalisis.")

    counts = percentile_counts(frame_counts, percentile=VIDEO_COUNT_PERCENTILE)
    return build_response(
        counts=counts,
        confidences=all_confidences,
        file_type="video",
        sampled_frames=sampled_frames,
        detections=best_detections,
    )
