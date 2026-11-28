from ultralytics import YOLO
from pathlib import Path
import cv2
import json

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

VIDEO_DIR = PROJECT_DIR / "sample_videos"
OUTPUT_DIR = PROJECT_DIR / "output"

TRACKED_DIR = OUTPUT_DIR / "tracked_videos"
DATA_DIR = OUTPUT_DIR / "tracking_data"

TRACKED_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

FORMATS = {".mp4", ".avi", ".mov", ".mkv"}

videos = [
    f for f in VIDEO_DIR.rglob("*")
    if f.is_file() and f.suffix.lower() in FORMATS
]

for video in videos:
    print(f"\nTracking: {video.name}")

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    cap.release()

    model = YOLO("yolo11n.pt")

    results = model.track(
        source=str(video),
        tracker="bytetrack.yaml",
        persist=True,
        device="cpu",
        imgsz=640,
        conf=0.35,
        vid_stride=2,
        save=True,
        project=str(TRACKED_DIR),
        name=video.stem,
        exist_ok=True,
        stream=True,
        verbose=False
    )

    records = []
    unique_ids = set()
    frame_index = 0

    for result in results:
        frame_index += 1

        if result.boxes is None or result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        track_ids = result.boxes.id.int().cpu().tolist()

        timestamp = (frame_index - 1) * 2 / fps

        for box, cls, conf, track_id in zip(
            boxes, classes, confidences, track_ids
        ):
            x1, y1, x2, y2 = box
            class_name = model.names[int(cls)]

            unique_ids.add(track_id)

            records.append({
                "frame": frame_index,
                "timestamp": round(timestamp, 2),
                "track_id": track_id,
                "object": class_name,
                "confidence": round(conf, 3),
                "bbox": [round(v, 2) for v in box],
                "centroid": [
                    round((x1 + x2) / 2, 2),
                    round((y1 + y2) / 2, 2)
                ]
            })

    json_path = DATA_DIR / f"{video.stem}.json"

    with open(json_path, "w") as file:
        json.dump(records, file, indent=4)

    print("Processed frames:", frame_index)
    print("Unique track IDs:", len(unique_ids))
    print("Tracking data saved:", json_path)

print("\nNEMESIS-X Tracking Completed!")