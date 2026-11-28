import json
import re
import cv2

from pathlib import Path
from ultralytics import YOLO

# Project paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

OUTPUT_DIR = PROJECT_DIR / "output"
TRACKED_DIR = OUTPUT_DIR / "tracked_videos"
DATA_DIR = OUTPUT_DIR / "tracking_data"

MODEL_PATH = BASE_DIR / "yolo11n.pt"

# Processing configuration
CONFIDENCE = 0.35
IMAGE_SIZE = 640
FRAME_STRIDE = 2
DEVICE = "cpu"

FORMATS = {".mp4", ".avi", ".mov", ".mkv"}

# Create output directories
TRACKED_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Reuse the model instead of loading it for every video
_model = None


def get_model():
    global _model

    if _model is None:
        print("Loading NEMESIS-X Detection Engine...")
        _model = YOLO(str(MODEL_PATH))

    return _model


def process_video(video_path, video_id=None):

    video_path = Path(video_path).resolve()

    # Validate video
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if video_path.suffix.lower() not in FORMATS:
        raise ValueError("Unsupported video format.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

    video_id = video_id or video_path.stem

    # Make video ID safe for filenames
    video_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(video_id))

    print(f"\nProcessing video: {video_path.name}")

    # Read video metadata
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Unable to open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("Invalid video properties.")

    model = get_model()

    # Run YOLO + ByteTrack
    results = model.track(
        source=str(video_path),
        tracker="bytetrack.yaml",
        persist=True,
        device=DEVICE,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        vid_stride=FRAME_STRIDE,
        save=True,
        project=str(TRACKED_DIR),
        name=video_id,
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

        timestamp = (frame_index - 1) * FRAME_STRIDE / fps

        for box, cls, conf, track_id in zip(
            boxes, classes, confidences, track_ids
        ):

            x1, y1, x2, y2 = box

            unique_ids.add(track_id)

            records.append({
                "video_id": video_id,
                "frame": frame_index,
                "timestamp": round(timestamp, 3),
                "track_id": track_id,
                "object": model.names[int(cls)],
                "confidence": round(conf, 3),
                "bbox": [round(v, 2) for v in box],
                "centroid": [
                    round((x1 + x2) / 2, 2),
                    round((y1 + y2) / 2, 2)
                ]
            })

    # Save tracking records
    json_path = DATA_DIR / f"{video_id}.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=4)

    # Locate the generated annotated video
    annotated_dir = TRACKED_DIR / video_id

    tracked_video_path = None

    if annotated_dir.exists():
        video_files = [
        file for file in annotated_dir.iterdir()
        if file.suffix.lower() in FORMATS
    ]

    if video_files:
        tracked_video_path = str(
            max(video_files, key=lambda file: file.stat().st_mtime)
        )

    print("\nProcessing completed!")
    print("Video ID:", video_id)
    print("Original resolution:", f"{width}x{height}")
    print("FPS:", round(fps, 2))
    print("Total source frames:", total_frames)
    print("Sampled frames processed:", frame_index)
    print("Detection records:", len(records))
    print("Unique track IDs:", len(unique_ids))
    print("Tracking JSON:", json_path)

    return {
        "video_id": video_id,
        "video_name": video_path.name,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "processed_frames": frame_index,
        "detection_records": len(records),
        "unique_tracks": len(unique_ids),
        "tracking_json": str(json_path),
        "annotated_video_directory": str(TRACKED_DIR / video_id),
        "tracked_video_path": tracked_video_path,
        "records": records
    }


if __name__ == "__main__":

    SAMPLE_VIDEO = PROJECT_DIR / "sample_videos" / "Traffic.mp4"

    result = process_video(SAMPLE_VIDEO, "Traffic_Test")

    print("\nNEMESIS-X Video Processor Ready!")