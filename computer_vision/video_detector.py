from ultralytics import YOLO
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

VIDEO_DIR = PROJECT_DIR / "sample_videos"
OUTPUT_DIR = PROJECT_DIR / "output"

# Supported formats
FORMATS = {".mp4", ".avi", ".mov", ".mkv"}

# Load pretrained model
print("Loading NEMESIS-X Detection Engine...")
model = YOLO("yolo11n.pt")

# Find all videos
videos = [
    file for file in VIDEO_DIR.rglob("*")
    if file.is_file() and file.suffix.lower() in FORMATS
]

if not videos:
    print("No videos found in sample_videos!")
    exit()

print(f"Found {len(videos)} videos.")

# Process each video
for video in videos:

    print(f"\nProcessing: {video.name}")

    results = model.predict(
        source=str(video),
        device="cpu",
        imgsz=640,
        conf=0.35,
        vid_stride=2,
        save=True,
        project=str(OUTPUT_DIR),
        name=video.parent.name,
        exist_ok=True,
        stream=True,
        verbose=False
    )

    frame_count = 0
    object_counts = {}

    for result in results:

        frame_count += 1

        for box in result.boxes:

            class_id = int(box.cls[0])
            name = model.names[class_id]

            object_counts[name] = (
                object_counts.get(name, 0) + 1
            )

    print("Frames processed:", frame_count)
    print("Detected objects:", object_counts)

print("\nNEMESIS-X Detection Completed!")
print("Check the output folder.")
