import time
import requests
import random

API_URL = "http://localhost:8000/api/v1/ingest"

print("--- Step 1: Feeding 60 baseline frames (Normal Scene) ---")
for i in range(60):
    t = round(i * 0.04, 3)
    payload = {
        "video_id": "test_cam_01",
        "frame_id": i,
        "timestamp": t,
        "detections": [
            {
                "track_id": 1,
                "label": "person",
                "confidence": 0.94,
                "bbox": [0.1 + (i * 0.001), 0.4, 0.15 + (i * 0.001), 0.8]
            },
            {
                "track_id": 2,
                "label": "person",
                "confidence": 0.91,
                "bbox": [0.7 - (i * 0.001), 0.4, 0.75 - (i * 0.001), 0.8]
            }
        ],
        # Standard centered baseline embedding
        "frame_embedding": [random.gauss(0.0, 0.05) for _ in range(64)]
    }
    try:
        res = requests.post(API_URL, json=payload).json()
        print(f"Frame {i:02d} | Calibrated: {res['calibrated']} | Score: {res['score']}")
    except Exception as e:
        print(f"Error on frame {i}: {e}")
        break
    time.sleep(0.015)

print("\n--- Step 2: Injecting Anomaly (Sudden sprint, convergence, orthogonal vector) ---")
for i in range(60, 95):
    step = i - 60
    t = round(i * 0.04, 3)
    payload = {
        "video_id": "test_cam_01",
        "frame_id": i,
        "timestamp": t,
        "detections": [
            # Track 1 sprints across the frame rapidly
            {
                "track_id": 1,
                "label": "person",
                "confidence": 0.96,
                "bbox": [0.2 + (step * 0.02), 0.4, 0.25 + (step * 0.02), 0.8]
            },
            # Track 2 collapses into the exact same coordinates (collision/snatch)
            {
                "track_id": 2,
                "label": "person",
                "confidence": 0.93,
                "bbox": [0.21 + (step * 0.02), 0.4, 0.26 + (step * 0.02), 0.8]
            },
            # New unexpected object appears
            {
                "track_id": 99,
                "label": "unidentified_item",
                "confidence": 0.89,
                "bbox": [0.22 + (step * 0.02), 0.6, 0.24 + (step * 0.02), 0.7]
            }
        ],
        # Opposing vector direction (negative space to trigger cosine distance)
        "frame_embedding": [random.gauss(-0.8, 0.05) for _ in range(64)]
    }
    try:
        res = requests.post(API_URL, json=payload).json()
        print(f"Frame {i:02d} | Score: {res['score']} | Alert Triggered: {res['alert_triggered']}")
        if res.get("alert_triggered") and res.get("alert"):
            print(f"  >>> ALERT: {res['alert']['explanation']}")
    except Exception as e:
        print(f"Error on frame {i}: {e}")
        break
    time.sleep(0.02)