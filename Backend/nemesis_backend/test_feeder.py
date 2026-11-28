import time
import requests

API_URL = "http://127.0.0.1:8000/api/v1/ingest"

VIDEO_ID = "test_cam_03"

# Consistent synthetic embeddings
BASELINE_EMBEDDING = [1.0] * 64
ANOMALY_EMBEDDING = [-1.0] * 64

ALERT_COUNT = 0


def send_frame(payload):
    global ALERT_COUNT

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=30
        )

        response.raise_for_status()
        result = response.json()

        if result.get("alert_triggered"):
            ALERT_COUNT += 1
            print("\n🚨 ANOMALY ALERT DETECTED!")
            print("Explanation:", result.get("alert", {}).get("explanation"))

        return result

    except requests.RequestException as error:
        print("Request failed:", error)
        return None


# ==========================================
# STEP 1: NORMAL BASELINE
# ==========================================

print("\n========== STEP 1: BASELINE CALIBRATION ==========\n")

for i in range(60):

    t = round(i * 0.04, 3)

    payload = {
        "video_id": VIDEO_ID,
        "frame_id": i,
        "timestamp": t,

        "detections": [
            {
                "track_id": 1,
                "label": "person",
                "confidence": 0.94,
                "bbox": [
                    0.1 + (i * 0.001),
                    0.4,
                    0.15 + (i * 0.001),
                    0.8
                ]
            },
            {
                "track_id": 2,
                "label": "person",
                "confidence": 0.91,
                "bbox": [
                    0.7 - (i * 0.001),
                    0.4,
                    0.75 - (i * 0.001),
                    0.8
                ]
            }
        ],

        "frame_embedding": BASELINE_EMBEDDING
    }

    result = send_frame(payload)

    if result is None:
        break

    print(
        f"Frame {i:02d} | "
        f"Calibrated: {result['calibrated']} | "
        f"Score: {result['score']}"
    )

    time.sleep(0.015)


# ==========================================
# STEP 2: SUSPICIOUS ACTIVITY
# ==========================================

print("\n========== STEP 2: ANOMALY INJECTION ==========\n")

for i in range(60, 95):

    step = i - 60
    t = round(i * 0.04, 3)

    payload = {
        "video_id": VIDEO_ID,
        "frame_id": i,
        "timestamp": t,

        "detections": [
            {
                "track_id": 1,
                "label": "person",
                "confidence": 0.96,
                "bbox": [
                    0.2 + (step * 0.02),
                    0.4,
                    0.25 + (step * 0.02),
                    0.8
                ]
            },
            {
                "track_id": 2,
                "label": "person",
                "confidence": 0.93,
                "bbox": [
                    0.21 + (step * 0.02),
                    0.4,
                    0.26 + (step * 0.02),
                    0.8
                ]
            },
            {
                "track_id": 99,
                "label": "unidentified_item",
                "confidence": 0.89,
                "bbox": [
                    0.22 + (step * 0.02),
                    0.6,
                    0.24 + (step * 0.02),
                    0.7
                ]
            }
        ],

        "frame_embedding": ANOMALY_EMBEDDING
    }

    result = send_frame(payload)

    if result is None:
        break

    print(
        f"Frame {i:02d} | "
        f"Score: {result['score']} | "
        f"Alert: {result['alert_triggered']}"
    )

    time.sleep(0.02)


# ==========================================
# STEP 3: TEST SUMMARY
# ==========================================

print("\n========== TEST SUMMARY ==========\n")

print("Video ID:", VIDEO_ID)
print("Injected anomaly frames: 35")
print("Total alerts generated:", ALERT_COUNT)

if ALERT_COUNT == 1:
    print("SUCCESS: One alert generated for the event.")
elif ALERT_COUNT == 0:
    print("WARNING: No alert generated. Check anomaly scores and threshold.")
else:
    print("WARNING: Multiple alerts generated. Check event grouping.")

print("\nNEMESIS-X test completed.")
