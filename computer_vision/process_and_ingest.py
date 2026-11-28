import sys
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

from video_processor import process_video


def send_frame(api_url, payload):
    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        api_url + "/api/v1/ingest",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main():

    video_path = Path(sys.argv[1])
    api_url = sys.argv[2].rstrip("/")
    video_id = sys.argv[3]

    print("Starting NEMESIS-X video analysis...")

    result = process_video(video_path, video_id)

    frames = defaultdict(list)
    timestamps = {}

    for record in result["records"]:
        frame_id = record["frame"]
        frames[frame_id].append(record)
        timestamps.setdefault(frame_id, record["timestamp"])

    alerts = []
    sent = 0

    print("Sending detections to reasoning engine...")

    for frame_id in sorted(frames):

        detections = []

        for record in frames[frame_id]:

            x1, y1, x2, y2 = record["bbox"]

            detections.append({
                "track_id": record["track_id"],
                "label": record["object"],
                "confidence": record["confidence"],
                "bbox": [
                    x1 / result["width"],
                    y1 / result["height"],
                    x2 / result["width"],
                    y2 / result["height"]
                ]
            })

        payload = {
            "video_id": video_id,
            "frame_id": frame_id,
            "timestamp": timestamps[frame_id],
            "detections": detections,
            "frame_embedding": None
        }

        response = send_frame(api_url, payload)
        sent += 1

        if response.get("alert"):
            alerts.append(response["alert"])

        if sent % 100 == 0:
            print(f"Frames transmitted: {sent}")

    summary = {
        "video_id": video_id,
        "processed_frames": result["processed_frames"],
        "detection_records": result["detection_records"],
        "unique_tracks": result["unique_tracks"],
        "frames_sent": sent,
        "alerts": alerts
    }

    print("\nNEMESIS_X_RESULT=" + json.dumps(summary))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("PROCESSING_ERROR:", str(error))
        sys.exit(1)