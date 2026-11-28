from collections import defaultdict
import numpy as np


def extract_evidence(alert, frames):

    involved_ids = set(alert.involved_track_ids)
    tracks = defaultdict(list)

    for frame in frames:
        for detection in frame.detections:

            if detection.track_id not in involved_ids:
                continue

            x1, y1, x2, y2 = detection.bbox

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            tracks[detection.track_id].append({
                "timestamp": frame.timestamp,
                "frame_id": frame.frame_id,
                "label": detection.label,
                "confidence": detection.confidence,
                "center": [center_x, center_y]
            })

    evidence_tracks = []

    for track_id, observations in tracks.items():

        observations.sort(key=lambda x: x["timestamp"])

        first = observations[0]
        last = observations[-1]

        displacement = float(
            np.linalg.norm(
                np.array(last["center"]) -
                np.array(first["center"])
            )
        )

        duration = last["timestamp"] - first["timestamp"]

        speed = displacement / duration if duration > 0 else 0.0

        evidence_tracks.append({
            "track_id": track_id,
            "label": first["label"],
            "observations": len(observations),
            "first_seen": first["timestamp"],
            "last_seen": last["timestamp"],
            "normalized_displacement": round(displacement, 4),
            "normalized_speed": round(speed, 4),
            "evidence_frames": [
                first["frame_id"],
                last["frame_id"]
            ]
        })

    return {
        "video_id": alert.video_id,
        "anomaly_score": alert.anomaly_score,
        "anomaly_interval": {
            "start": alert.start_time,
            "end": alert.end_time
        },
        "tracks": evidence_tracks
    }