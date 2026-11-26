import uuid
from typing import List
from app.schemas import FramePayload, AnomalyAlert

class EvidenceReasoner:
    @staticmethod
    def generate_alert(window: List[FramePayload], score: float, video_id: str) -> AnomalyAlert:
        start_frame = window[0]
        end_frame = window[-1]

        active_tracks = set()
        labels_seen = set()
        for f in window:
            for d in f.detections:
                active_tracks.add(d.track_id)
                labels_seen.add(d.label)

        entities_str = ", ".join(labels_seen) if labels_seen else "Unclassified entities"
        explanation = (
            f"Anomalous pattern identified (Score: {score:.2f}). "
            f"Active entities: [{entities_str}] linked to track IDs {sorted(list(active_tracks))}. "
            f"Temporal window: {start_frame.timestamp:.1f}s to {end_frame.timestamp:.1f}s. "
            f"Observed sudden kinematic deviation or spatial divergence from baseline."
        )

        evidence_frames = [f.frame_id for f in window[::max(1, len(window) // 4)]]

        return AnomalyAlert(
            event_id=f"evt_{str(uuid.uuid4())[:8]}",
            video_id=video_id,
            start_time=start_frame.timestamp,
            end_time=end_frame.timestamp,
            anomaly_score=score,
            involved_track_ids=list(active_tracks),
            explanation=explanation,
            evidence_frames=evidence_frames
        )