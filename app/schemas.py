from pydantic import BaseModel, Field
from typing import List, Optional

class DetectedObject(BaseModel):
    track_id: int
    label: str
    confidence: float
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2] normalized between 0.0 and 1.0")

class FramePayload(BaseModel):
    video_id: str
    frame_id: int
    timestamp: float
    detections: List[DetectedObject]
    frame_embedding: Optional[List[float]] = None

class AnomalyAlert(BaseModel):
    event_id: str
    video_id: str
    start_time: float
    end_time: float
    anomaly_score: float
    involved_track_ids: List[int]
    explanation: str
    evidence_frames: List[int]