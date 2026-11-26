from collections import deque
import numpy as np
from typing import Dict, Tuple, List
from app.schemas import FramePayload

class TemporalMemory:
    def __init__(self, window_size: int = 45):
        self.window_size = window_size
        self.frames: deque[FramePayload] = deque(maxlen=window_size)
        self.trajectories: Dict[int, deque[Tuple[float, float, float]]] = {}

    def push(self, frame: FramePayload):
        self.frames.append(frame)
        for det in frame.detections:
            cx = (det.bbox[0] + det.bbox[2]) / 2.0
            cy = (det.bbox[1] + det.bbox[3]) / 2.0
            if det.track_id not in self.trajectories:
                self.trajectories[det.track_id] = deque(maxlen=self.window_size)
            self.trajectories[det.track_id].append((frame.timestamp, cx, cy))

    def extract_interaction_vector(self) -> np.ndarray:
        if not self.frames:
            return np.zeros(4, dtype=np.float32)

        latest = self.frames[-1]
        active_ids = [d.track_id for d in latest.detections]
        obj_count = float(len(active_ids))

        min_distance = 1.0
        centers = [((d.bbox[0] + d.bbox[2]) / 2.0, (d.bbox[1] + d.bbox[3]) / 2.0) for d in latest.detections]
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = float(np.linalg.norm(np.array(centers[i]) - np.array(centers[j])))
                if dist < min_distance:
                    min_distance = dist

        velocities = []
        for tid in active_ids:
            pts = self.trajectories.get(tid, deque())
            if len(pts) >= 2:
                dt = pts[-1][0] - pts[0][0]
                if dt > 0.001:
                    disp = float(np.linalg.norm(np.array(pts[-1][1:]) - np.array(pts[0][1:])))
                    velocities.append(disp / dt)

        max_vel = float(np.max(velocities)) if velocities else 0.0
        mean_vel = float(np.mean(velocities)) if velocities else 0.0

        return np.array([obj_count, min_distance, max_vel, mean_vel], dtype=np.float32)