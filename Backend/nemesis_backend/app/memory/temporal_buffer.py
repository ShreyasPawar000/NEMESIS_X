from collections import deque
import numpy as np
from typing import Dict, Tuple
from app.schemas import FramePayload


class TemporalMemory:
    def __init__(self, window_size: int = 45):
        self.window_size = window_size
        self.frames: deque[FramePayload] = deque(maxlen=window_size)
        self.trajectories: Dict[int, deque[Tuple[float, float, float]]] = {}

    def push(self, frame: FramePayload):
        self.frames.append(frame)

        active_ids = set()

        for det in frame.detections:
            tid = det.track_id
            active_ids.add(tid)

            cx = (det.bbox[0] + det.bbox[2]) / 2.0
            cy = (det.bbox[1] + det.bbox[3]) / 2.0

            if tid not in self.trajectories:
                self.trajectories[tid] = deque(maxlen=self.window_size)

            self.trajectories[tid].append(
                (frame.timestamp, cx, cy)
            )

        # Remove tracks that have disappeared for too long
        for tid in list(self.trajectories):
            if tid not in active_ids:
                if len(self.trajectories[tid]) == 0:
                    del self.trajectories[tid]

    def extract_interaction_vector(self) -> np.ndarray:
        if not self.frames:
            return np.zeros(10, dtype=np.float32)

        latest = self.frames[-1]
        detections = latest.detections

        # 1. Object count
        obj_count = float(len(detections))

        # 2. Change in object count
        previous_count = (
            len(self.frames[-2].detections)
            if len(self.frames) >= 2 else obj_count
        )
        count_change = abs(obj_count - previous_count)

        # 3 & 4. Spatial relationships
        centers = [
            (
                (d.bbox[0] + d.bbox[2]) / 2.0,
                (d.bbox[1] + d.bbox[3]) / 2.0
            )
            for d in detections
        ]

        distances = []

        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.linalg.norm(
                    np.array(centers[i]) - np.array(centers[j])
                )
                distances.append(float(dist))

        min_distance = min(distances) if distances else 1.0
        mean_distance = float(np.mean(distances)) if distances else 1.0

        # 5–8. Motion, acceleration and direction
        velocities = []
        accelerations = []
        direction_changes = []
        stationary_count = 0

        for det in detections:
            tid = det.track_id
            pts = list(self.trajectories.get(tid, []))

            if len(pts) < 2:
                continue

            # Calculate recent velocities
            movements = []

            for i in range(1, len(pts)):
                dt = pts[i][0] - pts[i - 1][0]

                if dt <= 0.001:
                    continue

                dx = pts[i][1] - pts[i - 1][1]
                dy = pts[i][2] - pts[i - 1][2]

                speed = np.sqrt(dx ** 2 + dy ** 2) / dt

                movements.append((speed, dx, dy, dt))

            if not movements:
                continue

            latest_speed = movements[-1][0]
            velocities.append(latest_speed)

            # Stationary object detection
            if latest_speed < 0.01:
                stationary_count += 1

            # Acceleration
            if len(movements) >= 2:
                previous = movements[-2]
                current = movements[-1]

                avg_dt = (previous[3] + current[3]) / 2.0

                if avg_dt > 0.001:
                    acceleration = abs(
                        current[0] - previous[0]
                    ) / avg_dt

                    accelerations.append(acceleration)

                # Direction change
                v1 = np.array([previous[1], previous[2]])
                v2 = np.array([current[1], current[2]])

                denom = np.linalg.norm(v1) * np.linalg.norm(v2)

                if denom > 1e-7:
                    cosine = np.clip(
                        np.dot(v1, v2) / denom, -1.0, 1.0
                    )

                    angle = np.arccos(cosine)
                    direction_changes.append(float(angle))

        # Aggregate motion features
        max_velocity = max(velocities) if velocities else 0.0
        mean_velocity = float(np.mean(velocities)) if velocities else 0.0

        max_acceleration = max(accelerations) if accelerations else 0.0
        mean_acceleration = (
            float(np.mean(accelerations)) if accelerations else 0.0
        )

        mean_direction_change = (
            float(np.mean(direction_changes))
            if direction_changes else 0.0
        )

        stationary_ratio = (
            stationary_count / len(velocities)
            if velocities else 0.0
        )

        # Final 10-dimensional feature vector
        return np.array([
            obj_count,
            count_change,
            min_distance,
            mean_distance,
            max_velocity,
            mean_velocity,
            max_acceleration,
            mean_acceleration,
            mean_direction_change,
            stationary_ratio
        ], dtype=np.float32)
