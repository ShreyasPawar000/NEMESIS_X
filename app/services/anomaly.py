import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Optional, List

class AnomalyEngine:
    def __init__(self, warmup_frames: int = 50, contamination: float = 0.05):
        self.warmup_frames = warmup_frames
        self.detector = IsolationForest(contamination=contamination, random_state=42)
        self.warmup_samples: List[np.ndarray] = []
        self.is_calibrated = False
        self.baseline_embedding: Optional[np.ndarray] = None
        self.embedding_count = 0

    def score(self, kinematic_feats: np.ndarray, frame_embedding: Optional[List[float]] = None) -> float:
        # Calibration phase: learn the normal scene kinematics
        if not self.is_calibrated:
            self.warmup_samples.append(kinematic_feats)
            if frame_embedding is not None:
                emb = np.array(frame_embedding, dtype=np.float32)
                if self.baseline_embedding is None:
                    self.baseline_embedding = emb
                else:
                    self.baseline_embedding += emb
                self.embedding_count += 1

            if len(self.warmup_samples) >= self.warmup_frames:
                self.detector.fit(np.array(self.warmup_samples))
                if self.baseline_embedding is not None and self.embedding_count > 0:
                    self.baseline_embedding /= self.embedding_count
                self.is_calibrated = True
            return 0.0

        # Kinematic outlier score mapped to [0, 1]
        raw_score = self.detector.decision_function([kinematic_feats])[0]
        kinematic_score = float(np.clip(0.5 - (raw_score * 0.5), 0.0, 1.0))

        # Semantic cosine drift score
        semantic_score = 0.0
        if frame_embedding is not None and self.baseline_embedding is not None:
            emb = np.array(frame_embedding, dtype=np.float32)
            denom = (np.linalg.norm(emb) * np.linalg.norm(self.baseline_embedding)) + 1e-7
            cos_sim = float(np.dot(emb, self.baseline_embedding) / denom)
            semantic_score = float(np.clip(1.0 - cos_sim, 0.0, 1.0))

        final_score = (0.6 * kinematic_score) + (0.4 * semantic_score)
        return round(float(final_score), 3)