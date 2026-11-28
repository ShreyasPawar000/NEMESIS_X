import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Optional, List


class AnomalyEngine:
    def __init__(self, warmup_frames: int = 50, contamination: float = 0.05):
        self.warmup_frames = warmup_frames
        self.detector = IsolationForest(
            contamination=contamination,
            random_state=42
        )

        # Store ALL warmup samples (with their scores) for potential re-training
        self.all_warmup_samples: List[np.ndarray] = []
        self.all_warmup_scores: List[float] = []

        # The filtered samples used for training
        self.warmup_samples: List[np.ndarray] = []

        self.is_calibrated = False
        self.is_recalibrated = False

        self.baseline_embedding: Optional[np.ndarray] = None
        self.embedding_count = 0

        self.expected_features = 10

        # Track warmup statistics for early anomaly detection
        self._warmup_detector: Optional[IsolationForest] = None

        # Flag to indicate if anomaly was detected during warmup
        self.anomaly_during_warmup = False
        self.warmup_anomaly_threshold = 0.5  # Threshold to flag warmup frames as anomalous
        self.anomalous_frame_indices: List[int] = []  # Indices of anomalous frames in all_warmup_samples

    def _maybe_recalibrate(self):
        """After warmup completes, if anomalies were detected, re-train on clean samples only"""
        if not self.anomaly_during_warmup or self.is_recalibrated:
            return

        print(f"[RECALIBRATE] Anomalies detected during warmup, re-training on clean samples...")

        # Filter out anomalous frames
        clean_samples = [
            self.all_warmup_samples[i]
            for i in range(len(self.all_warmup_samples))
            if i not in self.anomalous_frame_indices
        ]

        if len(clean_samples) >= 10:
            self.warmup_samples = clean_samples
            self.detector.fit(np.asarray(self.warmup_samples, dtype=np.float32))

            if self.embedding_count > 0:
                self.baseline_embedding /= self.embedding_count

            self.is_calibrated = True
            self.is_recalibrated = True
            print(f"[RECALIBRATE] Model re-calibrated with {len(self.warmup_samples)} clean samples (excluded {len(self.anomalous_frame_indices)} anomalous frames)")
        else:
            # Not enough clean samples, fall back to all samples
            print(f"[RECALIBRATE WARNING] Only {len(clean_samples)} clean samples, using all {len(self.all_warmup_samples)} samples")
            self.warmup_samples = self.all_warmup_samples
            self.detector.fit(np.asarray(self.warmup_samples, dtype=np.float32))
            if self.embedding_count > 0:
                self.baseline_embedding /= self.embedding_count
            self.is_calibrated = True

    def score(
        self,
        kinematic_feats: np.ndarray,
        frame_embedding: Optional[List[float]] = None
    ) -> float:

        features = np.asarray(kinematic_feats, dtype=np.float32).flatten()

        if features.size != self.expected_features:
            raise ValueError(
                f"Expected {self.expected_features} features, "
                f"received {features.size}"
            )

        if not np.all(np.isfinite(features)):
            raise ValueError("Kinematic features contain invalid values")

        # Learn normal scene behaviour
        if not self.is_calibrated:
            # Always add to all samples for potential re-training
            self.all_warmup_samples.append(features.copy())

            # Early anomaly detection during warmup using incremental scoring
            warmup_score = 0.0
            if len(self.all_warmup_samples) >= 10 and self._warmup_detector is None:
                # Create a temporary detector for early anomaly detection
                self._warmup_detector = IsolationForest(
                    contamination=0.1,
                    random_state=42
                )

            if self._warmup_detector is not None and len(self.all_warmup_samples) >= 10:
                try:
                    self._warmup_detector.fit(np.asarray(self.all_warmup_samples, dtype=np.float32))
                    raw_score = self._warmup_detector.decision_function([features])[0]
                    warmup_score = float(np.clip(0.5 - (raw_score * 0.5), 0.0, 1.0))
                except Exception:
                    warmup_score = 0.0

            self.all_warmup_scores.append(warmup_score)

            # Check if this warmup frame is anomalous
            if warmup_score >= self.warmup_anomaly_threshold:
                self.anomaly_during_warmup = True
                self.anomalous_frame_indices.append(len(self.all_warmup_samples) - 1)
                print(f"[WARMUP ANOMALY] Frame {len(self.all_warmup_samples)}: Score={warmup_score:.3f}")

            # For initial training, use all samples (we'll filter later if needed)
            self.warmup_samples.append(features.copy())

            if frame_embedding is not None:
                emb = np.asarray(frame_embedding, dtype=np.float32)

                if np.all(np.isfinite(emb)):
                    if self.baseline_embedding is None:
                        self.baseline_embedding = emb.copy()
                    elif emb.shape == self.baseline_embedding.shape:
                        self.baseline_embedding += emb
                    else:
                        raise ValueError("Embedding dimensions do not match")

                    self.embedding_count += 1

            if len(self.all_warmup_samples) >= self.warmup_frames:
                # Initial calibration on all samples
                if len(self.warmup_samples) >= 10:
                    self.detector.fit(
                        np.asarray(self.warmup_samples, dtype=np.float32)
                    )

                    if self.embedding_count > 0:
                        self.baseline_embedding /= self.embedding_count

                    self.is_calibrated = True

                    # If anomalies were detected, schedule recalibration
                    if self.anomaly_during_warmup:
                        print(f"[CALIBRATION] Initial calibration after {len(self.warmup_samples)} frames (anomalies detected, will re-calibrate)")
                        self._maybe_recalibrate()
                    else:
                        print(f"[CALIBRATION] Model calibrated after {len(self.warmup_samples)} frames")
                else:
                    print(f"[CALIBRATION WARNING] Only {len(self.warmup_samples)} warmup samples, model may be unreliable")
                    self.detector.fit(
                        np.asarray(self.warmup_samples, dtype=np.float32)
                    )
                    if self.embedding_count > 0:
                        self.baseline_embedding /= self.embedding_count
                    self.is_calibrated = True

            # KEY FIX: Return the warmup score so alerts can trigger during warmup
            # Use a lower threshold for warmup alerts
            return round(float(warmup_score), 3)

        # Kinematic anomaly score
        raw_score = self.detector.decision_function([features])[0]

        kinematic_score = float(
            np.clip(0.5 - (raw_score * 0.5), 0.0, 1.0)
        )

        # Semantic deviation score
        semantic_score = 0.0

        if frame_embedding is not None and self.baseline_embedding is not None:
            emb = np.asarray(frame_embedding, dtype=np.float32)

            if emb.shape == self.baseline_embedding.shape:
                denominator = (
                    np.linalg.norm(emb)
                    * np.linalg.norm(self.baseline_embedding)
                ) + 1e-7

                cosine_similarity = float(
                    np.dot(emb, self.baseline_embedding) / denominator
                )

                cosine_similarity = float(
                    np.clip(cosine_similarity, -1.0, 1.0)
                )

                semantic_score = float(
                    np.clip(1.0 - cosine_similarity, 0.0, 1.0)
                )

        # Combined anomaly score
        final_score = (
            0.6 * kinematic_score +
            0.4 * semantic_score
        )

        return round(float(final_score), 3)

    def get_warmup_stats(self) -> dict:
        """Get statistics about warmup phase for diagnostics"""
        if not self.all_warmup_scores:
            return {"max_warmup_score": 0.0, "mean_warmup_score": 0.0, "samples": 0, "anomaly_detected": False}

        return {
            "max_warmup_score": max(self.all_warmup_scores),
            "mean_warmup_score": float(np.mean(self.all_warmup_scores)),
            "samples": len(self.all_warmup_scores),
            "anomaly_detected": self.anomaly_during_warmup,
            "anomaly_threshold": self.warmup_anomaly_threshold,
            "anomalous_frames": len(self.anomalous_frame_indices),
            "recalibrated": self.is_recalibrated
        }