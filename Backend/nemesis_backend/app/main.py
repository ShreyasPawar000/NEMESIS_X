from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from typing import List, Optional
import asyncio
import traceback
import os
import cv2
import json
from pathlib import Path

from app.schemas import FramePayload, AnomalyAlert
from app.memory.temporal_buffer import TemporalMemory
from app.services.anomaly import AnomalyEngine
from app.services.reasoner import EvidenceReasoner
from app.services.llm_explainer import generate_explanation
from app.services.pdf_report import generate_pdf_report


app = FastAPI(title="NEMESIS-X Reasoning Engine")


# ==========================================
# CONFIGURATION
# ==========================================

video_sessions = {}
connected_clients: List[WebSocket] = []

ANOMALY_TRIGGER = 0.45   # lowered from 0.65 to catch smaller deviations
WARMUP_ANOMALY_TRIGGER = 0.40   # lowered from 0.55 to catch warmup anomalies sooner
WARMUP_CONSECUTIVE_FRAMES = 1   # lowered from 2 — trigger on first anomalous frame
NORMAL_FRAMES_TO_RESET = 5


# ==========================================
# PER-VIDEO SESSION MANAGEMENT
# ==========================================

def get_video_session(video_id: str):

    if video_id not in video_sessions:

        video_sessions[video_id] = {
            "memory": TemporalMemory(window_size=40),

            "detector": AnomalyEngine(
                warmup_frames=50,
                contamination=0.08
            ),

            "alert_active": False,
            "normal_frames": 0
        }

        print(f"[SESSION] Created: {video_id}")

    return video_sessions[video_id]


# ==========================================
# WEBSOCKET BROADCAST
# ==========================================

async def broadcast_alert(alert):

    message = alert.model_dump_json()
    dead_connections = []

    for client in connected_clients:

        try:
            await client.send_text(message)

        except Exception as error:
            print("[WEBSOCKET ERROR]", error)
            dead_connections.append(client)

    for client in dead_connections:

        if client in connected_clients:
            connected_clients.remove(client)


# ==========================================
# WEBSOCKET CONNECTION
# ==========================================

@app.websocket("/ws/events")
async def websocket_endpoint(ws: WebSocket):

    await ws.accept()
    connected_clients.append(ws)

    print("[WEBSOCKET] Client connected")

    try:

        while True:
            await ws.receive_text()

    except WebSocketDisconnect:

        if ws in connected_clients:
            connected_clients.remove(ws)

        print("[WEBSOCKET] Client disconnected")


# ==========================================
# GEMINI INCIDENT REPORT
# ==========================================

async def generate_incident_report(alert, frames):

    print("\n[GEMINI] Generating incident report...")

    try:

        explanation = await asyncio.to_thread(
            generate_explanation,
            alert,
            frames
        )

        if explanation and explanation.strip():

            print("[GEMINI] Incident report generated successfully")

            return explanation.strip()

        print("[GEMINI] No report returned. Using fallback.")

    except Exception:

        print("[GEMINI ERROR]")
        traceback.print_exc()

    return alert.explanation


# ==========================================
# FRAME INGESTION API
# ==========================================

@app.post("/api/v1/ingest")
async def ingest_frame(frame: FramePayload):

    session = get_video_session(frame.video_id)

    memory = session["memory"]
    detector = session["detector"]

    # Store frame
    memory.push(frame)

    # Extract temporal features
    feats = memory.extract_interaction_vector()

    # Calculate anomaly score
    score = detector.score(
        feats,
        frame.frame_embedding
    )

    # Determine frame status
    is_warmup = not detector.is_calibrated
    is_scored = detector.is_calibrated and len(memory.frames) >= 10
    threshold = ANOMALY_TRIGGER if not is_warmup else WARMUP_ANOMALY_TRIGGER
    threshold_crossed = score >= threshold and (is_scored or is_warmup)
    alert_emitted = False

    # Track consecutive warmup frames above threshold
    if is_warmup and score >= threshold:
        session["warmup_consecutive_anomaly"] = session.get("warmup_consecutive_anomaly", 0) + 1
    else:
        session["warmup_consecutive_anomaly"] = 0

    print(
        f"[ANALYSIS] Video={frame.video_id} | "
        f"Frame={frame.frame_id} | "
        f"Objects={len(frame.detections)} | "
        f"Score={score:.3f} | "
        f"Calibrated={detector.is_calibrated} | "
        f"Warmup={is_warmup} | "
        f"Scored={is_scored} | "
        f"Threshold={threshold} | "
        f"Crossed={threshold_crossed} | "
        f"WarmupConsecutive={session.get('warmup_consecutive_anomaly', 0)}/{WARMUP_CONSECUTIVE_FRAMES} | "
        f"WarmupSamples={len(detector.warmup_samples)}/{detector.warmup_frames}"
    )

    alert_payload: Optional[AnomalyAlert] = None

    # ======================================
    # ANOMALY DETECTION
    # ======================================

    warmup_consecutive_met = is_warmup and session.get("warmup_consecutive_anomaly", 0) >= WARMUP_CONSECUTIVE_FRAMES

    if (score >= threshold and len(memory.frames) >= 10) and (not is_warmup or warmup_consecutive_met):

        session["normal_frames"] = 0
        alert_emitted = True

        if not session["alert_active"]:

            session["alert_active"] = True

            phase = "WARMUP" if is_warmup else "SCORED"
            print(
                f"\n[ALERT] New anomaly detected ({phase}): "
                f"{frame.video_id} | Score={score:.3f} | "
                f"Frame={frame.frame_id} | Threshold={threshold}"
            )

            # Generate evidence-based alert
            alert_payload = EvidenceReasoner.generate_alert(
                window=list(memory.frames),
                score=score,
                video_id=frame.video_id
            )

            # ==================================
            # GENERATE GEMINI REPORT
            # ==================================

            explanation = await generate_incident_report(
                alert_payload,
                list(memory.frames)
            )

            # Update alert with full report
            alert_payload = alert_payload.model_copy(
                update={
                    "explanation": explanation
                }
            )

            # ==================================
            # GENERATE PDF REPORT WITH SCREENSHOTS
            # ==================================

            try:
                # The alert's video_id might be different from the original tracking data video_id
                # (e.g., client uses a different ID when sending frames to API)
                # We need to find the tracking data that matches the alert's evidence frames AND timestamps
                evidence_frames = set(alert_payload.evidence_frames)

                # Get timestamps for evidence frames - use tracking data since memory.frames only keeps last 40 frames
                evidence_timestamps = {}

                video_path = None
                video_fps = 24.0
                video_width = 1280
                video_height = 720

                # Direct lookup: tracking files are named by video_id (see video_processor.py)
                BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
                tracking_video_id = alert_payload.video_id
                tracking_file = BASE_DIR / "output" / "tracking_data" / f"{tracking_video_id}.json"

                if tracking_file.exists():
                    print(f"[PDF DEBUG] Loading tracking data from {tracking_file}")
                    with open(tracking_file, 'r') as f:
                        tracking_records = json.load(f)
                    if tracking_records:
                        # Build mapping of frame -> timestamp for this tracking data
                        tracking_frame_timestamps = {r.get("frame"): r.get("timestamp") for r in tracking_records if "frame" in r and "timestamp" in r}

                        # Check if all evidence frames exist in this tracking data
                        missing_frames = [ef for ef in evidence_frames if ef not in tracking_frame_timestamps]
                        if missing_frames:
                            print(f"[PDF DEBUG] {tracking_file.name}: missing frames {missing_frames}")
                        else:
                            # Build evidence_timestamps from this tracking data
                            evidence_timestamps = {ef: tracking_frame_timestamps[ef] for ef in evidence_frames}

                            # Verify timestamp match within 0.1s tolerance using ALERT timestamps
                            matches = True
                            for ef in evidence_frames:
                                # Get alert's timestamp for this frame from memory (the authoritative source)
                                alert_timestamp = None
                                for frame in memory.frames:
                                    if frame.frame_id == ef:
                                        alert_timestamp = frame.timestamp
                                        break
                                if alert_timestamp is None:
                                    print(f"[PDF DEBUG] Frame {ef} not found in memory frames")
                                    matches = False
                                    break
                                diff = abs(tracking_frame_timestamps[ef] - alert_timestamp)
                                if diff > 0.1:
                                    matches = False
                                    break

                            if matches:
                                tracked_video_dir = BASE_DIR / "output" / "tracked_videos" / tracking_video_id
                                video_files = list(tracked_video_dir.glob("*.avi")) if tracked_video_dir.exists() else []
                                if video_files:
                                    video_path = str(video_files[0])
                                    cap = cv2.VideoCapture(video_path)
                                    video_fps = cap.get(cv2.CAP_PROP_FPS)
                                    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    cap.release()
                                    print(f"[PDF] Found matching tracked video: {video_path} (tracking: {tracking_video_id})")
                                else:
                                    print(f"[PDF WARNING] No tracked avi found for {tracking_video_id}")
                else:
                    print(f"[PDF WARNING] Tracking file not found: {tracking_file}")

                if video_path:
                    pdf_path = await asyncio.to_thread(
                        generate_pdf_report,
                        alert_payload,
                        list(memory.frames),
                        video_path,
                        video_fps,
                        video_width,
                        video_height
                    )
                    print(f"[PDF] Report generated: {pdf_path}")
                    alert_payload.pdf_report_path = pdf_path
                else:
                    print(f"[PDF WARNING] No tracked video found matching evidence frames: {evidence_frames}")
            except Exception as pdf_error:
                print(f"[PDF ERROR] {pdf_error}")
                traceback.print_exc()

            # ==================================
            # BROADCAST TO FRONTEND
            # ==================================

            await broadcast_alert(alert_payload)

            print("[ALERT] Processing completed")

        else:

            print(
                f"[ALERT ACTIVE] Existing event | "
                f"Video={frame.video_id}"
            )

    else:

        if is_scored:
            # Only count as normal frame if we're in scoring mode
            session["normal_frames"] += 1

        if session["normal_frames"] >= NORMAL_FRAMES_TO_RESET:

            if session["alert_active"]:
                print(
                    f"[ALERT RESET] Video={frame.video_id}"
                )

            session["alert_active"] = False

    # ======================================
    # API RESPONSE
    # ======================================

    return {
        "status": "ok",
        "video_id": frame.video_id,
        "calibrated": detector.is_calibrated,
        "score": score,
        "alert_triggered": alert_payload is not None,
        "alert": alert_payload,
        "diagnostics": {
            "is_warmup": is_warmup,
            "is_scored": is_scored,
            "threshold": threshold,
            "threshold_crossed": threshold_crossed,
            "warmup_samples": len(detector.warmup_samples),
            "warmup_frames_required": detector.warmup_frames,
            "normal_frames_count": session["normal_frames"],
            "alert_active": session["alert_active"]
        }
    }


# ==========================================
# HEALTH MONITORING
# ==========================================

@app.get("/api/v1/health")
def health():

    return {
        "status": "healthy",

        "active_video_sessions": len(video_sessions),

        "sessions": {

            video_id: {

                "buffer_depth": len(
                    session["memory"].frames
                ),

                "calibrated": session["detector"].is_calibrated,

                "alert_active": session["alert_active"],

                "normal_frames": session["normal_frames"]

            }

            for video_id, session in video_sessions.items()
        }
    }


# ==========================================
# PDF REPORT DOWNLOAD ENDPOINT
# ==========================================

@app.get("/api/v1/reports/{report_filename}")
def download_report(report_filename: str):
    """Download a generated PDF report by filename."""
    BASE_DIR = Path(__file__).resolve().parent.parent
    report_path = BASE_DIR / "output" / "reports" / report_filename

    if not report_path.exists():
        return Response(content="Report not found", status_code=404)

    with open(report_path, "rb") as f:
        content = f.read()

    return Response(
        content=content,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{report_filename}"'
        }
    )