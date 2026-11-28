from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    Response,
    UploadFile,
    File,
    Form,
    HTTPException
)

from typing import List, Optional
from pathlib import Path
import asyncio
import traceback
import shutil
import re
import json
import cv2

from app.schemas import FramePayload, AnomalyAlert
from app.memory.temporal_buffer import TemporalMemory
from app.services.anomaly import AnomalyEngine
from app.services.reasoner import EvidenceReasoner
from app.services.llm_explainer import generate_explanation
from app.services.pdf_report import generate_pdf_report


# ==========================================
# APPLICATION CONFIGURATION
# ==========================================

app = FastAPI(title="NEMESIS-X Reasoning Engine")

BACKEND_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BACKEND_DIR / "output"

TRACKED_VIDEO_DIR = OUTPUT_DIR / "tracked_videos"
TRACKING_DATA_DIR = OUTPUT_DIR / "tracking_data"
REPORT_DIR = OUTPUT_DIR / "reports"

TRACKED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
TRACKING_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# CONFIGURATION
# ==========================================

video_sessions = {}
connected_clients: List[WebSocket] = []

ANOMALY_TRIGGER = 0.45
WARMUP_ANOMALY_TRIGGER = 0.40
WARMUP_CONSECUTIVE_FRAMES = 1
NORMAL_FRAMES_TO_RESET = 5

SUPPORTED_VIDEO_FORMATS = {
    ".avi", ".mp4", ".mov", ".mkv"
}


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
            "normal_frames": 0,
            "warmup_consecutive_anomaly": 0
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
# VIDEO ASSET UPLOAD API
# ==========================================

@app.post("/api/v1/videos/upload")
async def upload_video_assets(
    video_id: str = Form(...),
    tracked_video: UploadFile = File(...),
    tracking_json: UploadFile = File(...)
):

    if not re.fullmatch(r"[a-zA-Z0-9_-]+", video_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid video ID"
        )

    extension = Path(
        tracked_video.filename or ""
    ).suffix.lower()

    if extension not in SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported tracked video format"
        )

    video_dir = TRACKED_VIDEO_DIR / video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    video_path = video_dir / f"{video_id}{extension}"
    json_path = TRACKING_DATA_DIR / f"{video_id}.json"

    try:

        # Save tracked video
        with open(video_path, "wb") as video_file:
            shutil.copyfileobj(
                tracked_video.file,
                video_file
            )

        # Read and validate tracking JSON
        json_content = await tracking_json.read()

        try:
            json.loads(json_content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise HTTPException(
                status_code=400,
                detail="Invalid tracking JSON"
            )

        with open(json_path, "wb") as json_file:
            json_file.write(json_content)

        print(f"[UPLOAD] Video saved: {video_path}")
        print(f"[UPLOAD] Tracking data saved: {json_path}")

        return {
            "status": "success",
            "video_id": video_id,
            "video_path": str(video_path),
            "tracking_path": str(json_path)
        }

    except HTTPException:
        raise

    except Exception as error:

        print("[UPLOAD ERROR]", error)

        raise HTTPException(
            status_code=500,
            detail="Failed to store video assets"
        )

    finally:

        await tracked_video.close()
        await tracking_json.close()


# ==========================================
# FRAME INGESTION API
# ==========================================

@app.post("/api/v1/ingest")
async def ingest_frame(frame: FramePayload):

    session = get_video_session(frame.video_id)

    memory = session["memory"]
    detector = session["detector"]

    # Store incoming frame
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

    is_scored = (
        detector.is_calibrated
        and len(memory.frames) >= 10
    )

    threshold = (
        WARMUP_ANOMALY_TRIGGER
        if is_warmup
        else ANOMALY_TRIGGER
    )

    threshold_crossed = (
        score >= threshold
        and (is_scored or is_warmup)
    )

    alert_emitted = False

    # Track consecutive warmup anomalies
    if is_warmup and score >= threshold:

        session["warmup_consecutive_anomaly"] += 1

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
        f"Crossed={threshold_crossed}"
    )

    alert_payload: Optional[AnomalyAlert] = None

    # ======================================
    # ANOMALY DETECTION
    # ======================================

    warmup_consecutive_met = (
        is_warmup
        and session["warmup_consecutive_anomaly"]
        >= WARMUP_CONSECUTIVE_FRAMES
    )

    should_trigger = (
        score >= threshold
        and len(memory.frames) >= 10
        and (not is_warmup or warmup_consecutive_met)
    )

    if should_trigger:

        session["normal_frames"] = 0
        alert_emitted = True

        if not session["alert_active"]:

            session["alert_active"] = True

            phase = "WARMUP" if is_warmup else "SCORED"

            print(
                f"\n[ALERT] New anomaly detected ({phase}): "
                f"{frame.video_id} | Score={score:.3f} | "
                f"Frame={frame.frame_id}"
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

            alert_payload = alert_payload.model_copy(
                update={
                    "explanation": explanation
                }
            )

            # ==================================
            # GENERATE PDF REPORT
            # ==================================

            try:

                evidence_frames = set(
                    alert_payload.evidence_frames
                )

                video_path = None

                video_fps = 24.0
                video_width = 1280
                video_height = 720

                tracking_video_id = alert_payload.video_id

                tracking_file = (
                    TRACKING_DATA_DIR
                    / f"{tracking_video_id}.json"
                )

                if tracking_file.exists():

                    print(
                        f"[PDF DEBUG] Loading tracking data: "
                        f"{tracking_file}"
                    )

                    with open(tracking_file, "r") as file:
                        tracking_records = json.load(file)

                    tracking_frame_timestamps = {
                        record["frame"]: record["timestamp"]
                        for record in tracking_records
                        if "frame" in record
                        and "timestamp" in record
                    }

                    missing_frames = [
                        ef for ef in evidence_frames
                        if ef not in tracking_frame_timestamps
                    ]

                    if missing_frames:

                        print(
                            f"[PDF WARNING] Missing evidence frames: "
                            f"{missing_frames}"
                        )

                    else:

                        # Verify timestamps
                        matches = True

                        for evidence_frame in evidence_frames:

                            alert_timestamp = next(
                                (
                                    stored_frame.timestamp
                                    for stored_frame in memory.frames
                                    if stored_frame.frame_id == evidence_frame
                                ),
                                None
                            )

                            if alert_timestamp is None:

                                print(
                                    f"[PDF DEBUG] Frame "
                                    f"{evidence_frame} missing from memory"
                                )

                                matches = False
                                break

                            tracking_timestamp = (
                                tracking_frame_timestamps[evidence_frame]
                            )

                            if abs(
                                tracking_timestamp - alert_timestamp
                            ) > 0.1:

                                matches = False
                                break

                        if matches:

                            tracked_video_dir = (
                                TRACKED_VIDEO_DIR
                                / tracking_video_id
                            )

                            video_files = (
                                [
                                    file
                                    for file in tracked_video_dir.iterdir()
                                    if file.suffix.lower()
                                    in SUPPORTED_VIDEO_FORMATS
                                ]
                                if tracked_video_dir.exists()
                                else []
                            )

                            if video_files:

                                video_path = str(video_files[0])

                                cap = cv2.VideoCapture(video_path)

                                if cap.isOpened():

                                    video_fps = cap.get(
                                        cv2.CAP_PROP_FPS
                                    )

                                    video_width = int(
                                        cap.get(
                                            cv2.CAP_PROP_FRAME_WIDTH
                                        )
                                    )

                                    video_height = int(
                                        cap.get(
                                            cv2.CAP_PROP_FRAME_HEIGHT
                                        )
                                    )

                                cap.release()

                                print(
                                    f"[PDF] Found tracked video: "
                                    f"{video_path}"
                                )

                            else:

                                print(
                                    f"[PDF WARNING] No tracked video "
                                    f"found for {tracking_video_id}"
                                )

                        else:

                            print(
                                "[PDF WARNING] Evidence timestamps "
                                "do not match tracking data"
                            )

                else:

                    print(
                        f"[PDF WARNING] Tracking file not found: "
                        f"{tracking_file}"
                    )

                # Generate PDF
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

                    print(
                        f"[PDF] Report generated: {pdf_path}"
                    )

                    alert_payload.pdf_report_path = pdf_path

                else:

                    print(
                        "[PDF WARNING] PDF generation skipped. "
                        "Matching tracked video unavailable."
                    )

            except Exception as pdf_error:

                print("[PDF ERROR]", pdf_error)
                traceback.print_exc()

            # ==================================
            # BROADCAST ALERT
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

    # Prevent path traversal
    if Path(report_filename).name != report_filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid report filename"
        )

    if not report_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid report format"
        )

    report_path = REPORT_DIR / report_filename

    if not report_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Report not found"
        )

    return Response(
        content=report_path.read_bytes(),
        media_type="application/pdf",
        headers={
            "Content-Disposition":
                f'attachment; filename="{report_filename}"'
        }
    )
