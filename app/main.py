from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List, Optional
from app.schemas import FramePayload, AnomalyAlert
from app.memory.temporal_buffer import TemporalMemory
from app.services.anomaly import AnomalyEngine
from app.services.reasoner import EvidenceReasoner

app = FastAPI(title="NEMESIS-X Reasoning Engine")

# In-memory session tracking
memory = TemporalMemory(window_size=40)
detector = AnomalyEngine(warmup_frames=50, contamination=0.08)
connected_clients: List[WebSocket] = []

ANOMALY_TRIGGER = 0.65

@app.websocket("/ws/events")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(ws)

@app.post("/api/v1/ingest")
async def ingest_frame(frame: FramePayload):
    memory.push(frame)
    feats = memory.extract_interaction_vector()
    score = detector.score(feats, frame.frame_embedding)

    alert_payload: Optional[AnomalyAlert] = None
    if score >= ANOMALY_TRIGGER and len(memory.frames) >= 10:
        alert_payload = EvidenceReasoner.generate_alert(
            window=list(memory.frames),
            score=score,
            video_id=frame.video_id
        )

        dead_connections = []
        for client in connected_clients:
            try:
                await client.send_text(alert_payload.model_dump_json())
            except Exception:
                dead_connections.append(client)
        for dead in dead_connections:
            connected_clients.remove(dead)

    return {
        "status": "ok",
        "calibrated": detector.is_calibrated,
        "score": score,
        "alert_triggered": alert_payload is not None,
        "alert": alert_payload
    }

@app.get("/api/v1/health")
def health():
    return {
        "status": "healthy",
        "buffer_depth": len(memory.frames),
        "calibrated": detector.is_calibrated
    }