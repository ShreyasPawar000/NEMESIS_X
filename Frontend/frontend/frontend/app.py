import subprocess
import tempfile
import json
import uuid
import re
import streamlit as st
from pathlib import Path
from components.api import get_health, start_alert_listener, drain_alerts
from streamlit_autorefresh import st_autorefresh
from components.ui import inject_css, metric_card, section_heading
st.set_page_config(page_title="NEMESIS-X | Temporal Intelligence", page_icon="◈", layout="wide")
ROOT = Path(__file__).parent
inject_css(ROOT / "assets" / "style.css")
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "last_ingest" not in st.session_state:
    st.session_state.last_ingest = None
if "page" not in st.session_state:
    st.session_state.page = "Command Centre"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
st.sidebar.markdown(
    '<div class="brand"><span class="brand-symbol">◈</span><div><b>NEMESIS<span>-X</span></b>'
    '<small>TEMPORAL VIDEO INTELLIGENCE</small></div></div>', unsafe_allow_html=True
)
st.sidebar.markdown('<div class="side-label">INVESTIGATION WORKSPACE</div>', unsafe_allow_html=True)
st.sidebar.radio("Navigate", ["Command Centre", "Footage Review", "Anomaly Desk", "Frame Ingest", "NEMESIS AI", "System Status"],
                 key="page", label_visibility="collapsed")
api_url = st.sidebar.text_input("REASONING ENGINE URL", "http://127.0.0.1:8000").rstrip("/")
ws_url = "ws" + api_url[len("http"): ] + "/ws/events" if api_url.startswith("http") else ""
if ws_url:
    start_alert_listener(ws_url)
# Refresh the page periodically so alerts received by the background listener
# become visible without requiring a manual browser refresh.
# Avoid automatic reruns while video processing is active.
if st.session_state.page != "Footage Review":
    st_autorefresh(
        interval=2000,
        key="nemesis_alert_refresh"
    )
for incoming_alert in drain_alerts():
    known_ids = {item.get("event_id") for item in st.session_state.alerts}
    if incoming_alert.get("event_id") not in known_ids:
        st.session_state.alerts.append(incoming_alert)
health = get_health(api_url)
sessions = health.get("sessions", {}) if health else {}
current_session = next(iter(sessions.values()), {}) if sessions else {}
buffer_depth = current_session.get("buffer_depth", "—")
calibrated = current_session.get("calibrated", False)
online = bool(health and health.get("status") == "healthy")
st.markdown(
    f'<div class="topbar"><div><span class="eyebrow">NEMESIS-X / INVESTIGATION NODE</span>'
    f'<div class="topbar-title">{st.session_state.page.upper()}</div></div>'
    f'<div class="connection {"online" if online else "offline"}"><i></i>'
    f'{"ENGINE CONNECTED" if online else "ENGINE UNAVAILABLE"}</div></div>',
    unsafe_allow_html=True
)
page = st.session_state.page
if page == "Command Centre":
    st.markdown(
        '<div class="hero"><div class="eyebrow">OPEN-WORLD VIDEO REASONING</div>'
        '<h1>Understand what<br><em>changes over time.</em></h1>'
        '<p>Most systems describe what appears in a frame. NEMESIS-X studies how a scene evolves, learns a baseline, and surfaces deviations with traceable evidence.</p></div>',
        unsafe_allow_html=True
    )
    a, b, c, d = st.columns(4)
    metric_card(a, "TEMPORAL BUFFER", str(buffer_depth) if health else "—", "frames currently retained")
    metric_card(b, "BASELINE MODEL", "CALIBRATED" if health and calibrated else "WARMING UP" if health else "—", "50-frame warm-up")
    metric_card(c, "SESSION ALERTS", str(len(st.session_state.alerts)), "alerts received by this UI")
    metric_card(d, "FLAG THRESHOLD", "0.65", "anomaly score trigger")
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    left, right = st.columns([1.25, .85], gap="large")
    with left:
        section_heading("01 / REASONING SEQUENCE", "How NEMESIS-X reads a scene")
        rows = [
            ("01", "OBSERVE", "Receive frame timestamps, detections, track IDs, bounding boxes and optional embeddings."),
            ("02", "REMEMBER", "Retain a rolling window of up to 40 frames and maintain object trajectories."),
            ("03", "LEARN", "Calibrate normal scene kinematics during the first 50 frames."),
            ("04", "COMPARE", "Combine kinematic deviation and semantic embedding drift into an anomaly score."),
            ("05", "EXPLAIN", "Return a time window, involved tracks, explanation and evidence frame IDs.")
        ]
        for n, title, desc in rows:
            st.markdown(f'<div class="process-row"><span>{n}</span><div><b>{title}</b><p>{desc}</p></div></div>', unsafe_allow_html=True)
    with right:
        section_heading("02 / CURRENT SIGNAL", "Latest captured alert")
        if st.session_state.alerts:
            alert = st.session_state.alerts[-1]
            st.markdown(f'<div class="alert-panel"><small>ANOMALY FLAGGED</small><strong>{alert.get("anomaly_score", 0):.3f}</strong><p>{alert.get("explanation", "")}</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty-panel"><div>⌁</div><b>No alert received</b><p>Alerts will appear here after frame ingestion and baseline calibration.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="caution"><b>READ THIS CORRECTLY</b><p>An anomaly score represents deviation from a learned baseline. It is not a probability of crime, intent or danger. Human review remains essential.</p></div>', unsafe_allow_html=True)
elif page == "Footage Review":
    section_heading("01 / FOOTAGE WORKSPACE", "Analyse video with NEMESIS-X")
    video = st.file_uploader(
        "Choose footage",
        type=["mp4", "mov", "avi", "mkv"]
    )
    if video:
        st.video(video)
        st.success(f"Video loaded: {video.name}")
        if st.button("RUN NEMESIS-X ANALYSIS", type="primary"):
            st.success("Analysis button triggered successfully!")
            cv_root = Path(
                r"F:\UG\UG\3rd Sem\Hackathon\computer_vision"
            )
            cv_python = cv_root / "venv" / "Scripts" / "python.exe"
            processor = cv_root / "process_and_ingest.py"
            if not cv_python.exists():
                st.error("CV Python environment not found.")
                st.stop()
            video_id = "upload_" + uuid.uuid4().hex[:8]
            suffix = Path(video.name).suffix
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp:
                temp.write(video.getbuffer())
                temp_path = Path(temp.name)
            try:
                with st.spinner(
                    "YOLO11n and ByteTrack are analysing your video..."
                ):
                    process = subprocess.run(
                        [
                            str(cv_python),
                            str(processor),
                            str(temp_path),
                            api_url,
                            video_id
                        ],
                        capture_output=True,
                        text=True,
                        timeout=3600
                    )
                st.code(process.stdout)
                if process.returncode != 0:
                    st.error("Video processing failed.")
                    st.code(process.stderr)
                else:
                    st.success("Video analysis completed!")
                    for line in process.stdout.splitlines():
                        if line.startswith("NEMESIS_X_RESULT="):
                            result = json.loads(
                                line.split("=", 1)[1]
                            )
                            st.metric(
                                "Frames Processed",
                                result["processed_frames"]
                            )
                            st.metric(
                                "Frames Sent",
                                result["frames_sent"]
                            )
                            st.metric(
                                "Unique Tracks",
                                result["unique_tracks"]
                            )
                            for alert in result["alerts"]:
                                st.session_state.alerts.append(alert)
                            st.metric(
                                "Alerts Generated",
                                len(result["alerts"])
                            )
            except subprocess.TimeoutExpired:
                st.error("Processing timed out.")
            finally:
                temp_path.unlink(missing_ok=True)
elif page == "Anomaly Desk":
    section_heading("01 / EVIDENCE REGISTER", "Alerts returned by the reasoner")

    if not st.session_state.alerts:
        st.markdown(
            '<div class="empty-panel"><div>◎</div>'
            '<b>Nothing flagged in this session</b>'
            '<p>The engine needs at least 10 frames in memory and a score '
            'of 0.65 or above to emit an alert. It first calibrates on 50 frames.</p></div>',
            unsafe_allow_html=True
        )

    for index, alert in enumerate(reversed(st.session_state.alerts), 1):
        event_id = alert.get("event_id", "EVENT")
        video_id = str(alert.get("video_id", "unknown"))
        score = float(alert.get("anomaly_score", 0) or 0)
        start_time = float(alert.get("start_time", 0) or 0)
        end_time = float(alert.get("end_time", 0) or 0)
        explanation = alert.get("explanation", "No explanation supplied.")
        track_ids = alert.get("involved_track_ids", []) or []
        evidence_frames = alert.get("evidence_frames", []) or []

        with st.expander(
            f'{index:02d} / {event_id} / SCORE {score:.3f}',
            expanded=index == 1
        ):
            st.markdown(explanation, unsafe_allow_html=False)

            x, y, z = st.columns(3)
            x.metric("VIDEO ID", video_id)
            y.metric("START", f"{start_time:.2f}s")
            z.metric("END", f"{end_time:.2f}s")

            st.write("**Involved track IDs**")
            st.write(", ".join(map(str, track_ids)) or "—")

            st.write("**Evidence frame IDs**")
            st.write(", ".join(map(str, evidence_frames)) or "—")

            report_text = f"""NEMESIS-X
VIDEO INCIDENT REPORT
==================================================

EVENT ID: {event_id}
VIDEO ID: {video_id}
ANOMALY SCORE: {score:.3f}

TEMPORAL INFORMATION
--------------------------------------------------
Start Time: {start_time:.2f} seconds
End Time: {end_time:.2f} seconds

INVOLVED TRACK IDS
--------------------------------------------------
{', '.join(map(str, track_ids)) or 'None'}

EVIDENCE FRAME IDS
--------------------------------------------------
{', '.join(map(str, evidence_frames)) or 'None'}

GEMINI INCIDENT ANALYSIS
--------------------------------------------------
{explanation}

==================================================
Generated by NEMESIS-X Temporal Intelligence
Anomaly scores indicate deviation from a learned baseline; they are not
probabilities of crime, intent, or danger. Human review is required.
"""
            safe_video_id = re.sub(r"[^A-Za-z0-9_-]+", "_", video_id)
            safe_event_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(event_id))

            st.download_button(
                label="⬇ DOWNLOAD INCIDENT REPORT (TEXT)",
                data=report_text,
                file_name=f"NEMESIS_X_Report_{safe_video_id}_{safe_event_id}.txt",
                mime="text/plain",
                key=f"download_report_txt_{index}_{safe_event_id}",
                use_container_width=True
            )

            # PDF Report download button
            if alert.get("pdf_report_path"):
                pdf_filename = Path(alert["pdf_report_path"]).name
                if st.button(
                    "⬇ DOWNLOAD PDF REPORT (WITH EVIDENCE SNAPSHOTS)",
                    key=f"download_pdf_{index}_{safe_event_id}",
                    use_container_width=True
                ):
                    with st.spinner("Downloading PDF report..."):
                        from components.api import download_report
                        pdf_content, error = download_report(api_url, pdf_filename)
                        if error:
                            st.error(f"Failed to download PDF: {error}")
                        else:
                            st.download_button(
                                label="⬇ CLICK TO SAVE PDF",
                                data=pdf_content,
                                file_name=pdf_filename,
                                mime="application/pdf",
                                key=f"save_pdf_{index}_{safe_event_id}",
                                use_container_width=True
                            )
                            st.success("PDF ready for download!")
            else:
                st.info("PDF report with evidence snapshots not available for this alert. (PDF generation requires backend to have processed the tracked video)")

    st.markdown(
        '<div class="caution"><b>EVIDENCE LIMITATION</b>'
        '<p>The current API returns evidence frame IDs, not the actual image files or clips. '
        'This page displays the IDs and explanation supplied by the backend.</p></div>',
        unsafe_allow_html=True
    )

elif page == "Frame Ingest":
    section_heading("01 / FRAME RELAY", "Transmit a frame to the reasoning engine")
    st.markdown('<p class="intro">A test console for the exact FramePayload accepted by FastAPI. This page does not perform object detection.</p>', unsafe_allow_html=True)
    import json
    from components.api import post_frame
    example = {
        "video_id": "camera_north_01", "frame_id": 1, "timestamp": 0.04,
        "detections": [
            {"track_id": 1, "label": "person", "confidence": 0.94, "bbox": [0.1, 0.2, 0.3, 0.8]},
            {"track_id": 2, "label": "person", "confidence": 0.91, "bbox": [0.6, 0.2, 0.8, 0.8]}
        ],
        "frame_embedding": None
    }
    raw = st.text_area("JSON FRAME PAYLOAD", json.dumps(example, indent=2), height=330)
    if st.button("TRANSMIT FRAME", type="primary", use_container_width=True):
        try:
            payload = json.loads(raw)
            result, error = post_frame(api_url, payload)
            if error:
                st.error(error)
            else:
                st.session_state.last_ingest = result
                alert = result.get("alert")
                if alert:
                    st.session_state.alerts.append(alert)
                    st.warning("The backend returned an anomaly alert.")
                else:
                    st.success("Frame accepted.")
                st.json(result)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
    if st.session_state.last_ingest:
        section_heading("02 / LAST RESPONSE", "Backend acknowledgement")
        st.json(st.session_state.last_ingest)
elif page == "NEMESIS AI":
    section_heading("01 / INVESTIGATION ASSISTANT", "Ask questions about your footage")
    st.markdown(
        '<p class="intro">NEMESIS AI is an interactive investigation assistant concept. '
        'This chat interface is currently a frontend prototype; it is not connected to an AI model '
        'or the video-analysis pipeline yet.</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="caution"><b>PROTOTYPE MODE</b><p>Responses below are demonstration messages, '
        'not findings generated from your actual video footage.</p></div>',
        unsafe_allow_html=True
    )
    if not st.session_state.chat_history:
        st.markdown(
            '<div class="empty-panel"><div>◈</div><b>Start an investigation</b>'
            '<p>Try asking: "What unusual activity was detected?" or '
            '"When did the latest alert occur?"</p></div>',
            unsafe_allow_html=True
        )
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("Ask NEMESIS AI about the footage...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        if st.session_state.alerts:
            latest = st.session_state.alerts[-1]
            answer = (
                "This is a prototype response, not an AI-generated investigation. "
                f"The latest alert currently stored in this session has score "
                f"{latest.get('anomaly_score', 0):.3f}, video ID "
                f"{latest.get('video_id', '—')}, and explanation: "
                f"{latest.get('explanation', 'No explanation supplied.')}"
            )
        else:
            answer = (
                "NEMESIS AI is not connected to an AI model yet, so I cannot analyse "
                "the footage or verify suspicious activity. Once the AI service and "
                "video-analysis data are integrated, you will be able to ask questions "
                "about detected events, timestamps, and tracked objects."
            )
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
    if st.button("CLEAR CONVERSATION"):
        st.session_state.chat_history = []
        st.rerun()
elif page == "System Status":
    section_heading("01 / SERVICE HEALTH", "Reasoning engine connection")
    if health:
        st.success("FastAPI health endpoint responded.")
        st.json(health)
    else:
        st.error("Backend unavailable. Start FastAPI and verify the URL in the sidebar.")
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    section_heading("02 / ROUTE MAP", "Available backend interfaces")
    for method, route, detail in [
        ("GET", "/api/v1/health", "Health status, buffer depth and calibration state"),
        ("POST", "/api/v1/ingest", "Submit a frame and receive score / optional alert"),
        ("WS", "/ws/events", "WebSocket broadcast for anomaly alerts")
    ]:
        st.markdown(f'<div class="route"><code>{method}</code><b>{route}</b><span>{detail}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="caution"><b>NOT IMPLEMENTED IN THE SUPPLIED BACKEND</b><p>Video upload/decoding, object detection, tracker execution, relation graphs, evidence image retrieval and persistent alert history require additional CV/backend integration.</p></div>', unsafe_allow_html=True)
st.markdown('<div class="footer">NEMESIS-X &nbsp; / &nbsp; TEMPORAL REASONING &nbsp; / &nbsp; HUMAN REVIEW REQUIRED</div>', unsafe_allow_html=True)
