# NEMESIS-X Frontend

## Run locally
1. Start the FastAPI backend.
2. Open PowerShell in this folder.
3. Install: `python -m pip install -r requirements.txt`
4. Run: `python -m streamlit run app.py`
5. Open `http://localhost:8501`.

Set the backend URL in the sidebar if the API is not at `http://127.0.0.1:8000`.

## Connected backend routes
- `GET /api/v1/health`
- `POST /api/v1/ingest`
- `WS /ws/events` exists in the backend; this frontend version does not yet subscribe to it.

## Honest feature boundaries
The supplied backend ingests frame-level detections and optional embeddings. It does not currently decode uploaded videos, run object detection/tracking, expose relationship graphs, serve evidence images/clips, or persist alert history. The Footage Review upload is a local preview only. Use a CV producer to decode frames and send valid payloads to `/api/v1/ingest`.

The backend's anomaly score is a deviation signal, not a probability of crime, intent or danger. Keep human review in the loop.
