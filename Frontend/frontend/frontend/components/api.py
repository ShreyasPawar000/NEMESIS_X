import json
import queue
import threading
import time

import requests
import websocket

_alert_queue = queue.Queue()
_listener_lock = threading.Lock()
_listener_started = False


def get_health(base_url: str):
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/v1/health", timeout=2.5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def post_frame(base_url: str, payload: dict):
    try:
        response = requests.post(f"{base_url.rstrip('/')}/api/v1/ingest", json=payload, timeout=8)
        response.raise_for_status()
        return response.json(), None
    except requests.HTTPError as exc:
        body = exc.response.text if exc.response is not None else str(exc)
        return None, f"Backend rejected the payload: {body}"
    except requests.RequestException as exc:
        return None, f"Could not reach backend: {exc}"


def start_alert_listener(ws_url: str):
    """Start one background WebSocket listener for backend alert broadcasts."""
    global _listener_started
    with _listener_lock:
        if _listener_started:
            return
        _listener_started = True

    def listen_forever():
        while True:
            try:
                def on_open(ws):
                    # The FastAPI endpoint waits for a client message while keeping
                    # the socket open for server-to-client alert broadcasts.
                    ws.send("subscribe")

                def on_message(ws, message):
                    try:
                        payload = json.loads(message)
                        if isinstance(payload, dict):
                            _alert_queue.put(payload)
                    except json.JSONDecodeError:
                        pass

                app = websocket.WebSocketApp(
                    ws_url,
                    on_open=on_open,
                    on_message=on_message,
                )
                app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:
                pass
            time.sleep(3)

    threading.Thread(target=listen_forever, daemon=True, name="nemesis-ws-listener").start()


def drain_alerts():
    alerts = []
    while True:
        try:
            alerts.append(_alert_queue.get_nowait())
        except queue.Empty:
            break
    return alerts


def download_report(base_url: str, report_filename: str):
    """Download a PDF report from the backend."""
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/v1/reports/{report_filename}", timeout=10)
        response.raise_for_status()
        return response.content, None
    except requests.RequestException as exc:
        return None, f"Could not download report: {exc}"
