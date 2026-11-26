// Connects to local FastAPI backend WebSocket
const wsProtocol = window.location.protocol === "https:" ? "wss://" : "ws://";
const wsHost = window.location.hostname || "localhost";
const wsUrl = `${wsProtocol}${wsHost}:8000/ws/events`;

const socket = new WebSocket(wsUrl);

const statusBadge = document.getElementById("connectionStatus");
const statusText = document.getElementById("statusText");
const alertsList = document.getElementById("alertsList");
const placeholderText = document.getElementById("placeholderText");

let hasReceivedAlert = false;

// 1. Connection Event Listeners
socket.onopen = () => {
  statusBadge.className = "status-badge connected";
  statusText.textContent = "Live WS Connected";
};

socket.onclose = () => {
  statusBadge.className = "status-badge disconnected";
  statusText.textContent = "Disconnected";
};

socket.onerror = (error) => {
  console.error("WebSocket connection error:", error);
  statusBadge.className = "status-badge disconnected";
  statusText.textContent = "Connection Error";
};

// 2. Incoming Threat/Anomaly Stream Handler
socket.onmessage = (event) => {
  try {
    const data = JSON.parse(event.data);

    // Clear initial placeholder message on first alert received
    if (!hasReceivedAlert) {
      if (placeholderText) placeholderText.remove();
      hasReceivedAlert = true;
    }

    // Render alert card into container
    const card = document.createElement("div");
    card.className = "alert-card";

    const scorePercent = (data.anomaly_score * 100).toFixed(1);
    const tracksDisplay = Array.isArray(data.involved_track_ids) && data.involved_track_ids.length > 0
      ? data.involved_track_ids.join(", ")
      : "None";

    card.innerHTML = `
      <div class="alert-header">
        <strong>Event ID: ${data.event_id}</strong>
        <span class="score-badge">Score: ${scorePercent}%</span>
      </div>
      <div class="alert-body">${data.explanation}</div>
      <div class="alert-meta">
        Timeline: ${Number(data.start_time).toFixed(2)}s → ${Number(data.end_time).toFixed(2)}s | Tracks: [${tracksDisplay}]
      </div>
    `;

    // Prepend to show the most recent alert at the top
    alertsList.prepend(card);
  } catch (err) {
    console.error("Failed to parse alert payload:", err);
  }
};