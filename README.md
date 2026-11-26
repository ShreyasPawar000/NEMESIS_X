# ◈ NEMESIS-X
### Open-World Video Intelligence with Temporal, Spatial & Semantic Reasoning

**Don't just watch videos, understand them.**

NEMESIS-X is an AI-powered video intelligence system designed to analyse video footage, understand object movements over time, identify unusual behavioural patterns, and generate AI-assisted incident reports.

Unlike traditional object detection systems that analyse individual frames, NEMESIS-X focuses on temporal behaviour, object tracking, scene changes, and anomaly detection.

---

## 🚀 Key Features

- **Real-Time Object Detection:** Detect objects using YOLO11n.
- **Multi-Object Tracking:** Track objects across video frames using ByteTrack.
- **Temporal Intelligence:** Analyse movement patterns and behavioural changes over time.
- **Anomaly Detection:** Identify deviations from learned scene behaviour using Isolation Forest.
- **AI-Powered Incident Reports:** Generate natural-language incident explanations using Google Gemini.
- **Live Alert System:** Transmit anomaly alerts through WebSockets.
- **Video Analysis:** Upload video footage and process it through the integrated computer vision pipeline.
- **Incident Report Download:** Download generated incident reports in TXT format.
- **Investigation Dashboard:** Explore detected anomalies through a Streamlit interface.

---

## 🧠 How NEMESIS-X Works

The system follows a five-stage intelligence pipeline:

| Stage | Process | Technology |
|---|---|---|
| 01 | Video Observation | OpenCV |
| 02 | Object Detection | YOLO11n |
| 03 | Object Tracking | ByteTrack |
| 04 | Temporal Analysis & Anomaly Detection | Python, Isolation Forest |
| 05 | Incident Explanation | Google Gemini |

### System Workflow

```text
       VIDEO INPUT
            |
            v
    COMPUTER VISION
            |
       YOLO11n
            |
            v
     OBJECT DETECTION
            |
            v
       BYTETRACK
            |
      OBJECT TRACKING
            |
            v
    TEMPORAL ANALYSIS
            |
            v
    ANOMALY DETECTION
            |
       FASTAPI ENGINE
            |
       +----+----+
       |         |
       v         v
   GEMINI AI   LIVE ALERTS
       |         |
       v         |
 INCIDENT REPORT |
       |         |
       +----+----+
            |
            v
     STREAMLIT DASHBOARD
```

---

## 🛠️ Technology Stack

### Computer Vision
- Python
- OpenCV
- Ultralytics YOLO11n
- ByteTrack

### Backend
- Python
- FastAPI
- Pydantic
- Scikit-learn
- WebSockets

### Artificial Intelligence
- Google Gemini API
- Isolation Forest
- Temporal Feature Extraction

### Frontend
- Streamlit
- HTML
- CSS

---

## 📁 Project Structure

```text
NEMESIS-X/
│
├── computer_vision/
│   ├── video_detector.py
│   ├── video_tracker.py
│   ├── video_processor.py
│   ├── process_and_ingest.py
│   └── yolo11n.pt
│
├── Backend/
│   └── nemesis_backend/
│       ├── app/
│       │   ├── main.py
│       │   ├── memory/
│       │   └── services/
│       ├── test_feeder.py
│       └── requirements.txt
│
├── Frontend/
│   └── frontend/
│       └── frontend/
│           ├── app.py
│           ├── assets/
│           └── components/
│
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.12
- Git
- Windows 10/11 or compatible operating system
- Google Gemini API key

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/NEMESIS-X.git
```

```bash
cd NEMESIS-X
```

### 2. Configure the Backend

Open a terminal:

```powershell
cd Backend/nemesis_backend
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Configure your Gemini API key:

```powershell
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

Start the backend:

```powershell
python -m uvicorn app.main:app --reload
```

Backend runs at:

`http://127.0.0.1:8000`

### 3. Start the Frontend

Open another terminal:

```powershell
cd Frontend/frontend/frontend
```

Run:

```powershell
python -m streamlit run app.py
```

The dashboard will be available at:

`http://localhost:8501`

### 4. Computer Vision Integration

The frontend invokes the computer vision processing pipeline when footage is submitted for analysis.

Ensure that:
- The Computer Vision virtual environment is configured.
- YOLO11n model weights are available.
- Required Python dependencies are installed.

---

## 🧪 Testing

NEMESIS-X includes a synthetic anomaly testing script.

Run:

```powershell
cd Backend/nemesis_backend
python test_feeder.py
```

The test sends baseline frames followed by simulated anomalous frames to the reasoning engine.

The system is expected to:
- Calibrate its baseline.
- Detect the injected anomaly.
- Generate an alert.
- Prevent repeated alerts for the same active event.
- Generate an AI-assisted incident explanation when the Gemini service is available.

---

## 📊 Anomaly Detection

NEMESIS-X extracts temporal features from tracked objects, including:

- Object count
- Object count variation
- Pairwise spatial distance
- Object velocity
- Acceleration
- Direction changes
- Stationary ratio

These features are analysed using an Isolation Forest model to identify deviations from learned baseline behaviour.

An anomaly score exceeding the configured threshold can trigger an incident alert.

**Important:** An anomaly score indicates deviation from learned behaviour. It does not establish criminal activity, intent, or danger. Human verification is required.

---

## 🔮 Future Enhancements

- Evidence frame extraction and visualization.
- Advanced spatial relationship graphs.
- Improved semantic scene understanding.
- Persistent incident history.
- Multi-camera intelligence.
- Natural-language video investigation.
- Advanced behavioural pattern recognition.

---

## 👥 Team

**Project:** NEMESIS-X

**Domain:** Artificial Intelligence | Computer Vision | Video Intelligence | Cyber Security

**Team Size:** 3 Members

### Team Contributions

- Computer Vision: Object detection, tracking and video processing.
- Backend: FastAPI, temporal memory, anomaly detection and AI integration.
- Frontend: Streamlit dashboard, investigation interface and report downloads.

---

## ⚠️ Limitations

- Detection performance depends on video quality and available computational resources.
- CPU-based video processing may be slower than GPU-based processing.
- Anomaly detection depends on the quality of baseline calibration.
- Gemini explanations depend on API availability and quota.
- The current investigation assistant is a frontend prototype.

---

## 📜 Disclaimer

NEMESIS-X is an experimental academic and hackathon project developed for video intelligence research and demonstration.

It is not a replacement for professional surveillance, security investigation, or human decision-making.

---

### ◈ NEMESIS-X

**Don't just watch videos, understand them.**
