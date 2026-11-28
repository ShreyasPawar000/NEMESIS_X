# ◈ NEMESIS-X
### Open-World Video Intelligence with Temporal, Spatial & Semantic Reasoning

**Don't just watch videos, understand them.**

NEMESIS-X is an AI-powered video intelligence system designed to analyse video footage, track objects over time, identify unusual behavioural patterns and generate AI-assisted incident reports.

Unlike traditional surveillance systems that focus primarily on detecting objects in individual frames, NEMESIS-X analyses temporal behaviour, object movement and interactions to identify deviations from learned scene behaviour.

---

## 🚀 Key Features

- **Object Detection:** Detect objects using YOLO11n.
- **Multi-Object Tracking:** Track objects across video frames using ByteTrack.
- **Temporal Intelligence:** Analyse movement patterns and behavioural changes over time.
- **Anomaly Detection:** Identify deviations from learned scene behaviour using Isolation Forest.
- **AI-Powered Incident Reports:** Generate natural-language incident explanations using Google Gemini.
- **Live Alert System:** Transmit anomaly alerts through WebSockets.
- **Video Analysis:** Upload and process video footage through the integrated computer vision pipeline.
- **Incident Reports:** Generate and download investigation reports.
- **Investigation Dashboard:** Explore detected anomalies through a Streamlit interface.

---

## 🧠 System Architecture

NEMESIS-X follows a five-stage intelligence pipeline:

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
                   v
               YOLO11n
                   |
                   v
           OBJECT DETECTION
                   |
                   v
               BYTETRACK
                   |
                   v
            OBJECT TRACKING
                   |
                   v
           TEMPORAL ANALYSIS
                   |
                   v
          ANOMALY DETECTION
                   |
                   v
             FASTAPI ENGINE
                   |
             +-----+-----+
             |           |
             v           v
          GEMINI AI   LIVE ALERTS
             |           |
             v           |
       INCIDENT REPORT   |
             |           |
             +-----+-----+
                   |
                   v
          STREAMLIT DASHBOARD
```

---

## 🛠️ Technology Stack

### Computer Vision
- Python 3.12
- OpenCV
- Ultralytics YOLO11n
- ByteTrack

### Backend
- FastAPI
- Pydantic
- NumPy
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

## 📂 Project Structure

```text
NEMESIS_X/
│
├── Backend/
│   └── nemesis_backend/
│       ├── app/
│       │   ├── main.py
│       │   ├── schemas.py
│       │   ├── memory/
│       │   │   └── temporal_buffer.py
│       │   └── services/
│       │       ├── anomaly.py
│       │       ├── evidence_extractor.py
│       │       ├── llm_explainer.py
│       │       ├── pdf_report.py
│       │       └── reasoner.py
│       ├── test_feeder.py
│       └── requirements.txt
│
├── computer_vision/
│   ├── video_detector.py
│   ├── video_tracker.py
│   ├── video_processor.py
│   ├── process_and_ingest.py
│   ├── requirements.txt
│   └── yolo11n.pt
│
├── Frontend/
│   └── frontend/
│       └── frontend/
│           ├── app.py
│           ├── assets/
│           │   └── style.css
│           ├── components/
│           │   ├── api.py
│           │   └── ui.py
│           └── requirements.txt
│
├── sample_videos/
│   ├── assailant-steals-computer.mp4
│   ├── Thief.mp4
│   └── two-thieves.mp4
│
├── .gitignore
└── README.md
```

---

## 🎬 Sample Video Demonstrations

This repository includes three sample videos for testing the NEMESIS-X application.

| Sample Video | Purpose |
|---|---|
| `assailant-steals-computer.mp4` | Test video analysis and object tracking |
| `Thief.mp4` | Test video processing and temporal analysis |
| `two-thieves.mp4` | Test multi-object tracking and behavioural analysis |

You can find these videos in the `sample_videos/` directory.

Upload them through the Footage Review section of the application to test the processing pipeline.

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.12
- Git
- Windows 10/11 or compatible operating system
- Google Gemini API key (for Gemini-generated explanations)
- Internet connection for installing dependencies

**Hardware:** The current computer vision pipeline supports CPU-based processing. An NVIDIA GPU is not mandatory.

### 1. Clone the Repository

```powershell
git clone https://github.com/ShreyasPawar000/NEMESIS_X.git

cd NEMESIS_X
```

### 2. Configure the Backend

Open a terminal:

```powershell
cd Backend/nemesis_backend

python -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Configure the Gemini API key:

```powershell
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

Start the backend:

```powershell
python -m uvicorn app.main:app --reload
```

Backend:

`http://127.0.0.1:8000`

API documentation:

`http://127.0.0.1:8000/docs`

### 3. Configure Computer Vision

Open another terminal:

```powershell
cd computer_vision

python -m venv venv

.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

The YOLO11n model weights are included in the repository.

### 4. Configure the Frontend

Open another terminal from the project root:

```powershell
cd Frontend/frontend/frontend

pip install -r requirements.txt
```

Start Streamlit:

```powershell
python -m streamlit run app.py
```

Frontend:

`http://localhost:8501`

### 5. Start Analysing Videos

1. Start the backend.
2. Start the frontend.
3. Open the Streamlit dashboard.
4. Navigate to Footage Review.
5. Upload a sample video or your own footage.
6. Start video processing.
7. Review the available anomaly alerts and incident reports.

**Note:** Video processing requires the Computer Vision dependencies and YOLO model. Ensure the application's configured Python environment can access them.

---

## 🔍 Anomaly Detection

NEMESIS-X extracts temporal features from tracked objects, including:

- Object count
- Object count variation
- Pairwise spatial distance
- Object velocity
- Acceleration
- Direction changes
- Stationary ratio

These features are analysed using an Isolation Forest model to identify deviations from learned baseline behaviour.

When an anomaly score crosses the configured threshold, the system can generate an incident alert.

### Important Distinction

An anomaly score represents deviation from learned scene behaviour. It is not a probability of danger, proof of theft or evidence of criminal intent.

Human verification is required before interpreting an alert as a real-world incident.

---

## 🧪 Testing

NEMESIS-X includes a synthetic anomaly testing script.

Navigate to:

```powershell
cd Backend/nemesis_backend
```

Run:

```powershell
python test_feeder.py
```

The test sends baseline frames followed by simulated anomalous frames to the reasoning engine.

Expected behaviour:

- Calibrate the baseline.
- Detect the injected anomaly.
- Generate an alert.
- Prevent repeated alerts for the same active event.
- Generate an AI-assisted explanation when the Gemini service is available.

This is a controlled synthetic test, not a measurement of real-world detection accuracy.

---

## 🔮 Future Enhancements

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

- **Computer Vision:** Object detection, tracking and video processing.
- **Backend:** FastAPI, temporal memory, anomaly detection and AI integration.
- **Frontend:** Streamlit dashboard, investigation interface and report downloads.

---

## ⚠️ Limitations

- Detection performance depends on video quality and computational resources.
- CPU-based video processing may be slower than GPU-based processing.
- Anomaly detection depends on baseline calibration.
- Gemini explanations depend on API availability and quota.
- Statistical anomalies do not automatically identify specific criminal activities.
- The NEMESIS AI chat interface is currently a prototype.

---

## 📜 Disclaimer

NEMESIS-X is an experimental academic and hackathon project developed for video intelligence research and demonstration.

It is not a replacement for professional surveillance, security investigation or human decision-making.

---

### ◈ NEMESIS-X

**Don't just watch videos, understand them.**
