# CrowdGuard 🛡️

**AI-powered real-time crowd safety and stampede detection system**

CrowdGuard uses computer vision and machine learning to monitor crowd density, detect dangerous anomalies, and alert operators before incidents escalate. Built as an academic project demonstrating end-to-end ML system design — from YOLOv8 person detection to federated learning simulation.

---

## ✨ Features

### Real-time Detection
- **Live person detection** via YOLOv8n with bounding boxes and track IDs
- **Centroid tracking** — persistent person IDs across frames for velocity analysis
- **10×10 density heatmap** overlaid on the live video feed
- **Composite risk scoring** (0–100) → SAFE / CAUTION / WARNING / CRITICAL

### 6 Anomaly Types
| Anomaly | Description |
|---|---|
| `HIGH_DENSITY` | Crowd density exceeds safe threshold |
| `RAPID_MOVEMENT` | Sudden high-velocity crowd movement |
| `SUDDEN_DISPERSAL` | Sharp drop in person count (possible incident) |
| `CROWD_SURGE` | Directional pressure building in crowd |
| `STATIONARY_CROWD` | Large group with near-zero velocity |
| `FIGHTING` | High bounding-box overlap + high velocity |

### Alert Management
- Real-time alert push via WebSocket
- Alert deduplication with configurable cooldown
- Filter, bulk dismiss, and CSV export
- Full alert history with confidence scores

### Analytics Dashboard
- KPI cards with time-range filtering (15 min → 24 hr)
- Density over time, risk score over time, alert frequency by type, person count distribution

### Federated Learning Simulation
- 3 virtual nodes (Mall Entrance, Food Court, Parking Lot)
- FedAvg aggregation over 10 rounds
- Accuracy convergence from ~63% → ~89%
- Privacy-preserving: only model weights shared, never raw data

### Configurable Settings
- All detection thresholds adjustable via UI
- Hot-reload: changes apply to active session within 1 second

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                       │
│  Dashboard │ Alerts │ Analytics │ Settings │ Federated  │
└──────────────────────┬──────────────────────────────────┘
                       │ REST API + WebSocket
┌──────────────────────▼──────────────────────────────────┐
│                   FastAPI Backend                       │
│  /api/video  /api/alerts  /api/analytics  /api/settings │
│  /api/federated                           /ws           │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                   ML Pipeline                           │
│  YOLOv8n → Centroid Tracker → Heatmap → Anomaly Engine  │
│                           → Risk Scorer → Alert Manager │
└──────────────────────┬──────────────────────────────────┘
                       │
              SQLite (crowdguard.db)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, Vite, TypeScript, TailwindCSS, Zustand, Framer Motion, Recharts |
| Backend | FastAPI, Uvicorn, SQLAlchemy, SQLite |
| ML | YOLOv8n (Ultralytics), OpenCV, PyTorch, NumPy |
| Training | VisDrone2019-DET dataset, AdamW, Soft-NMS |
| Real-time | WebSocket (native browser + FastAPI) |

---

## 📊 Model Performance (VisDrone val set)

| Model | mAP@0.5 | Precision | Recall | F1 |
|---|---|---|---|---|
| Pretrained YOLOv8n (COCO baseline) | ~0.08 | ~0.45 | ~0.18 | ~0.26 |
| Fine-tuned YOLOv8n (VisDrone) | ~0.28 | ~0.62 | ~0.38 | ~0.47 |
| Fine-tuned + Soft-NMS (σ=0.5) | ~0.30 | ~0.64 | ~0.40 | ~0.49 |

**+250% mAP@0.5 improvement** from fine-tuning on VisDrone over the pretrained COCO baseline.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Webcam (for live detection)

### 1. Clone the repo
```bash
git clone https://github.com/your-username/crowdguard.git
cd crowdguard
```

### 2. Backend setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Start the backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend setup
```bash
cd frontend
npm install
npm run dev
```

### 4. Open the app
Navigate to **http://localhost:5173**

### 5. (Optional) Seed demo data
```bash
# Add sample alerts and density logs for the analytics dashboard
python seed_dummy_data.py
python seed_density.py
```

---

## 🗂️ Project Structure

```
crowdguard/
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # Configuration management
│   ├── database.py              # SQLAlchemy setup
│   ├── models/                  # ORM models (Alert, Session, DensityLog, Settings)
│   ├── routers/                 # API route handlers
│   │   ├── video.py             # Video session management
│   │   ├── alerts.py            # Alert CRUD
│   │   ├── analytics.py         # Analytics endpoints
│   │   ├── settings.py          # Settings hot-reload
│   │   ├── websocket.py         # WebSocket broadcast
│   │   └── federated.py         # Federated learning simulation
│   ├── services/                # Business logic
│   │   ├── alert_manager.py     # Alert deduplication
│   │   ├── session_manager.py   # Session lifecycle
│   │   └── cleanup_service.py   # Scheduled cleanup
│   ├── ml/                      # ML pipeline
│   │   ├── detector.py          # YOLOv8 person detection
│   │   ├── tracker.py           # Centroid tracker
│   │   ├── heatmap.py           # Density heatmap
│   │   ├── anomaly_engine.py    # 6-type anomaly detection
│   │   ├── risk_scorer.py       # Composite risk scoring
│   │   ├── video_processor.py   # Pipeline orchestrator
│   │   ├── detector_improved.py # Soft-NMS implementation
│   │   ├── dataset_prep.py      # VisDrone → YOLO conversion
│   │   ├── evaluate_baseline.py # Baseline evaluation
│   │   ├── train.py             # Fine-tuning script
│   │   ├── compare_models.py    # Model comparison
│   │   └── plot_metrics.py      # Results visualisation
│   └── utils/
│       └── logger.py            # Coloured logging
├── frontend/
│   └── src/
│       ├── pages/               # Dashboard, Alerts, Analytics, Settings
│       ├── components/          # feed/, alert/, analytics/, federated/, layout/
│       ├── hooks/               # useWebSocket, useAlerts, useVideoControl
│       ├── store/               # Zustand slices (video, alert, analytics, ws)
│       └── utils/               # API client, formatters, constants
├── seed_dummy_data.py           # Seed alerts + sessions
├── seed_density.py              # Seed density logs (24h)
├── train_model.sh               # Full training pipeline script
├── .env.example                 # Environment variable template
└── .gitignore
```

---

## 🎓 Training Pipeline (Academic)

To reproduce the VisDrone training results:

```bash
# 1. Download VisDrone2019-DET dataset and place at datasets/VisDrone/
# 2. Run the full pipeline
bash train_model.sh
```

This runs in order:
1. `dataset_prep.py` — convert VisDrone annotations to YOLO format
2. `evaluate_baseline.py` — evaluate pretrained YOLOv8n baseline
3. `train.py` — fine-tune on VisDrone (50 epochs, AdamW)
4. `compare_models.py` — compare all 3 model variants
5. `plot_metrics.py` — generate result plots in `results/plots/`

Results saved to `results/baseline_metrics.json`, `results/training_metrics.json`, `results/comparison_results.json`.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/video/start` | Start a detection session |
| `POST` | `/api/video/stop` | Stop active session |
| `GET` | `/api/video/status` | Session status |
| `POST` | `/api/video/upload` | Upload video file |
| `GET` | `/api/alerts` | List alerts (filterable, paginated) |
| `PUT` | `/api/alerts/{id}/dismiss` | Dismiss an alert |
| `GET` | `/api/analytics/kpis?minutes=60` | KPI metrics for time window |
| `GET` | `/api/analytics/density-timeseries?minutes=60` | Density time series |
| `GET` | `/api/analytics/risk-timeseries?minutes=60` | Risk score time series |
| `GET` | `/api/settings` | Get current settings |
| `PUT` | `/api/settings` | Update settings (hot-reload) |
| `POST` | `/api/federated/simulate` | Start FL simulation |
| `GET` | `/api/federated/status` | FL simulation status |
| `WS` | `/ws` | WebSocket for real-time frames + alerts |

Full interactive docs at **http://localhost:8000/docs** (Swagger UI).

---

## 🔮 Future Work

- **Mobile app** — React Native for security personnel alerts
- **IoT / RTSP camera support** — direct IP camera integration
- **Crowd flow prediction** — LSTM model for 5–10 min density forecasting
- **Multi-camera aggregation** — unified risk score across a venue
- **Real federated learning** — replace simulation with Flower (flwr) framework
- **Video clip archiving** — auto-save 10 s clips around each alert event
- **Audio anomaly detection** — microphone input for stampede sound detection
- **Cloud deployment** — Docker + AWS/GCP with auto-scaling

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">Built with ❤️ for crowd safety research</p>
