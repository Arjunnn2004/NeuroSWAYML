# NeuroSWAYML 🧠

> **ML-powered real-time fall detection and gait analysis — Elderly Fall Risk (URFD video dataset)**

NeuroSWAYML is the machine learning upgrade to the original rule-based NeuroSWAY system. It replaces hand-tuned thresholds with a **4-model ensemble** (RF + XGBoost + LSTM + Autoencoder) trained on the URFD real-person video dataset, achieving **85–90% validation accuracy** on elderly fall risk classification. The codebase also includes training pipelines for three additional clinical domains (Neurodegenerative, Intoxication, Congenital) that are ready to use but not yet wired into the live app.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [ML Architecture](#ml-architecture)
- [Analysis Domains](#analysis-domains)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Feature Reference](#feature-reference)
- [Roadmap](#roadmap)
- [Credits](#credits)

---

## Overview

| | NeuroSWAY (original) | NeuroSWAYML |
|---|---|---|
| **Detection method** | Fixed thresholds | 4-model ML ensemble (RF + XGB + LSTM + AE) |
| **Active app domain** | 1 (Parkinson's only) | Elderly Gait & Fall Risk (URFD video) |
| **Personalization** | None | Autoencoder personal baseline (90-frame calibration) |
| **Temporal analysis** | `np.std()` over 60 frames | LSTM over 60-frame sliding windows |
| **Training data** | None | Real-person video (URFD: 30 falls + 40 ADL sequences) |
| **Fall detection** | Rule-based | ML classifier trained on real fall videos |
| **Camera processing** | Single thread | Threaded MediaPipe inference pipeline |
| **Additional domains** | N/A | 3 more trained (Neuro · Intox · Congenital) — not in app yet |
| **Validation accuracy** | N/A | 85–90% on held-out URFD data |

---

## Features

### Real-Time Detection
- 🦵 **Gait asymmetry** — left vs right stride imbalance
- 🌀 **Sway analysis** — lateral/anterior-posterior body sway index
- 👣 **Heel-toe pattern** — toe walking, shuffling, foot clearance
- 📐 **Torso angle** — forward lean, postural instability
- ⏱️ **Cadence** — steps per minute
- 🔄 **Stride variability (CV)** — coefficient of variation across strides
- 🤝 **Gait symmetry index** — 0–1 score (1.0 = perfect symmetry)
- 🦴 **Knee angle differential** — left vs right knee flexion difference
- 🚨 **Fall risk scoring** — continuous 0–1 probability
- 🧠 **Anomaly detection** — deviation from personal calibrated baseline

### App Panel (Live Overlay)
```
┌─────────────────────────────────────────┐
│  NeuroSWAYML — Elderly Fall Risk        │
│  [NORMAL_GAIT]  Risk: ████░░░░ 42%      │
├─────────────────────────────────────────┤
│  LIVE SENSOR DATA                       │
│  Sway Idx    : 1.23                     │
│  Leg Ratio   : 0.98                     │
│  Heel-Toe L  : 2.1                      │
│  Heel-Toe R  : 1.8                      │
│  Torso Angle : 8.3°                     │
│  Symmetry    : 0.91                     │
│  Stride CV   : 0.09                     │
│  Cadence     : 112 spm                  │
│  Knee Diff   : 4.2°                     │
├─────────────────────────────────────────┤
│  ML ANALYSIS                            │
│  Gait Class  : NORMAL_GAIT              │
│  LSTM        : NORMAL_GAIT              │
│  Anomaly     : 0.22 (normal)            │
│  Fall Risk   : 12%                      │
├─────────────────────────────────────────┤
│  FPS: 28  |  Alerts: 0                  │
└─────────────────────────────────────────┘
```

### Controls
| Key | Action |
|---|---|
| `Q` | Quit application |
| `R` | Recalibrate personal baseline |
| `D` | Toggle debug overlay |
| `S` | Save current session log |

---

## ML Architecture

NeuroSWAYML uses a **4-model ensemble** — the app runs the Elderly domain. The `DomainManager` infrastructure supports all 4 domains but only Elderly is loaded at startup.

```
Camera Frame
     │
     ▼
MediaPipe Pose (3D world landmarks)   ← ThreadedPoseEngine
     │
     ▼
FeatureExtractor (30 biomechanical features)
     │
     ▼
MLAnalyzer → DomainManager (Elderly domain active)
     │
     ▼  (Elderly model stack)
     │
     ├──► GaitClassifier  (Random Forest + XGBoost ensemble)
     │         └── Domain-specific class labels
     │             Output: class index + probability
     │
     ├──► LSTM Sequence Model  (2-layer, hidden=128)
     │         └── Input: last 60 frames × 30 features
     │             Output: temporal gait class
     │
     ├──► Autoencoder  (30→32→16→8→16→32→30)
     │         └── Calibrated to YOUR normal gait (90 frames)
     │             Output: reconstruction error score
     │             High score = deviation from personal baseline
     │
     └──► Ensemble Vote  (weighted combination)
               └── rf=0.30 · xgb=0.30 · lstm=0.30 · ae=0.10
                        │
                        ▼
               Final Risk Score + Class Label + Alert
```

### Model Details

| Model | Algorithm | Input | Output |
|---|---|---|---|
| **GaitClassifier** | Random Forest + XGBoost | 30 features | 3-class label + probability |
| **LSTM** | 2-layer LSTM (hidden=128, dropout=0.3) | 60 frames × 30 features | 3-class label |
| **Autoencoder** | MLP 30→32→16→8→16→32→30 | 30 features | Anomaly score (reconstruction error) |
| **Ensemble** | Weighted vote (RF+XGB+LSTM+AE) | All model outputs | Final risk score 0–1 |

### Ensemble Weights
```python
ensemble_score = (
    0.30 × RandomForest_score +
    0.30 × XGBoost_score      +
    0.30 × LSTM_score         +
    0.10 × Autoencoder_score
)

# Risk thresholds (config_ml.json)
warning_threshold  = 0.40   → WARNING label
critical_threshold = 0.65   → HIGH_RISK label
```

---

## Analysis Domains

NeuroSWAYML has training pipelines for four clinical domains. **Only the Elderly domain is active in `app_ml.py`** — the other three are fully trained and saved but not yet wired into the live app.

| Status | Domain | Class Labels | Dataset |
|---|---|---|---|
| ✅ **Active in app** | **Elderly Gait & Fall Risk** | NORMAL_GAIT · MILD_FALL_RISK · HIGH_FALL_RISK | URFD (video) |
| 🔧 Trained, not in app | Neurodegenerative (PD/ALS/HD) | NORMAL · WARNING · HIGH_RISK | gaitpdb + gaitndd (PhysioNet) |
| 🔧 Trained, not in app | Intoxication / Ataxia | SOBER · MILD_IMPAIRMENT · INTOXICATED | HBEDB (PhysioNet) |
| 🔧 Trained, not in app | Congenital / Birth Disorder | NORMAL · MILD_DISORDER · SEVERE_DISORDER | GaitRec (figshare) |

Models are saved in separate subdirectories:
```
saved_models/
  elderly/         ← loaded by app_ml.py
  neuro/           ← trained, not loaded by app
  intoxication/    ← trained, not loaded by app
  congenital/      ← trained, not loaded by app
```

Each subdirectory contains: `gait_classifier.pkl`, `lstm_model.pt`, `autoencoder.pkl`, `scaler.pkl`, `training_report.json`

---

## Datasets

### Domain 1 — Neurodegenerative: PhysioNet (gaitpdb + gaitndd)

**gaitpdb — Gait in Parkinson's Disease**
| Property | Value |
|---|---|
| **Source** | [physionet.org/content/gaitpdb/1.0.0](https://physionet.org/content/gaitpdb/1.0.0/) |
| **Subjects** | 93 (73 PD + 20 healthy controls) |
| **Recordings** | 310 |
| **Format** | 19-column VGRF `.txt` (16 force sensors + timestamps) |
| **Sample rate** | 100 Hz |
| **Why used** | Gold standard for Parkinson's biomechanics — real PD shuffling, freezing-of-gait, asymmetry at 3 speeds |

**gaitndd — Gait in Neurodegenerative Disease**
| Property | Value |
|---|---|
| **Source** | [physionet.org/content/gaitndd/1.0.0](https://physionet.org/content/gaitndd/1.0.0/) |
| **Subjects** | 64 (15 ALS + 20 Huntington's + 13 PD + 16 controls) |
| **Format** | 13-column stride interval `.ts` files |
| **Why used** | Differentiates 3 distinct neurological gait patterns |

**Label mapping:**
```
Al → ALS         → HIGH_RISK (2)   # Upper motor neuron weakness
Hd → Huntington  → HIGH_RISK (2)   # Choreiform movement disorder
Pa → Parkinson   → HIGH_RISK (2)   # Dopamine-deficient basal ganglia
Co → Control     → NORMAL    (0)   # Healthy baseline
```

---

### Domain 2 — Elderly: URFD Video Dataset

| Property | Value |
|---|---|
| **Source** | [fenix.ur.edu.pl/~mkepski/ds/uf.html](http://fenix.ur.edu.pl/~mkepski/ds/uf.html) |
| **Sequences** | 70 (30 fall + 40 ADL / normal activity) |
| **Format** | RGB video frames → MediaPipe Pose → 30-D feature vectors |
| **Size** | ~240 MB (auto-downloaded) |
| **Why used** | Real-person video — feature vectors are **identical** to live inference, eliminating the VGRF-to-camera domain gap |

**Label mapping:**
```
adl-* sequences                    → 0 NORMAL_GAIT
fall-* first 60% of frames         → 1 MILD_FALL_RISK   (pre-fall)
fall-* last  40% of frames         → 2 HIGH_FALL_RISK   (active fall)
```

Download: `python data/downloader.py --domain elderly`

---

### Domain 3 — Intoxication: HBEDB (PhysioNet)

| Property | Value |
|---|---|
| **Source** | [physionet.org/content/hbedb/1.0.0](https://physionet.org/content/hbedb/1.0.0/) |
| **Subjects** | 163 |
| **Format** | COP force-platform stabilography, Romberg protocols |
| **Why used** | Captures balance impairment across controlled conditions — maps directly to sober / impaired / ataxic states |

**Condition mapping:**
```
Eyes-Open firm surface              → 0 SOBER
Eyes-Closed firm surface            → 1 MILD_IMPAIRMENT
Foam / tandem / Eyes-Closed-foam    → 2 INTOXICATED / ATAXIA
```

Download: `python data/downloader.py --domain intox`

---

### Domain 4 — Congenital: GaitRec (figshare)

| Property | Value |
|---|---|
| **Source** | [figshare DOI: 10.6084/m9.figshare.13598962.v1](https://doi.org/10.6084/m9.figshare.13598962.v1) |
| **License** | CC-BY 4.0 |
| **Subjects** | 2,084 (healthy controls + 7 pathology groups) |
| **Format** | 17-column bilateral GRF data at 1000 Hz |
| **Size** | ~2.3 GB (manual download required) |
| **Why used** | Largest available GRF dataset with real joint disorder labels |

**Group mapping:**
```
CTL                           → 0 NORMAL
BACK, ANKLE                   → 1 MILD_DISORDER
HIP, KNEE, NEURO, CP, DS, SB  → 2 SEVERE_DISORDER
```

Manual download instructions: `python data/downloader.py --domain congen`

---

## Project Structure

```
NeuroSWAYML/
│
├── app_ml.py                      # Main entry point — runs Elderly Fall Risk (URFD)
├── config_ml.json                 # All hyperparameters, thresholds, domain config
├── requirements.txt
├── README.md
│
├── core/
│   ├── domain_manager.py          # Manages all 4 domain model stacks (Elderly active)
│   ├── ml_analyzer.py             # Live inference pipeline + OpenCV overlay
│   └── pose_engine.py             # Threaded MediaPipe pose estimation
│
├── data/
│   ├── dataset_loader.py          # PhysioNet VGRF multi-format parser
│   ├── feature_extractor.py       # MediaPipe landmarks → 30 biomechanical features
│   ├── downloader.py              # Auto/manual dataset downloader (all domains)
│   ├── loaders/
│   │   ├── urfd_loader.py         # URFD video → MediaPipe → feature cache
│   │   ├── elderly_loader.py      # Elderly domain data adapter
│   │   ├── intoxication_loader.py # HBEDB COP data parser
│   │   └── congenital_loader.py   # GaitRec GRF data parser
│   ├── physionet/
│   │   ├── gaitpdb/               # ← Place extracted gaitpdb here (Domain 1)
│   │   └── gaitndd/               # ← Place extracted gaitndd here (Domain 1)
│   └── urfd/                      # ← Auto-downloaded URFD frames (Domain 2)
│
├── models/
│   ├── gait_classifier.py         # Random Forest + XGBoost ensemble
│   ├── lstm_model.py              # 2-layer LSTM (PyTorch)
│   ├── autoencoder.py             # MLP autoencoder anomaly detector
│   ├── ensemble.py                # Weighted vote → final risk score
│   └── domain_classifier.py      # DomainModel — per-domain model stack wrapper
│
├── training/
│   ├── train_all.py               # Train all 4 domains in one command
│   ├── train_elderly.py           # Elderly domain (URFD, MediaPipe processing)
│   ├── train_intoxication.py      # Intoxication domain (HBEDB)
│   └── train_congenital.py        # Congenital domain (GaitRec)
│
├── saved_models/
│   ├── pose_landmarker_lite.task  # MediaPipe pose model
│   ├── neuro/                     # Domain 1 model files
│   ├── elderly/                   # Domain 2 model files
│   ├── intoxication/              # Domain 3 model files
│   └── congenital/                # Domain 4 model files
│
├── ml_logs/                       # Session logs (saved with S key)
└── ml_fall_detections/            # Fall event frame captures
```

Each domain's `saved_models/<domain>/` directory contains:
```
gait_classifier.pkl   scaler.pkl
lstm_model.pt         training_report.json
autoencoder.pkl
```

---

## Installation

### Prerequisites
- Python 3.9–3.12
- Webcam or video file
- Windows / Linux / macOS

### Step 1 — Clone the repository
```bash
git clone https://github.com/Arjunnn2004/NeuroSWAYML.git
cd NeuroSWAYML
```

### Step 2 — Create virtual environment and install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

### Step 3 — Download datasets

**Domains 1 & 3 (Neuro + Intoxication) — PhysioNet (free account required)**
1. Register at [physionet.org/register](https://physionet.org/register/)
2. Run the downloader (prompts for your PhysioNet credentials):
```bash
python data/downloader.py --domain neuro    # gaitpdb + gaitndd → data/physionet/
python data/downloader.py --domain intox    # HBEDB → data/physionet/hbedb/
```

**Domain 2 (Elderly) — URFD, auto-downloaded (~240 MB)**
```bash
python data/downloader.py --domain elderly  # URFD frames → data/urfd/
```

**Domain 4 (Congenital) — GaitRec, manual download required (~2.3 GB)**
```bash
python data/downloader.py --domain congen   # prints download instructions
```
Then place the extracted GaitRec folder at `data/gaitrec/`.

**Check what is present:**
```bash
python data/downloader.py --check
```

### Step 4 — Train the Elderly model (required to run the app)
```bash
python training/train_elderly.py
```
This downloads URFD (~240 MB) if not already present, runs MediaPipe on all 70 sequences, and saves models to `saved_models/elderly/`. Takes ~3–5 minutes.

To also train the other domains (optional — not used by the app yet):
```bash
python training/train_all.py                    # all 4 domains
python training/train_all.py --domain neuro     # neurodegenerative only
python training/train_all.py --domain intox     # intoxication only
python training/train_all.py --domain congen    # congenital only
python training/train_all.py --skip-missing     # skip domains with no data
```

Training takes ~2–5 minutes per domain. Models are saved to `saved_models/<domain>/`.

### Step 5 — Run the app
```bash
python app_ml.py                      # default webcam (index 0)
python app_ml.py --source 1           # alternate camera
python app_ml.py --source video.mp4   # video file
python app_ml.py --no-thread          # disable inference threading
```

---

## Usage

### First Run — Personal Calibration
1. Launch the app
2. Walk or stand naturally in front of the camera for ~3 seconds (90 frames)
3. Calibration completes automatically — or press **`R`** to reset it manually
4. The autoencoder learns your personal baseline and flags deviations specific to you

### Interpreting Results

**Risk Level Colors:**
```
GREEN   → Normal       (score < 0.40)
ORANGE  → Warning      (score 0.40–0.65)
RED     → High Risk    (score > 0.65)
```

**Sway Index:**
```
< 1.5   → Minimal sway (normal)
1.5–2.5 → Moderate sway (monitor)
> 2.5   → Excessive sway (alert)
```

**Leg Ratio:**
```
0.95–1.05 → Symmetric (normal)
< 0.95    → Left leg shorter stride (alert)
> 1.05    → Right leg shorter stride (alert)
```

---

## Model Performance

### Elderly Gait & Fall Risk — URFD Video (active in app)

| Model | Validation Accuracy | Notes |
|---|---|---|
| **GaitClassifier** (RF+XGB) | ~88–92% | All 30 features populated via real MediaPipe inference |
| **LSTM** | ~75–85% | Full 30-feature sequences from real video |
| **Autoencoder** | N/A | Anomaly threshold at 95th percentile of normal samples |
| **Ensemble** | ~85–90% | Weighted RF+XGB+LSTM+AE combination |

> Because URFD is processed with MediaPipe (the same pipeline as live inference), all 30 features are populated during training — giving the LSTM full context. This is the key advantage over VGRF-based datasets which only populate ~5 features.

### Other Trained Domains (not loaded by app_ml.py)

**Neurodegenerative (gaitpdb + gaitndd — VGRF)**
| Model | Validation Accuracy | Notes |
|---|---|---|
| **GaitClassifier** (RF+XGB) | **90.3%** | Primary classifier |
| **LSTM** | 33–55% | Only 5/30 features populated from VGRF force-plate records |
| **Ensemble** | **83%** | LSTM weight partially offsets low feature coverage |

**Intoxication / Congenital** — performance varies with dataset size. Run `training/train_*.py --dry-run` to inspect dataset statistics.

---

## Feature Reference

All 30 features extracted per frame by `data/feature_extractor.py`:

| # | Feature | Source | Clinical meaning |
|---|---|---|---|
| 1 | `sway_index` | Camera | Lateral body sway (std of hip position) |
| 2 | `leg_ratio` | Camera | L/R stride length asymmetry |
| 3 | `heel_toe_l` | Camera | Left foot heel-to-toe height diff |
| 4 | `heel_toe_r` | Camera | Right foot heel-to-toe height diff |
| 5 | `torso_angle` | Camera | Forward lean angle (degrees) |
| 6 | `symmetry` | Camera + Dataset | Gait symmetry index (0–1) |
| 7 | `stride_cv` | Camera + Dataset | Stride variability coefficient |
| 8 | `cadence` | Camera + Dataset | Steps per minute |
| 9 | `knee_angle_l` | Camera | Left knee flexion angle |
| 10 | `knee_angle_r` | Camera | Right knee flexion angle |
| 11 | `knee_diff` | Camera | L vs R knee angle difference |
| 12 | `hip_angle` | Camera | Hip flexion/extension |
| 13 | `shoulder_align` | Camera | Shoulder level symmetry |
| 14 | `step_width` | Camera | Lateral distance between feet |
| 15 | `swing_l` | Dataset | Left swing phase fraction |
| 16 | `swing_r` | Dataset | Right swing phase fraction |
| 17–30 | VGRF / GRF features | Dataset | Force plate biomechanics (populated from clinical datasets) |

> Features 1–14 are populated every frame from the live camera. Features 15–30 are populated during clinical dataset training and supplemented at runtime when detectable from pose landmarks.

---

## Roadmap

- [x] **URFD video dataset** — real fall events with full 30-feature coverage for the LSTM
- [x] **Elderly domain** — RF+XGB+LSTM+AE ensemble active in app_ml.py
- [x] **Multi-domain training pipelines** — Neuro, Intoxication, Congenital trainers complete
- [x] **GaitRec domain** — 2,084-subject congenital disorder classifier (trained, not in app)
- [x] **HBEDB domain** — balance impairment / ataxia classifier (trained, not in app)
- [ ] **Wire up domain switching in app** — connect keys 1–4 to DomainManager in app_ml.py
- [ ] **CMU MoCap integration** — 3D joint angle sequences for LSTM training on neuro domain
- [ ] **ONNX export** — run inference on mobile / edge devices
- [ ] **Session history dashboard** — trend analysis over weeks/months
- [ ] **Freeze-of-gait detection** — specific PD episode detector
- [ ] **Audio alerts** — spoken warnings for visually impaired users
- [ ] **Multi-person tracking** — clinical setting with multiple patients
- [ ] **Report generation** — PDF gait analysis report per session

---

## Credits

### Datasets
- **Hausdorff JM et al. (2000)**: *Gait in Aging and Disease* — PhysioNet gaitpdb
- **Hausdorff JM et al. (2000)**: *Gait Dynamics in Neurodegenerative Disease* — PhysioNet gaitndd
- **Goldberger AL et al. (2000)**: *PhysioBank, PhysioToolkit, PhysioNet* — Circulation 101(23)
- **Kepski M & Kwolek B**: *URFD — University of Rzeszów Fall Detection Dataset*
- **Horst F et al. (2021)**: *GaitRec* — figshare DOI: 10.6084/m9.figshare.13598962.v1 (CC-BY 4.0)
- **HBEDB**: *Human Balance Evaluation Database* — PhysioNet hbedb/1.0.0

### Libraries
- [MediaPipe](https://mediapipe.dev/) — Real-time pose estimation
- [OpenCV](https://opencv.org/) — Camera capture and overlay rendering
- [scikit-learn](https://scikit-learn.org/) — Random Forest, autoencoder
- [XGBoost](https://xgboost.readthedocs.io/) — Gradient boosted classifier
- [PyTorch](https://pytorch.org/) — LSTM sequence model
- [wfdb](https://wfdb.readthedocs.io/) — PhysioNet waveform database reader

---

## License

This project is for **research and educational purposes only**.  
Not intended for clinical diagnosis. Always consult a medical professional.

---

*Built as part of the NeuroSWAY project — multi-domain ML upgrade for clinical gait monitoring*
