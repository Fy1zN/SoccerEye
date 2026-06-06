# ⚽ SoccerEye

### AI-Powered Football Match Analysis & Player Tracking Platform

Track players, referees, and ball movement in football matches using Computer Vision, YOLOv8, and ByteTrack.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![Computer%20Vision-OpenCV-green)
![Tracking-ByteTrack-red)
![Status-Active-brightgreen)
![License-MIT-yellow)

---

# 📌 Overview

SoccerEye is an AI-driven football analytics system that automatically detects and tracks players, referees, and the ball from match footage using state-of-the-art computer vision techniques.

The platform processes football videos frame-by-frame, generates persistent player IDs, and visualizes tracking results through annotated match footage.

---

# 🚀 Features

### 🎯 Object Detection
- Player Detection
- Referee Detection
- Ball Detection
- Goalkeeper Detection

### 📍 Multi-Object Tracking
- ByteTrack Integration
- Persistent Player IDs
- Real-Time Object Association
- Track Management Across Frames

### 📊 Match Visualization
- Player Tracking Overlays
- Ball Position Indicators
- Referee Identification
- Annotated Match Video Generation

### ⚡ Performance Optimizations
- Batch Inference Processing
- YOLOv8-based Detection Pipeline
- Efficient Video Frame Handling

---

# 🏗️ Project Structure

```bash
SoccerEye/
│
├── trackers/
│   ├── __init__.py
│   └── tracker.py
│
├── utils/
│   ├── __init__.py
│   ├── bbox_utils.py
│   └── video_utils.py
│
├── models/
│   └── best.pt
│
├── Input_data/
│   └── input_videos
│
├── output/
│   └── generated_results
│
├── training/
│
├── inference.ipynb
├── main.py
└── requirements.txt
```

---

# 🧠 Tech Stack

### Computer Vision
- OpenCV
- NumPy

### Deep Learning
- YOLOv8 (Ultralytics)

### Object Tracking
- ByteTrack
- Supervision

### Development
- Python 3.10+
- Jupyter Notebook
- VS Code

---

# 🔄 Pipeline

```text
Input Match Video
        │
        ▼
Frame Extraction
        │
        ▼
YOLOv8 Detection
        │
        ▼
ByteTrack Tracking
        │
        ▼
Object Association
        │
        ▼
Player ID Assignment
        │
        ▼
Frame Annotation
        │
        ▼
Output Video Generation
```

---

# 📈 Current Capabilities

✅ Player Tracking

✅ Ball Tracking

✅ Referee Tracking

✅ Persistent IDs

✅ Video Annotation

✅ Batch Inference

---

# 🔮 Roadmap

### Phase 1 (Completed)
- Object Detection
- Player Tracking
- Ball Tracking
- Video Annotation

### Phase 2
- Team Classification
- Ball Possession Analysis
- Player Speed Estimation
- Distance Covered Metrics

### Phase 3
- Tactical Heatmaps
- Passing Network Analysis
- Expected Threat (xT)
- Match Intelligence Dashboard

### Phase 4
- Real-Time Match Processing
- Multi-Camera Tracking
- Advanced Football Analytics Platform

---

# 🛠️ Installation

```bash
git clone https://github.com/Fy1zN/SoccerEye.git

cd SoccerEye

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

---

# ▶️ Usage

```bash
python main.py
```

The generated annotated video will be saved inside:

```bash
output/
```

---

# 📸 Sample Output

SoccerEye generates annotated football footage featuring:

- Player IDs
- Ball Tracking
- Referee Tracking
- Visual Tracking Markers
- Match Analytics Overlays

---

# 🤝 Contributions

Contributions, feature requests, and improvements are welcome.

Feel free to open issues or submit pull requests.

---

# 📜 License

This project is licensed under the MIT License.

---

### Developed by Krish Malhotra ⚽
