# ⚽ SoccerEye

### AI-Powered Football Match Analysis & Player Tracking Platform

Track players, referees, and ball movement in football matches using Computer Vision, YOLOv8, and ByteTrack.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6F00?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![ByteTrack](https://img.shields.io/badge/ByteTrack-Tracking-red?style=for-the-badge)
![OpenCV Optical Flow](https://img.shields.io/badge/Optical%20Flow-Camera%20Motion-success?style=for-the-badge)
![MIT License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

# 📌 Overview

SoccerEye is an AI-powered football analytics platform that transforms raw match footage into actionable insights using Computer Vision and Deep Learning.

The system automatically detects players, referees, goalkeepers, and the ball, assigns persistent tracking IDs, classifies teams, estimates camera movement, and calculates real-time ball possession statistics.

Designed as a modular football analysis pipeline, SoccerEye serves as a foundation for advanced analytics such as player speed estimation, heatmaps, passing networks, and tactical intelligence systems.

---

# 🚀 Features

## 🎯 Object Detection

* Player Detection
* Goalkeeper Detection
* Referee Detection
* Ball Detection
* YOLOv8-based Detection Pipeline

---

## 📍 Multi-Object Tracking

* ByteTrack Integration
* Persistent Player IDs
* Multi-Frame Object Association
* Real-Time Object Tracking
* Stable Identity Management

---

## 👕 Team Classification

* Automatic Team Assignment
* Jersey Color Analysis
* Team Color Extraction
* Team-Based Player Identification

---

## ⚽ Ball Possession Analysis

* Ball-to-Player Assignment
* Possession Holder Detection
* Team Possession Tracking
* Real-Time Possession Percentage Calculation
* Live Possession Overlay

---

## 🎥 Camera Movement Estimation

* Optical Flow Tracking
* Feature Point Tracking
* Frame-to-Frame Camera Motion Detection
* Camera Shift Compensation Support

---

## 📊 Match Visualization

* Player Tracking Overlays
* Team Color Visualization
* Ball Tracking Indicators
* Ball Possession Markers
* Team Possession Dashboard
* Camera Movement Dashboard
* Annotated Match Video Generation

---

## ⚡ Performance Optimizations

* Batch Inference Processing
* Stub-Based Caching
* Ball Position Interpolation
* Efficient Video Frame Processing
* Reduced Reprocessing Time

---

# 🏗️ Project Structure

```bash
SoccerEye/
│
├── camera_movement_estimator/
│   ├── __init__.py
│   └── camera_movement_estimator.py
│
├── player_ball_assigner/
│   ├── __init__.py
│   └── player_ball_assigner.py
│
├── team_assigner/
│   ├── __init__.py
│   └── team_assigner.py
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
│
├── output/
│
├── stubs/
│
├── inference.ipynb
├── main.py
├── requirements.txt
└── README.md
```

---

# 🧠 Tech Stack

## Computer Vision

* OpenCV
* Optical Flow
* Feature Tracking

## Deep Learning

* YOLOv8 (Ultralytics)

## Object Tracking

* ByteTrack
* Supervision

## Data Processing

* NumPy
* Pandas

## Development

* Python 3.10+
* Jupyter Notebook
* VS Code

---

# 🔄 Processing Pipeline

```text
Input Football Video
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
Player ID Assignment
          │
          ▼
Team Classification
          │
          ▼
Ball Assignment
          │
          ▼
Ball Possession Analysis
          │
          ▼
Camera Movement Estimation
          │
          ▼
Frame Annotation
          │
          ▼
Output Analytics Video
```

---

# 📈 Current Capabilities

✅ Player Detection

✅ Goalkeeper Detection

✅ Referee Detection

✅ Ball Detection

✅ Persistent Player Tracking

✅ Team Classification

✅ Ball Possession Detection

✅ Live Team Possession Statistics

✅ Camera Movement Estimation

✅ Ball Position Interpolation

✅ Match Video Annotation

---

# 📸 Generated Analytics

### Player Analytics

* Persistent Player IDs
* Team Assignment
* Ball Possession Holder Detection

### Team Analytics

* Team Possession Percentage
* Team-Based Tracking

### Match Analytics

* Camera Movement Analysis
* Ball Tracking
* Object Tracking

### Visual Overlays

* Team Colors
* Possession Indicators
* Analytics Dashboard
* Camera Motion Dashboard

---

# 🔮 Roadmap

## Phase 1 — Completed ✅

* Object Detection
* Player Tracking
* Ball Tracking
* Team Assignment
* Ball Possession Analysis
* Camera Movement Estimation
* Video Annotation

---

## Phase 2 — In Progress 🚧

* View Transformer
* Real-World Pitch Coordinates
* Speed Estimation
* Distance Covered Metrics

---

## Phase 3

* Tactical Heatmaps
* Passing Network Analysis
* Shot Detection
* Expected Threat (xT)
* Possession Zones

---

## Phase 4

* Real-Time Match Analysis
* Multi-Camera Tracking
* Tactical Intelligence Dashboard
* AI-Powered Match Insights

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

Run the complete pipeline:

```bash
python main.py
```

Generated outputs will be saved in:

```bash
output/
```

---

# 📸 Sample Output

The generated video contains:

* Player Tracking IDs
* Ball Tracking
* Team Classification
* Ball Possession Indicators
* Team Possession Statistics
* Camera Movement Statistics
* Annotated Match Analytics

---

# 🤝 Contributing

Contributions, suggestions, and feature requests are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Developer

**Krish Malhotra**

AI • Computer Vision • Machine Learning • Football Analytics

---

### ⚽ SoccerEye — Turning Football Footage into Match Intelligence
