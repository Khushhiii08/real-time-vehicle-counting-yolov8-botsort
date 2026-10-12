---
title: Real Time Vehicle Counting
sdk: streamlit
app_file: app.py
---

# 🚗 Real-Time Vehicle Counting using YOLOv8 + BoT-SORT

This repository presents a **real-time vehicle counting system** built using modern object detection and multi-object tracking techniques. The system leverages **YOLOv8** for fast and accurate vehicle detection and **BoT-SORT** for robust identity tracking, enabling reliable vehicle counting in dynamic traffic scenarios.

A **Streamlit-based web application** is provided to demonstrate **real-time inference and visualization** of the proposed pipeline.

---

## 🚦 Key Features

- Real-time vehicle detection using **YOLOv8**
- Robust multi-object tracking with **BoT-SORT**
- Line-based vehicle counting to avoid double counting
- Streamlit web interface for interactive visualization
- Supports uploaded traffic videos
- Modular, research-oriented project structure

---

## 🧠 System Overview

**Processing Pipeline:**
```
Input Video 
↓
YOLOv8 Object Detection
↓
BoT-SORT Multi-Object Tracking
↓
Virtual Line Crossing Logic
↓
Vehicle Count Output
```
---

## 📂 Project Structure
```
├── app.py                     # Streamlit application (inference-only)
├── device.py                  # Automatic CPU / GPU selection
├── virtual_line.py            # Line-crossing vehicle counting logic
├── evaluation.py              # FPS and runtime performance metrics
│
├── trackers/
│   └── botsort.yaml           # BoT-SORT configuration file
│
├── preprocessing/             # Dataset preprocessing & EDA (offline)
│
├── streamlit display.pdf      #results
├── IEEEConferencePaper/       # Accepted research paper
│
├── requirements.txt
├── README.md
└── LICENSE
```
---

---

## 🚗 Vehicle Counting Logic

A **virtual line-based counting mechanism** ensures accurate vehicle counts:

- Each detected vehicle is assigned a unique track ID
- The centroid trajectory of each vehicle is tracked across frames
- A vehicle is counted **only once** when it crosses the predefined line
- Direction-based checks prevent duplicate counting

This approach remains reliable even under moderate occlusion and dense traffic.

---

## 📊 Model Details

### YOLOv8
- Anchor-free object detection architecture
- Optimized for real-time inference
- Strong trade-off between speed and accuracy

### BoT-SORT
- Advanced multi-object tracker with ReID embeddings
- Superior identity preservation compared to DeepSORT and ByteTrack
- Configured with a frame buffer to handle short-term occlusions

---

## 📦 Models

Due to GitHub file size limitations, trained model weights are **not stored directly in this repository**.

📥 Download pretrained models from: https://drive.google.com/drive/folders/16rCglZ9cv0il5h_tRIjz3csWCWal4H7W?usp=sharing

---

## 🗂 Dataset

The dataset used for training and evaluation is available here: https://github.com/Khushhiii08/IDD_Detection-Dataset-for-YOLO
---

---

## 🖥️ Streamlit Application

- Training
- dataset preprocessing
- EDA 
- Upload traffic videos for analysis
- Visualize bounding boxes, track IDs, and vehicle counts
- Observe real-time inference performance (FPS)

---

## ⚙️ Installation (Local)

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Khushhiii08/real-time-vehicle-counting-yolov8-botsort.git
cd real-time-vehicle-counting-yolov8-botsort
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### ▶️ Run the Application
```bash
streamlit run app.py
```

## 🧪 Experimental Notes
-	YOLOv8 + BoT-SORT forms the end-to-end real-time pipeline
-	RF-DETR and YOLOv12 were evaluated at the detection level (offline)
-	BoT-SORT demonstrated superior identity consistency over alternative trackers
-	Detailed metrics and qualitative results are provided in the paper and the streamlit-display pdf

## 🚧 Limitations
-	Performance degradation in extreme low-light conditions
-	Occasional ID drift under very dense traffic
-	Transformer-based detectors not integrated for live inference due to computational constraints

## 🔮 Future Work
-	Vehicle speed estimation using trajectory analysis
- Traffic congestion and flow analysis
-	Night-time and adverse weather enhancements
-	Edge deployment (Jetson / Raspberry Pi)
-	Cloud-based analytics dashboard for long-term monitoring

## 🧾 License
This project is licensed under the MIT License.

## 🙏 Acknowledgements
-	Ultralytics YOLOv8
-	BoT-SORT authors
-	Streamlit
-	PyTorch open-source community
