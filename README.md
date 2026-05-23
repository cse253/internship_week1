# Sign Language Recognition using Machine Learning

A real-time Sign Language Recognition system built using Python, OpenCV, MediaPipe, and Scikit-learn. This project was developed as part of a one-week internship plan focused on practical AI/ML and Computer Vision workflows.

---

## Project Goal

To build a system that can detect hand gestures from a live webcam feed and classify them into sign language labels (ASL alphabet / emotion labels) using a trained Machine Learning model.

---

## Tools and Technologies

| Tool | Purpose |
|---|---|
| Python 3.11 | Core programming language |
| OpenCV | Webcam capture and frame processing |
| MediaPipe | Hand landmark detection (21 keypoints) |
| NumPy | Numerical operations |
| Pandas | Dataset loading and preprocessing |
| Scikit-learn | Model training and evaluation (Random Forest) |
| EmoSign Dataset | Emotion-labeled ASL video dataset |
| Git & GitHub | Version control and project hosting |

---

## Project Structure

```
hand_project/
├── webcam_capture.py         # Day 1 - Basic webcam feed
├── hand_detection.py         # Day 2 - Hand landmark detection
├── collect_gesture_data.py   # Day 3 - Dataset collection via webcam
├── bootstrap_asl_dataset.py  # Day 3 - Extract landmarks from images
├── train_model.py            # Day 4 - Model training and evaluation
├── gesture_recognition.py    # Day 5 - Real-time gesture recognition
├── asl_dataset/              # Collected ASL landmark CSVs
├── EmoSign/                  # EmoSign emotion dataset + landmarks
├── Unvoiced/                 # ASL test images
├── hand_landmarker.task      # MediaPipe hand landmarker model
├── asl_model.pkl             # Trained Random Forest model
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Methodology and Workflow

### Day 1 — Environment Setup
- Installed Python, VS Code, OpenCV, MediaPipe, NumPy
- Built `webcam_capture.py` to capture and display live webcam frames

### Day 2 — Hand Detection
- Used MediaPipe HandLandmarker to detect 21 hand keypoints per frame
- Drew landmarks on the webcam feed in real time (`hand_detection.py`)

### Day 3 — Dataset Collection
- Collected 30–50 hand landmark samples per gesture using `collect_gesture_data.py`
- Each sample = 21 landmarks × (x, y, z) = 63 features saved to CSV
- Used `bootstrap_asl_dataset.py` to extract landmarks from ASL test images
- Integrated EmoSign dataset (emotion labels: positive, negative, neutral)

### Day 4 — Model Training
- Loaded and combined all landmark CSVs using Pandas
- Split data 80/20 into training and test sets
- Trained a `RandomForestClassifier` with 100 estimators
- Evaluated using accuracy score and classification report

### Day 5 — Real-Time Recognition
- Loaded trained model (`asl_model.pkl`) in `gesture_recognition.py`
- Detected landmarks from live webcam feed every frame
- Fed 63 landmark features into model for instant prediction
- Displayed predicted gesture + confidence % on screen

### Day 6 — Refactoring and GitHub
- Refactored all scripts with clear comments and structure
- Generated `requirements.txt`
- Pushed project to GitHub with descriptive commit messages

---

## How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Collect gesture data (optional)
```
python collect_gesture_data.py
```

### 3. Train the model
```
python train_model.py
```

### 4. Run real-time recognition
```
python gesture_recognition.py
```

Press `q` to quit the webcam window.

---

## Future Enhancements

- Add more gestures and full ASL alphabet support
- Recognize full words and sentences from sequences of gestures
- Convert recognized gestures to speech using text-to-speech
- Build a web app using Flask or Streamlit for browser-based access
- Deploy as a mobile app for accessibility use cases
- Use deep learning (LSTM/Transformer) for temporal gesture sequences
- Add support for two-hand gestures

---

## Ethical and Accessibility Use

This project aims to bridge communication gaps for the Deaf and hard-of-hearing community. Sign language recognition systems can be used in:
- Real-time communication assistance
- Educational tools for learning sign language
- Accessibility features in video conferencing apps

---


