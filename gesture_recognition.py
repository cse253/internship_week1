import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ── 1. Train model from dataset ──────────────────────────────────────────────
files = ["open_palm.csv", "fist.csv", "thumbs_up.csv"]
all_data = [pd.read_csv(os.path.join("dataset", f)) for f in files]
data = pd.concat(all_data, ignore_index=True)

X = data.drop("gesture", axis=1)
y = data["gesture"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Model trained. Accuracy on test set:", round(model.score(X_test, y_test) * 100, 2), "%")

# ── 2. MediaPipe setup ───────────────────────────────────────────────────────
mp_hands = mp.tasks.vision.HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None

def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=result_callback
)

# ── 3. Webcam + real-time prediction ────────────────────────────────────────
cap = cv2.VideoCapture(0)

with mp_hands.create_from_options(options) as landmarker:
    timestamp = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        landmarker.detect_async(mp_image, timestamp)
        timestamp += 1

        if latest_result and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:
                h, w, _ = frame.shape

                # Draw landmarks
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                # Build feature row and predict
                row = []
                for lm in hand_landmarks:
                    row.extend([lm.x, lm.y, lm.z])

                prediction = model.predict([row])[0]
                confidence = round(max(model.predict_proba([row])[0]) * 100, 1)

                # Show prediction on screen
                label = f"{prediction} ({confidence}%)"
                cv2.rectangle(frame, (0, 0), (350, 50), (0, 0, 0), -1)
                cv2.putText(frame, label, (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
