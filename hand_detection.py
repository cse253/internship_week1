# Hand Detection with ASL Alphabet Prediction
# Detects hand landmarks and predicts ASL letter using trained Random Forest model

import cv2
import mediapipe as mp
import pickle
import os

# ── 1. Load trained ASL Alphabet model ───────────────────────────────────────
MODEL_PATH = "asl_model.pkl"
if not os.path.exists(MODEL_PATH):
    print(f"Model '{MODEL_PATH}' not found. Please run train_model.py first.")
    exit(1)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print("ASL model loaded successfully. Starting webcam...")

# ── 2. MediaPipe Solutions Hands setup ───────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Support up to 2 hands in hand_detection.py
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert image color to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                # Draw landmarks on frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Build 63 features
                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])

                # Predict ASL letter
                prediction = model.predict([row])[0]
                confidence = round(max(model.predict_proba([row])[0]) * 100, 1)

                # Draw prediction info near the hand (Wrist landmark is index 0)
                wrist = hand_landmarks.landmark[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                
                # Show label text above wrist
                cv2.putText(frame, f"{prediction.upper()} ({confidence}%)", 
                            (cx - 30, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Hand Detection - ASL Alphabet", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
