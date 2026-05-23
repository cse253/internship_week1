# Real-Time ASL Alphabet Gesture Recognition & Word Builder
# Detects hand landmarks, predicts ASL alphabet, builds words, and speaks them live!

import cv2
import mediapipe as mp
import pickle
import os
import threading

# ── 1. Load trained ASL Alphabet model ───────────────────────────────────────
MODEL_PATH = "asl_model.pkl"
if not os.path.exists(MODEL_PATH):
    print(f"Model '{MODEL_PATH}' not found. Please run train_model.py first.")
    exit(1)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print("ASL model loaded successfully. Starting webcam...")

# ── 2. TTS Voice Setup (Windows Offline SAPI5 support) ────────────────────────
def speak(text):
    def _speak_thread():
        try:
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
        except Exception as e:
            print(f"TTS not available: {e}")
    threading.Thread(target=_speak_thread, daemon=True).start()

# ── 3. MediaPipe Tasks API setup (compatible with mediapipe 0.10+) ───────────
mp_hands = mp.tasks.vision.HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = [None]

def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    latest_result[0] = result

landmarker_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=result_callback
)

# Word Builder state variables
current_sentence = ""
last_raw_prediction = None
stable_prediction = None
prediction_frames = 0
DEBOUNCE_THRESHOLD = 15  # frames to hold a sign to commit it

# Visual styling colors (BGR)
COLOR_TEXT = (255, 255, 255)
COLOR_HUD = (10, 10, 10)
COLOR_BORDER = (0, 255, 0)
COLOR_TEXT_ACCENT = (0, 255, 0)

cap = cv2.VideoCapture(0)

print("\n=============================================")
print("ASL Alphabet Recognition is Ready!")
print("Controls:")
print(" - Hold a hand sign steady to commit the letter")
print(" - Hold the 'space' sign to insert a space")
print(" - Hold the 'del' sign to delete the last letter")
print(" - Press 'C' key to CLEAR the current word/sentence")
print(" - Press 'Q' key to QUIT the application")
print("=============================================\n")

speak("ASL Recognition Ready")

with mp_hands.create_from_options(landmarker_options) as landmarker:
    timestamp = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        landmarker.detect_async(mp_image, timestamp)
        timestamp += 1

        detected_hand = False
        prediction = "nothing"
        confidence = 1.0

        if latest_result[0] and latest_result[0].hand_landmarks:
            detected_hand = True
            for hand_landmarks in latest_result[0].hand_landmarks:
                # Draw landmarks
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                # Build 63-feature row (21 landmarks x 3 coords)
                row = []
                for lm in hand_landmarks:
                    row.extend([lm.x, lm.y, lm.z])

                # Feed to Random Forest model
                prediction = model.predict([row])[0]
                confidence = max(model.predict_proba([row])[0])

        # ── 4. Debouncing & Character Committing Logic ───────────────────────
        if detected_hand:
            if prediction == last_raw_prediction:
                prediction_frames += 1
            else:
                last_raw_prediction = prediction
                prediction_frames = 0

            # Commit the character once it passes the frame threshold
            if prediction_frames == DEBOUNCE_THRESHOLD:
                if prediction == "space":
                    current_sentence += " "
                    speak("Space")
                elif prediction == "del":
                    if len(current_sentence) > 0:
                        removed_char = current_sentence[-1]
                        current_sentence = current_sentence[:-1]
                        speak(f"delete {removed_char}")
                    else:
                        speak("empty")
                elif prediction == "nothing":
                    pass
                else:
                    current_sentence += prediction.upper()
                    speak(prediction)
                
                stable_prediction = prediction
                prediction_frames = 0
        else:
            # If no hand detected, slowly fade the raw prediction
            last_raw_prediction = None
            prediction_frames = 0

        # ── 5. Render Translucent HUD & text overlays ───────────────────────
        # Top HUD bar (translucent background)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), COLOR_HUD, -1)
        # Bottom Sentence bar
        cv2.rectangle(overlay, (0, h - 80), (w, h), COLOR_HUD, -1)
        # Apply translucent overlay (alpha=0.7)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw borders
        cv2.line(frame, (0, 100), (w, 100), COLOR_BORDER, 2)
        cv2.line(frame, (0, h - 80), (w, h - 80), COLOR_BORDER, 2)

        # Top Bar text
        if detected_hand:
            conf_str = f"{round(confidence * 100, 1)}%"
            pred_text = f"Sign: {prediction.upper()} ({conf_str})"
            # Progress bar for committing
            progress = min(prediction_frames / DEBOUNCE_THRESHOLD, 1.0)
            bar_width = int(progress * 200)
            cv2.rectangle(frame, (10, 75), (210, 85), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 75), (10 + bar_width, 85), COLOR_BORDER, -1)
        else:
            pred_text = "Waiting for hand sign..."

        cv2.putText(frame, pred_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT, 2)
        cv2.putText(frame, "Keys: C = Clear | Q = Quit", (w - 320, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Bottom Bar Text (Current Word / Sentence)
        sentence_display = f"Text: {current_sentence}"
        cv2.putText(frame, sentence_display, (15, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, COLOR_TEXT_ACCENT, 2)

        # Draw a cursor at the end of the text
        text_size = cv2.getTextSize(sentence_display, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)[0]
        cursor_x = 15 + text_size[0] + 5
        if cv2.waitKey(1) & 0xFF == ord('c') or (cv2.getTickCount() // (cv2.getTickFrequency() * 0.5)) % 2 == 0:
            cv2.rectangle(frame, (cursor_x, h - 55), (cursor_x + 12, h - 25), COLOR_TEXT_ACCENT, -1)

        # Show frame
        cv2.imshow("ASL Alphabet Sign Language Recognition", frame)

        # Handle Keyboard Events
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord('q'):
            break
        elif keypress == ord('c'):
            current_sentence = ""
            speak("cleared")
            print("Cleared current sentence.")

cap.release()
cv2.destroyAllWindows()
