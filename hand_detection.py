import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.tasks.vision.HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"

latest_result = None

def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=result_callback
)

cap = cv2.VideoCapture(0)

with mp_hands.create_from_options(options) as landmarker:
    timestamp = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        landmarker.detect_async(mp_image, timestamp)
        timestamp += 1

        if latest_result:
            for hand_landmarks in latest_result.hand_landmarks:
                h, w, _ = frame.shape
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        cv2.imshow("Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
