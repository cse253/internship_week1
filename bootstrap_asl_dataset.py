import cv2
import mediapipe as mp
import csv
import os
import glob

# 1. Setup paths
TEST_IMAGES_DIR = os.path.join("Unvoiced", "Test Images")
OUTPUT_DIR = "asl_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Get list of all test images
image_files = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg"))
print(f"Found {len(image_files)} test images in '{TEST_IMAGES_DIR}'")

# 3. MediaPipe tasks API (compatible with mediapipe 0.10+)
mp_hands = mp.tasks.vision.HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

success_count = 0
failed_count = 0
not_detected = []

# Header format
header = ["label"]
for i in range(21):
    header += [f"x{i}", f"y{i}", f"z{i}"]

with mp_hands.create_from_options(options) as landmarker:
    for img_path in image_files:
        basename = os.path.basename(img_path)

        # Parse label from file name (e.g. 'A_test.jpg' -> 'a')
        label_raw = basename.split("_")[0].lower()

        if label_raw == "nothing":
            label = "nothing"
        elif label_raw == "space":
            label = "space"
        elif label_raw == "del":
            label = "del"
        else:
            label = label_raw

        print(f"Processing '{basename}' as label '{label}'...")

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [ERROR] Could not read image: {img_path}")
            failed_count += 1
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Synchronous detection on static image
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            csv_path = os.path.join(OUTPUT_DIR, f"{label}.csv")
            file_exists = os.path.exists(csv_path)

            with open(csv_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(header)

                hand_landmarks = result.hand_landmarks[0]
                row = [label]
                for lm in hand_landmarks:
                    row.extend([lm.x, lm.y, lm.z])

                writer.writerow(row)

            print(f"  [SUCCESS] Extracted landmarks -> {csv_path}")
            success_count += 1
        else:
            print(f"  [WARNING] No hand detected in '{basename}'")
            not_detected.append(label)

print("\n=== Bootstrapping Complete ===")
print(f"Successfully processed: {success_count} images.")
print(f"No hand detected in {len(not_detected)} images: {not_detected}")
print(f"Landmark dataset saved in '{OUTPUT_DIR}/' directory.")
