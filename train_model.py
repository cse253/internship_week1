import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

LANDMARK_DIR = "asl_dataset"

# ── 1. Check if landmark CSVs exist ─────────────────────────────────────────
landmark_files = glob.glob(os.path.join(LANDMARK_DIR, "*.csv"))

if not landmark_files:
    print(f"No landmark files (*.csv) found in '{LANDMARK_DIR}'.")
    print("Please run bootstrap_asl_dataset.py first to initialize the dataset,")
    print("or run collect_gesture_data.py to collect custom webcam landmarks.")
    exit(1)

print(f"Found {len(landmark_files)} classes in '{LANDMARK_DIR}':")
for f in landmark_files:
    print(f" - {os.path.basename(f)}")

# ── 2. Load and combine landmark CSVs ────────────────────────────────────────
all_data = []
for f in landmark_files:
    try:
        df = pd.read_csv(f)
        if not df.empty:
            all_data.append(df)
    except Exception as e:
        print(f"Error reading {f}: {e}")

if not all_data:
    print("No valid data loaded from landmark files.")
    exit(1)

data = pd.concat(all_data, ignore_index=True)

print("\nDataset Loaded Successfully")
print("Shape:", data.shape)
print("\nClass Value Counts:")
print(data["label"].value_counts())

# ── 3. Preprocessing ─────────────────────────────────────────────────────────
X = data.drop("label", axis=1)   # 63 landmark features (21 x x,y,z)
y = data["label"]                 # ASL alphabet label

# ── 4. Train/Test Split ───────────────────────────────────────────────────────
# If dataset is extremely small (e.g. only 1 sample per class during initial bootstrap),
# train_test_split might fail if test_size=0.2 and there is only 1 sample.
# We will use split only if there are enough samples per class, or train on full data if extremely small.
min_class_count = data["label"].value_counts().min()
if min_class_count < 2:
    print("\n[WARNING] Some classes have less than 2 samples. Training on the full dataset without split.")
    X_train, X_test, y_train, y_test = X, X, y, y
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

# ── 5. Train Random Forest Model ─────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── 7. Save model for use in gesture_recognition.py ──────────────────────────
with open("asl_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nModel saved to asl_model.pkl")
