import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

dataset_folder = "dataset"

all_data = []

files = ["open_palm.csv", "fist.csv", "thumbs_up.csv"]

for file in files:
    path = os.path.join(dataset_folder, file)
    df = pd.read_csv(path)
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

print("Dataset Loaded Successfully")
print(data.head())

X = data.drop("gesture", axis=1)
y = data["gesture"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))