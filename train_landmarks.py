import csv
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

csv_file = Path("dataset") / "landmarks.csv"

X = []
y = []

with open(csv_file, "r") as file:
    reader = csv.reader(file)

    for row in reader:
        if len(row) == 64:
            y.append(row[0])
            X.append([float(value) for value in row[1:]])

X = np.array(X)
y = np.array(y)

if len(set(y)) < 2:
    raise SystemExit("Collect at least two different letters.")

print("Total samples:", len(X))
print("Letters:", sorted(set(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", probability=True)
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

joblib.dump(model, "landmark_model.joblib")

print("New landmark model saved successfully!")