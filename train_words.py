
from pathlib import Path
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

DATA = Path("dataset") / "word_sequences"

X = []
y = []

for word_folder in sorted(DATA.iterdir()):
    if word_folder.is_dir():
        for file in word_folder.glob("*.npy"):
            sequence = np.load(file)

            if sequence.shape == (30, 63):
                X.append(sequence.flatten())
                y.append(word_folder.name)

X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))
print("Words:", sorted(set(y)))

if len(set(y)) < 2 or len(X) < 10:
    raise SystemExit("Collect more word samples first.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", probability=True)
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

joblib.dump(model, "word_model.joblib")
print("Word model saved successfully!")
