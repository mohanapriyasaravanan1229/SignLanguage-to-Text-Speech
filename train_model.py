"""Train a basic HOG + SVM image classifier. Run: python train_model.py"""
from pathlib import Path
import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

DATA = Path("dataset")
SIZE = (128, 128)
hog = cv2.HOGDescriptor(SIZE, (16, 16), (8, 8), (8, 8), 9)
X, y = [], []
for folder in sorted(p for p in DATA.iterdir() if p.is_dir()) if DATA.exists() else []:
    for path in folder.glob("*.jpg"):
        img = cv2.imread(str(path))
        if img is not None:
            gray = cv2.cvtColor(cv2.resize(img, SIZE), cv2.COLOR_BGR2GRAY)
            X.append(hog.compute(gray).flatten())
            y.append(folder.name)
if len(set(y)) < 2 or len(y) < 10:
    raise SystemExit("Collect at least 10 images for each of at least 2 labels first.")
X = np.asarray(X)
y = np.asarray(y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True))
model.fit(Xtr, ytr)
pred = model.predict(Xte)
print("Test accuracy:", round(accuracy_score(yte, pred), 3))
print(classification_report(yte, pred))
joblib.dump(model, "sign_model.joblib")
print("Saved sign_model.joblib")
