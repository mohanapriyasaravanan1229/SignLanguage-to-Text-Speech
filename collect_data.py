"""Collect labeled webcam images. Run: python collect_data.py"""
from pathlib import Path
import cv2
import numpy as np
label = input("Enter sign label (e.g. HELLO): ").strip().upper()
if not label:
    raise SystemExit("A label is required.")
folder = Path("dataset") / label
folder.mkdir(parents=True, exist_ok=True)
count = len(list(folder.glob("*.jpg")))
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise SystemExit("Could not open webcam. Check camera permissions.")
print("Press S to save a frame; Q to quit.")
while True:
    ok, frame = cam.read()
    if not ok:
        break
    cv2.putText(frame, f"{label} | saved: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Collect sign images", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        count += 1
        cv2.imwrite(str(folder / f"{label}_{count:04d}.jpg"), frame)
        print("Saved", count)
    elif key == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()
