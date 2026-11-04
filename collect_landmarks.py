import cv2
import mediapipe as mp
import csv
from pathlib import Path

label = input("Enter sign letter (A-Z): ").strip().upper()

if len(label) != 1 or not label.isalpha():
    raise SystemExit("Enter one letter from A to Z.")

folder = Path("dataset")
folder.mkdir(exist_ok=True)

csv_file = folder / "landmarks.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise SystemExit("Camera could not open.")

count = 0
print("Show your hand sign.")
print("Press S to save a landmark sample.")
print("Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    landmarks = None

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        points = []
        for point in hand.landmark:
            points.extend([point.x, point.y, point.z])

        landmarks = points

        mp.solutions.drawing_utils.draw_landmarks(
            frame, hand, mp_hands.HAND_CONNECTIONS
        )

    cv2.putText(
        frame, f"Letter: {label} | Saved: {count}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 255, 0), 2
    )

    cv2.imshow("Collect Hand Landmarks", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        if landmarks is not None:
            with open(csv_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([label] + landmarks)

            count += 1
            print("Saved:", count)
        else:
            print("No hand detected. Try again.")

    elif key == ord("q"):
        break

camera.release()
hands.close()
cv2.destroyAllWindows()