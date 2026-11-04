
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import time

WORDS = ["HELLO", "THANK_YOU", "PLEASE", "SORRY", "YES"]
SEQUENCE_LENGTH = 30

print("Available words:")
for word in WORDS:
    print("-", word)

label = input("\nEnter a word exactly as shown above: ").strip().upper()

if label not in WORDS:
    raise SystemExit("Please enter one of the five available words.")

folder = Path("dataset") / "word_sequences" / label
folder.mkdir(parents=True, exist_ok=True)

existing = list(folder.glob("*.npy"))
sample_number = len(existing) + 1

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise SystemExit("Could not open camera.")

print("\nShow the", label, "sign.")
print("Press S to record one sequence of 30 frames.")
print("Press Q to quit.")

while True:
    success, frame = camera.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            result.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS
        )

    cv2.putText(
        frame, f"Word: {label} | Samples: {sample_number - 1}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )
    cv2.imshow("Word Sequence Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        sequence = []
        print("Recording... perform the sign now.")

        for i in range(SEQUENCE_LENGTH):
            success, frame = camera.read()
            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if not result.multi_hand_landmarks:
                print("Hand lost. Sequence not saved. Try again.")
                sequence = []
                break

            points = []
            hand = result.multi_hand_landmarks[0]

            for point in hand.landmark:
                points.extend([point.x, point.y, point.z])

            sequence.append(points)

            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )
            cv2.putText(
                frame, f"Recording: {i + 1}/{SEQUENCE_LENGTH}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2
            )
            cv2.imshow("Word Sequence Collection", frame)
            cv2.waitKey(1)

        if len(sequence) == SEQUENCE_LENGTH:
            filename = folder / f"sample_{sample_number:03d}.npy"
            np.save(filename, np.array(sequence, dtype=np.float32))
            print("Saved:", filename)
            sample_number += 1

        time.sleep(0.5)

    elif key == ord("q"):
        break

camera.release()
hands.close()
cv2.destroyAllWindows()
print("Collection finished.")
