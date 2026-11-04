
from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import joblib
import numpy as np

app = Flask(__name__)

letter_model = joblib.load("landmark_model.joblib")
word_model = joblib.load("word_model.joblib")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

camera = cv2.VideoCapture(0)

sequence_buffer = []
latest_prediction = "Show a sign"
latest_type = "Waiting"

@app.route("/")
def home():
    return render_template("index.html")

def generate_frames():
    global sequence_buffer, latest_prediction, latest_type

    while True:
        success, frame = camera.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            points = []

            for point in hand.landmark:
                points.extend([point.x, point.y, point.z])

            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

            # Recognize individual letters
            letter = letter_model.predict([points])[0]

            # Collect 30 frames for word recognition
            sequence_buffer.append(points)

            if len(sequence_buffer) >= 30:
                word_input = np.array(
                    sequence_buffer[-30:], dtype=np.float32
                ).flatten()

                word = word_model.predict([word_input])[0]

                latest_prediction = word
                latest_type = "Word"

                sequence_buffer.clear()

            cv2.putText(
                frame, "Letter: " + str(letter),
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )

            cv2.putText(
                frame, "Word: " + str(latest_prediction),
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 0, 0), 2
            )

        else:
            sequence_buffer.clear()

        ret, buffer = cv2.imencode(".jpg", frame)

        if ret:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/prediction")
def prediction():
    return jsonify({
        "prediction": latest_prediction,
        "type": latest_type
    })

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
