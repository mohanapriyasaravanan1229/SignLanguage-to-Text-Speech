# Sign Language to Text & Speech Translator

## Status
This is a starter scaffold, not a finished sign-language recognition system. The web page provides a camera preview and browser speech output. Recognition still needs dataset collection, model training/testing, and integration.

## Problem and approach
The project aims to make communication easier between sign-language users and people who do not understand signs. The starter approach collects labeled hand-sign images and trains a baseline image classifier.

## Tech stack
Python, Flask, OpenCV, scikit-learn, NumPy, joblib, HTML, CSS, JavaScript Web Speech API.

## Setup
1. Activate your Python virtual environment.
2. Install: `pip install -r requirements.txt`
3. Run: `python app.py`
4. Open `http://127.0.0.1:5000` and allow camera access.

## Collect data
Run `python collect_data.py`. Enter a label such as HELLO. Press S to save images and Q to quit. Repeat for at least two labels with varied examples.

## Train
Run `python train_model.py` after collecting at least 10 images for each of at least two labels. It prints test metrics and saves `sign_model.joblib`.

## Important next work
Integrate model predictions into the live app, collect a balanced dataset, test with unseen signers and lighting, and report only measured performance. Add screenshots and a demo video after testing.

## AI tools disclosure
List the AI tools used and how they were used. Update honestly before submission.

## Team
Add your team name and member names before submission.
