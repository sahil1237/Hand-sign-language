# Sign Language Translator

A beginner-friendly Python project that uses a webcam to detect hand landmarks, classify selected sign-language gestures, display the result as subtitles, and convert the text to speech.

## Current stage
Stage 1: project setup and webcam/hand detection.

## Setup on Windows
Open Command Prompt in this folder and run:

python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

Then run:

python app.py

The first prototype will show the webcam and detected hand landmarks. ML training and speech will be added in later stages.
