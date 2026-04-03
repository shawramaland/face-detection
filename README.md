# Face Detecter 🐹

A fun real-time face expression detector that mirrors your emotions with hamster memes.

It uses your webcam to detect your facial expression and shows a matching hamster image side by side.

## How it works

- Shows your webcam feed on the left
- Detects your emotion in real time
- Displays the matching hamster meme on the right
- Close your eyes and see what happens 👀

## Emotions detected

| Your face | Hamster |
|---|---|
| Happy | Happy hamster |
| Angry | Angry hamster |
| Surprised | Interested hamster |
| Neutral | Neutral hamster |
| Eyes closed | Dead hamster |
| No face found | Serious hamster |

## Setup

**Requirements:** Python 3.x

Install dependencies:
```bash
pip install -r requirements.txt
```

Run:
```bash
python main.py
```

> First run will automatically download the emotion model (~33MB). After that it's instant.

Press **Q** to quit.

## ⚠️ Work in Progress

This project is still being improved! Known issues:
- Emotion detection can be inconsistent with extreme expressions
- Eye-closed detection sometimes triggers during squinting
- Face detection may briefly lose your face during big expressions

Feel free to clone it, play with it, and improve it yourself!

## Built with

- [OpenCV](https://opencv.org/) — webcam feed & face detection
- [ONNX Runtime](https://onnxruntime.ai/) — emotion model inference
- [emotion-ferplus](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus) — pre-trained emotion recognition model
