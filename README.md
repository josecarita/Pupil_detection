# 👀 Remote-Class Attention Monitor

> Computer-vision prototype that estimates whether a student in an online class is paying attention, using the webcam.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Detection-0097A7)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)

## Idea

In remote classes the teacher cannot see if students are following. This project looks at the student's face through the webcam and labels each frame as **"Atento"** (attentive) or **"No atento"** (not attentive).

## How it works (current version)

1. **MediaPipe Face Detection** finds the face and returns the position of both eyes.
2. The line between the eyes is compared with the horizontal to get the **head tilt angle**.
3. Decision rule:
   - no face in the frame → **No atento**
   - tilt between −40° and 40° → **Atento**
   - larger tilt (head resting or turned away) → **No atento**

The result is drawn on the image together with the eye points.

## Usage

```bash
pip install opencv-python mediapipe numpy
```

**Real time (webcam):**

```bash
python detectorrealtime.py   # press Esc to quit
```

**Batch (images):** put pictures in a folder called `imagenes de muestra`, then:

```bash
python detector.py           # results are saved in "imagenes resultantes"
```

## Roadmap

The goal is to judge attention from the eyes themselves, not only head position:

- [ ] **Eye openness** – use MediaPipe Face Mesh landmarks and the Eye Aspect Ratio (EAR) to detect closed eyes / drowsiness.
- [ ] **Gaze direction** – use the iris landmarks to tell if the student is looking at the screen.
- [ ] Combine the signals over time (e.g. % of time attentive per minute) instead of a per-frame label.
- [ ] Privacy: process everything locally and only report the attention score.
