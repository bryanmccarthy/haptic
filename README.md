# Haptic

A hand gesture recognition system that maps user-defined hand gestures to keyboard keypresses. The application uses MediaPipe for hand tracking and TensorFlow for gesture classification.


### Features

- Real-time hand tracking and landmark detection
- Custom gesture recording and training
- Map gestures to keyboard keypresses
- Inference mode to execute keypresses based on recognized gestures

### Requirements

- Python 3.11
- TensorFlow
- MediaPipe
- OpenCV
- DearPyGui
- PyAutoGUI

### Setup

Create and activate a virtual environment:

```
source venv/bin/activate
```

Install dependencies:

```
pip3 install -r requirements.txt
```

### Run

```
python3.11 src/main.py
```

### Refs

- https://dearpygui.readthedocs.io/en/latest/index.html
- https://pyautogui.readthedocs.io/en/latest/
- https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
- https://github.com/kinivi/hand-gesture-recognition-mediapipe
- https://storage.googleapis.com/mediapipe-assets/gesture_recognizer/model_card_hand_gesture_classification_with_faireness_2022.pdf
