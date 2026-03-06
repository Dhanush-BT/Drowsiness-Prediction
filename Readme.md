# Driver Drowsiness Detection using LSTM & Facial Features

**Driver Drowsiness Detection System** is a real-time AI solution that predicts driver fatigue using temporal facial signals extracted from video. It leverages **deep learning (LSTM in PyTorch)** to analyze sequences of eye, mouth, and head movement features, providing accurate alerts for drowsiness and enhancing road safety.

---

## Key Features

- **Real-time Detection:** Monitors driver behavior from video/webcam feed.
- **Temporal Analysis:** LSTM network analyzes sequences of 30 frames to capture subtle fatigue patterns.
- **Multiple Facial Signals:** Extracts EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio), head pose (Pitch, Yaw, Roll), and PERCLOS (Eye closure percentage).
- **Visual Feedback:** Displays live alert overlay (`ALERT` / `DROWSY`) on video frames.
- **Final Verdict:** Provides overall fatigue status after video using majority vote over predictions.

---

## Project Structure
