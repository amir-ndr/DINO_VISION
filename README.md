# DINO_VISION
Dino Run with YOLO-Based Hand Gesture Control

# 🦖 Dino Run with YOLO-Based Hand Gesture Control

This project is a computer vision-powered twist on the classic Chrome "Dino Run" game. The player controls the dinosaur **using hand gestures** detected in real time with a **YOLOv8 object detection model**.

🎮 Built with **Pygame** and **OpenCV**, the game includes custom animations, sprite collisions, jumping physics, and a hand-tracking input mechanism using a webcam.

---

## 🎯 Features

- ✅ Classic Dino Run gameplay (cacti, flying pteras, clouds, and scrolling ground)
- 🎥 Real-time hand gesture control via **YOLOv8**
  - Open hand (class 16) → Jump
  - close fist (class 15) → Do nothing
- 📈 Dynamic difficulty: game speed increases over time
- 🔊 Sound effects for jumping, dying, and checkpoints
- 🧠 Score tracking with high score saving

---

## 🛠 Tech Stack

- **Python 3.8+**
- **Pygame** – for rendering and game loop
- **OpenCV** – for webcam integration
- **YOLOv8** – for real-time hand gesture detection (via [Ultralytics YOLO](https://github.com/ultralytics/ultralytics))

---
