# Air Shapes 3D

**Draw in the air. Create in 3D.**

Air Shapes 3D is a real-time computer vision project that lets you draw shapes using hand gestures in front of your webcam. It detects your hand, tracks your fingertip movements, recognizes the shape you draw, and transforms it into a rotating 3D model.

The project combines hand tracking, geometric shape recognition, and custom 3D rendering to create an interactive, touch-free drawing experience.

## ✨ Features

- **Real-time hand tracking** using MediaPipe.
- **Air drawing** controlled by pinching your thumb and index finger.
- **Automatic shape recognition** for circles, squares, rectangles, triangles, stars, and hearts.
- **3D model generation** from recognized shapes.
- **Custom 3D rendering** with lighting, shading, perspective, and visible mesh edges.
- **Smooth gesture tracking** to reduce jitter while drawing.
- **Interactive webcam interface** with a live camera view and a 3D preview.
- **Touch-free interaction** without a mouse or drawing tablet.

## 🛠️ Technologies

- Python
- OpenCV
- MediaPipe
- NumPy
- Mathematical geometry and 3D mesh rendering

## 🖥️ How It Works

1. **Hand detection:** The webcam captures live video, and MediaPipe detects hand landmarks.
2. **Gesture tracking:** The program tracks the positions of your thumb and index fingertip.
3. **Air drawing:** Pinching your thumb and index finger starts drawing. Separating them finishes the stroke.
4. **Shape recognition:** The program analyzes the completed stroke using contour approximation, convex hulls, circularity, aspect ratios, and convexity defects.
5. **3D generation:** The recognized shape is converted into a 3D mesh. Circles become spheres, while polygonal shapes become 3D prisms.
6. **Rendering:** The generated model rotates continuously in the 3D preview panel.



## 🎮 Controls

| Action               | How to use it                               |
| -------------------- | ------------------------------------------- |
| Start drawing        | Pinch your thumb and index finger together. |
| Finish drawing       | Separate your thumb and index finger.       |
| Clear the canvas     | Press `C`.                                  |
| Exit the application | Press `Q` or `Esc`.                         |

For the best results, keep your hand visible, use good lighting, and draw a reasonably large, closed shape.

## 📐 Supported Shapes

| Shape     | 3D representation  |
| --------- | ------------------ |
| Circle    | Sphere             |
| Square    | Square prism       |
| Rectangle | Rectangular prism  |
| Triangle  | Triangular prism   |
| Star      | Star-shaped prism  |
| Heart     | Heart-shaped prism |

Shape recognition is based on geometric heuristics, so results may vary depending on drawing accuracy and hand-tracking stability.

## 📁 Project Structure

```text
air_shapes/
├── air_shapes.py
├── hand_landmarker.task
├── requirements.txt
└── README.md
```

The `hand_landmarker.task` file is a downloaded model asset rather than a Python dependency. Follow its licensing and redistribution terms before including it in your repository.


## 💡 Motivation

This project explores how computer vision and geometry can turn a simple human gesture into an interactive 3D experience. It is a hands-on experiment in combining real-time perception, shape recognition, and graphics without relying on a traditional input device.

---

*Built with Python, computer vision, and curiosity.*
