# Code Calculated the Area by Camera

A computer vision project that uses a camera and a YOLO segmentation model to detect an object and calculate its area from the captured image.

The project uses a YOLO-based ONNX segmentation model to generate a segmentation mask of the target object. The detected mask is then processed to calculate the object's area.

---

## Features

- 📷 Capture images from a camera
- 🤖 YOLO object segmentation using ONNX
- 🎯 Detect and segment the target object
- 📐 Calculate the object's area
- 🖥️ GUI for testing the system
- 💾 Save captured images for testing and analysis
- 🔌 ONNX-based inference

---

## Project Structure

```text
code_calculated_the_area_by_camera/
│
├── captures/
│   └── Captured images and test data
│
├── area_calc.py
│   └── Area calculation and image processing
│
├── main_gui_test.py
│   └── Main GUI application
│
├── best_yolo_seg_309.onnx
│   └── YOLO segmentation model
│
├── README.md
│   └── Project documentation
│
└── LICENSE
    └── MIT License
```

---

## System Workflow

```text
             Camera
                │
                ▼
        Capture Image
                │
                ▼
       Image Pre-processing
                │
                ▼
       YOLO Segmentation
                │
                ▼
       Segmentation Mask
                │
                ▼
        Image Processing
                │
                ▼
       Pixel Area Calculation
                │
                ▼
      Physical Area Estimation
                │
                ▼
              GUI
                │
                ▼
          Display Result
```

---

## How It Works

### 1. Capture Image

The system uses a camera to capture an image of the target object.

Captured images can be stored in the:

```text
captures/
```

directory for testing and analysis.

---

### 2. Object Segmentation

The project uses the following ONNX model:

```text
best_yolo_seg_309.onnx
```

The model is used to detect and segment the target object.

Unlike traditional object detection, segmentation provides a pixel-level mask of the detected object.

```text
Input Image
     │
     ▼
YOLO Segmentation Model
     │
     ▼
Object Detection
     │
     ▼
Segmentation Mask
```

---

### 3. Calculate Pixel Area

After obtaining the segmentation mask, the system calculates the number of pixels belonging to the detected object.

The basic concept is:

```text
Pixel Area = Number of pixels inside the segmentation mask
```

For example:

```text
Segmentation Mask

0 0 0 0 0
0 1 1 1 0
0 1 1 1 0
0 1 1 1 0
0 0 0 0 0

Pixel Area = 9 pixels
```

---

### 4. Convert Pixel Area to Real Area

If the camera system is calibrated, the pixel area can be converted into a real-world measurement.

For example:

```text
Physical Area = Pixel Area × Conversion Factor
```

The physical area can be represented as:

```text
mm²
cm²
m²
```

depending on the calibration configuration.

---

## Area Calibration

Accurate physical-area measurement requires camera calibration.

A typical calibration process is:

```text
Reference Object
       │
       ▼
Known Physical Size
       │
       ▼
Measure Pixel Size
       │
       ▼
Calculate Scale
       │
       ▼
Pixel → Real-world Conversion
```

For example, if a reference object has a known physical area, the system can calculate a conversion factor between pixels and real-world dimensions.

The accuracy of the final measurement depends on:

- Camera resolution
- Camera distance
- Camera angle
- Lens distortion
- Lighting conditions
- Segmentation accuracy
- Object position
- Calibration accuracy

For consistent measurements, the camera should preferably remain in a fixed position.

---

## Requirements

The project requires:

- Python 3.x
- OpenCV
- NumPy
- ONNX Runtime
- GUI framework used by the project
- Compatible camera

Typical Python packages include:

```bash
pip install opencv-python
pip install numpy
pip install onnxruntime
```

If the GUI uses PyQt5:

```bash
pip install PyQt5
```

---

## Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
```

Go to the project directory:

```bash
cd code_calculated_the_area_by_camera
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
```

Activate the environment:

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` does not exist, install the required packages manually.

---

## Running the Project

Run the main GUI application:

```bash
python main_gui_test.py
```

The GUI will start the camera and inference workflow.

The main area calculation logic is implemented in:

```text
area_calc.py
```

---

## Main Files

| File / Directory | Description |
|---|---|
| `captures/` | Captured images and test data |
| `area_calc.py` | Area calculation and image processing |
| `main_gui_test.py` | Main GUI application |
| `best_yolo_seg_309.onnx` | YOLO segmentation model |
| `README.md` | Project documentation |
| `LICENSE` | MIT License |

---

## Model

The project uses:

```text
best_yolo_seg_309.onnx
```

The ONNX model is responsible for detecting and segmenting the target object.

The general inference pipeline is:

```text
Image
  │
  ▼
Pre-processing
  │
  ▼
ONNX Runtime
  │
  ▼
YOLO Segmentation
  │
  ▼
Post-processing
  │
  ▼
Segmentation Mask
```

The model can potentially be replaced with another compatible YOLO segmentation ONNX model, provided that the preprocessing and post-processing code is updated accordingly.

---

## Captures

The `captures/` directory contains images captured from the camera or used during testing.

Example:

```text
captures/
├── image_001.jpg
├── image_002.jpg
├── image_003.jpg
└── ...
```

These images can be used for:

- Testing the segmentation model
- Debugging
- Comparing area measurements
- Evaluating detection accuracy
- Testing different lighting conditions

---

## Accuracy Considerations

The system's area measurement accuracy depends on both the segmentation model and the camera calibration.

Important factors include:

### Camera

The camera should ideally have a fixed:

- Position
- Height
- Viewing angle
- Focus
- Resolution

### Lighting

Consistent lighting helps improve segmentation accuracy.

Strong shadows, reflections, or insufficient lighting may reduce detection accuracy.

### Calibration

The pixel-to-real-world conversion should be calibrated for the specific camera setup.

A segmentation mask alone provides an area in pixels. It does not automatically provide an accurate physical measurement.

---

## Troubleshooting

### Camera is not detected

Check that the camera is connected and available.

If OpenCV is used, camera indexes normally start from:

```text
0
1
2
...
```

You may need to change the camera index in the Python code.

---

### ONNX model cannot be loaded

Make sure the model exists:

```text
best_yolo_seg_309.onnx
```

Check that the model path used by the Python code is correct.

---

### Segmentation result is incorrect

Possible causes include:

- Incorrect model input size
- Incorrect preprocessing
- Incorrect confidence threshold
- Incorrect segmentation post-processing
- Poor lighting
- Camera angle
- Target object differs from the training data

---

### Area calculation is inaccurate

Check the calibration first.

The following factors can affect the result:

```text
Camera Position
       +
Camera Angle
       +
Lens Distortion
       +
Object Distance
       +
Segmentation Accuracy
       +
Calibration
       ↓
Final Area Accuracy
```

---

## Future Improvements

- [ ] Improve GUI design
- [ ] Add automatic camera calibration
- [ ] Add perspective correction
- [ ] Add lens distortion correction
- [ ] Add real-time FPS display
- [ ] Display segmentation confidence
- [ ] Support multiple objects
- [ ] Save measurement history
- [ ] Export measurement results to CSV
- [ ] Add measurement statistics
- [ ] Add automatic reference-object detection
- [ ] Improve segmentation accuracy
- [ ] Add GPU acceleration
- [ ] Add automated testing
- [ ] Create `requirements.txt`
- [ ] Package the application for Windows/Linux

---

## Architecture

The overall application architecture can be summarized as:

```text
┌─────────────────┐
│     Camera      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Image Capture   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Pre-processing  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ YOLO Segmentation│
│      ONNX       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Segmentation    │
│     Mask        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Area Calculator │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      GUI        │
└─────────────────┘
```

---

## Technologies

The project is based on:

- **Python**
- **OpenCV**
- **NumPy**
- **YOLO Segmentation**
- **ONNX**
- **ONNX Runtime**
- **Computer Vision**
- **Camera-based Measurement**
- **GUI Application**

---

## Project Objective

The main objective of this project is to develop a computer-vision-based system capable of:

1. Capturing an object using a camera.
2. Detecting and segmenting the object using a YOLO segmentation model.
3. Extracting the object's segmentation mask.
4. Calculating the object's pixel area.
5. Converting the pixel area into a physical area using calibration.
6. Displaying the result through a graphical user interface.

---

## License

This project is licensed under the MIT License.

See the `LICENSE` file for more information.

---

## Author

**tienvovn**

GitHub:

```text
https://github.com/tienvovn
```

---

## Notes

This project is currently a prototype for camera-based object-area measurement.

Before using the system for applications requiring high measurement accuracy, the results should be validated against physical measurements and an appropriate calibration procedure.
