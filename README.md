# 🚗 Driver Drowsiness Detection

A real-time **Driver Drowsiness Detection System** built with **YOLO, OpenCV, PyTorch, and Streamlit**.

The system analyzes images, uploaded videos, and live webcam frames to detect whether a driver is **Awake** or **Drowsy**. It supports GPU acceleration through CUDA when a compatible NVIDIA GPU is available.

---

## ✨ Features

* 🧠 YOLO-based drowsiness detection
* 🟢 Awake detection
* 🔴 Drowsy detection
* 🖼️ Image inference
* 🎥 Video inference
* 📷 Live webcam detection
* ⚡ CUDA / GPU acceleration
* 🎚️ Adjustable detection confidence
* 🧩 Modular project structure
* 🌐 Streamlit web interface
* 📓 Training and evaluation notebook

---

## 🏗️ Project Structure

```text
Driver Drowsiness/
│
├── app.py
├── config.py
├── detector.py
├── camera.py
├── utils.py
├── requirements.txt
├── README.md
│
├── models/
│   └── best.pt
│
└── notebook/
    └── drowsiness_detection.ipynb
```

### File Responsibilities

| File / Folder      | Description                                       |
| ------------------ | ------------------------------------------------- |
| `app.py`           | Streamlit application and user interface          |
| `config.py`        | Model path, device, and application configuration |
| `detector.py`      | YOLO model loading and inference                  |
| `camera.py`        | Webcam initialization and frame handling          |
| `utils.py`         | Image and frame processing utilities              |
| `requirements.txt` | Python dependencies                               |
| `models/best.pt`   | Trained YOLO weights                              |
| `notebook/`        | Training and evaluation notebook                  |

---

## 🔄 Detection Pipeline

```text
                 Input
                   │
        ┌──────────┼──────────┐
        │          │          │
      Image      Video      Webcam
        │          │          │
        └──────────┼──────────┘
                   │
                   ▼
              YOLO Model
                   │
                   ▼
          Object Detection
                   │
             ┌─────┴─────┐
             │           │
           Awake       Drowsy
             │           │
           🟢 Green     🔴 Red
             │           │
             └─────┬─────┘
                   │
                   ▼
             Streamlit UI
```

---

## 🛠️ Technologies

* **Python**
* **YOLO**
* **Ultralytics**
* **PyTorch**
* **OpenCV**
* **Streamlit**
* **NumPy**
* **CUDA** (optional)

---

## 📋 Requirements

Recommended:

* Python 3.10+
* NVIDIA GPU with CUDA support *(optional)*
* Windows / Linux
* Webcam for live detection

The application can also run on CPU if CUDA is unavailable.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/driver-drowsiness-detection.git

cd driver-drowsiness-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model

Place the trained YOLO model inside the `models` directory:

```text
models/
└── best.pt
```

The application expects:

```text
Driver Drowsiness/
│
├── app.py
├── config.py
├── detector.py
├── camera.py
├── utils.py
│
└── models/
    └── best.pt
```

The model path is configured relative to the project directory.

---

## ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

Usually:

```text
http://localhost:8501
```

---

# 🖼️ Image Detection

Select the **Image** tab and upload:

* `.jpg`
* `.jpeg`
* `.png`

The YOLO model detects the driver's state and displays bounding boxes with confidence scores.

Example:

```text
Awake 0.94
Drowsy 0.87
```

---

# 🎥 Video Detection

Select the **Video** tab and upload a supported video:

```text
.mp4
.avi
.mov
.mkv
```

The application processes the video frame-by-frame and displays the detection results in the Streamlit interface.

---

# 📷 Webcam Detection

Select the **Webcam** tab and choose your camera index.

For example:

```text
Camera 0
Camera 1
```

Then click:

```text
Start Camera
```

If camera `0` does not work, try camera `1`.

---

# 📓 Notebook

The `notebook/` directory contains the Jupyter Notebook used for the model development and evaluation process.

```text
notebook/
└── Yolo.ipynb
```

The notebook contains the relevant workflow for:

* Dataset preparation
* Model training
* Model validation
* Performance evaluation
* Visualization of evaluation results

---

# 📊 Evaluation Results

The drowsiness detection model was evaluated on a validation set containing **1,890 images**:

* **809 Awake**
* **1,081 Drowsy**

The model demonstrates high precision and recall across both classes.

## Performance Metrics

| Class           | Images | Instances | Box(P) | Recall | mAP50 | mAP50-95 |
| :-------------- | :----: | :-------: | :----: | :----: | :---: | :------: |
| **All Classes** |  1890  |    1890   |  0.989 |  0.988 | 0.992 |   0.906  |
| **Awake**       |   809  |    809    |  0.987 |  0.986 | 0.990 |   0.909  |
| **Drowsy**      |  1081  |    1081   |  0.990 |  0.989 | 0.993 |   0.904  |

### 🚀 Speed & Efficiency

| Metric             |        Time       |
| :----------------- | :---------------: |
| Preprocess         |    1.2 ms/image   |
| Inference          |    3.8 ms/image   |
| Postprocess        |    1.3 ms/image   |
| **Total Pipeline** | **~6.3 ms/image** |

The measured inference time indicates that the model is suitable for near real-time computer vision applications, depending on the hardware and input resolution.

---

## 📈 Confusion Matrix Insights

The evaluation produced the following results:

* **True Positives — Awake:** 797 instances correctly classified as Awake.
* **True Positives — Drowsy:** 1,067 instances correctly classified as Drowsy.
* **Awake → Drowsy:** 14 instances were incorrectly classified.
* **Drowsy → Awake:** 12 instances were incorrectly classified.
* **Background False Positives:** Minimal background confusion, with 4 Awake and 10 Drowsy false-positive instances.

These results indicate strong discrimination between the **Awake** and **Drowsy** classes on the validation dataset.

---

## ⚡ GPU Acceleration

The application automatically checks whether CUDA is available:

```python
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
```

If CUDA is available:

```text
Using device: cuda:0
```

Otherwise:

```text
Using device: cpu
```

You can verify your PyTorch CUDA installation with:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 🎚️ Confidence Threshold

The Streamlit sidebar provides a confidence slider.

For example:

```text
0.50
```

means detections below 50% confidence are filtered out.

Increasing the threshold generally produces fewer but more confident detections.

---

## 🧪 Detection Classes

The model supports two classes:

```text
0 → Awake
1 → Drowsy
```

The application displays the detected class and confidence score:

```text
Awake 0.96
```

or:

```text
Drowsy 0.91
```

---

## 🔮 Future Improvements

* [ ] Add temporal drowsiness analysis
* [ ] Add Eye Aspect Ratio (EAR)
* [ ] Add PERCLOS-based detection
* [ ] Add continuous drowsiness scoring
* [ ] Add audio alarm when drowsiness is detected
* [ ] Add detection history
* [ ] Add FPS monitoring
* [ ] Improve webcam streaming performance
* [ ] Add REST API using FastAPI
* [ ] Add Docker support
* [ ] Add support for multiple drivers
* [ ] Improve detection under low-light conditions
* [ ] Handle sunglasses and partially occluded faces

---

## ⚠️ Limitations

This project is intended as a computer vision demonstration and should not be considered a certified automotive safety system.

Detection performance can be affected by:

* Poor lighting
* Camera quality
* Driver pose
* Face occlusion
* Sunglasses
* Motion blur
* Low-resolution video
* Extreme head angles

---

## 👨‍💻 Author
**Seif Nasser**
**Bassam**
**Menna Farouk**
**Amr*



---

## 📄 License

This project is available for educational and research purposes.

If you use this project or its model in another project, please provide appropriate attribution.
