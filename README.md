Vehicle Parking Detection-YOLOv8 

An AI-powered parking space detection system built using **Ultralytics YOLOv8**. This project enables real-time detection of available parking spaces in images and videos, with support for a **Streamlit web interface** for easy user interaction.
 📌 Features
- **Train & Validate YOLOv8** on a custom parking dataset  
- **Run Inference** on images and videos  
- **Confusion Matrix Visualization** for model performance analysis  
- **Streamlit Web App** for user-friendly image & video processing  
- **Google Drive Integration** to save trained models  

## 📁 Project Structure
```
├── dataset/              # YOLOv8 training dataset (from Roboflow)
├── runs/                 # YOLOv8 training and validation results
├── app.py                # Streamlit web application
├── train.py              # Model training script
├── inference.py          # Inference script for image/video
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

## 🚀 Quick Start

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Train the YOLOv8 Model
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  
model.train(data="path/to/data.yaml", epochs=20, imgsz=640, device="cuda")
```

### 3️⃣ Run Inference on an Image
```python
model.predict(source="image.jpg", save=True, conf=0.5)
```

### 4️⃣ Start the Web App
```bash
streamlit run app.py
```

## 📊 Visualizing the Confusion Matrix
```python
model.val()
```
The confusion matrix is saved in `runs/detect/val/confusion_matrix.png`.

## 📄 License
This project is open-source under the **MIT License**.

