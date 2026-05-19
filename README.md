# Ai-Deep-Fake


# AI-Based Deepfake and Face Swap Detection System

## Overview
This project is an AI-powered Deepfake and Face Swap Detection System developed to identify manipulated images using Deep Learning techniques. The system helps detect fake or AI-generated media to reduce misinformation, privacy misuse, and digital fraud.

The project was developed during the Codeastra Hackathon 2025, where our team ranked in the Top 10 among 250+ teams.

---

# Features
- Upload image for detection
- Detect real vs fake images
- CNN-based Deep Learning model
- Flask web application integration
- Confidence score prediction
- Image preprocessing pipeline
- Error handling for invalid uploads

---

# Technologies Used

## Programming Language
- Python

## Frameworks & Libraries
- Flask
- TensorFlow / Keras
- OpenCV
- NumPy

## Deep Learning
- Convolutional Neural Network (CNN)

---

# Project Workflow

1. User uploads an image through the Flask web interface.
2. The uploaded image is stored temporarily.
3. Image preprocessing is applied:
   - resizing
   - normalization
   - array conversion
4. The processed image is passed into the trained CNN model.
5. The model predicts whether the image is:
   - Real Image
   - Fake Image
6. Prediction result and confidence score are displayed.

---

# Folder Structure

```bash
project/
│
├── app.py
├── utils.py
├── requirements.txt
├── model/
│   └── fake_image_model.h5
│
├── static/
│   └── uploads/
│
├── templates/
│   └── index.html
│
└── README.md
```

---

# Model Information

- Model Type: CNN (Convolutional Neural Network)
- Framework: TensorFlow/Keras
- Classification Type: Binary Classification
- Output:
  - Real Image
  - Fake Image

---

# Image Preprocessing

The preprocessing pipeline includes:
- Image resizing
- Pixel normalization
- Noise handling
- Input shape conversion

These steps improve model accuracy and prediction consistency.

---

# Results

- Achieved 85%+ detection accuracy
- Reduced false positives significantly
- Ranked 9th among 250+ teams at Codeastra Hackathon 2025

---

# Challenges Faced

- Detecting highly realistic deepfake images
- Handling inconsistent image quality
- Reducing false positives
- Managing limited training time during hackathon

---

# Future Improvements

- Real-time video deepfake detection
- Advanced transformer-based models
- Improved dataset scaling
- Deployment on cloud platforms
- Original media restoration using inverse deepfake techniques

---

# Installation

## Clone Repository

```bash
git clone <your-github-repo-link>
cd <project-folder>
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Application

```bash
python app.py
```

---

# Requirements

Example dependencies:

```txt
tensorflow
flask
opencv-python
numpy
```

---

# Screenshots

## Upload Interface
(Add screenshot here)

## Prediction Result
(Add prediction output screenshot here)

---

# Learning Outcomes

Through this project, I gained practical experience in:
- Deep Learning
- CNN architecture
- Flask integration
- Image preprocessing
- Model deployment workflow
- Team collaboration
- AI-based image classification

---

# Author

Aditya Kumar Jha

- GitHub:(https://github.com/AdityaCypherX)
- LinkedIn: (https://www.linkedin.com/in/adityajha01/)
