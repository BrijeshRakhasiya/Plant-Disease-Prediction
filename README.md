# 🌿 Plant Disease Prediction using CNN

This project is a deep learning-based web application that classifies plant leaf images to detect diseases using a Convolutional Neural Network (CNN). The app is built with **TensorFlow**, **Keras**, and **Streamlit**, and trained on the PlantVillage dataset.

---

## 🔍 Project Overview

- 🔬 **Model Type**: Convolutional Neural Network (CNN)
- 🎯 **Validation Accuracy**: `88.04%`
- 🖼️ **Frontend**: Streamlit web app
- 🧠 **Backend**: TensorFlow (Keras model)
- 📦 **Dataset**: [PlantVillage - Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

---

## 📂 Dataset

- **Source**: [Kaggle - PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Classes**: 38
- **Images**: 50,000+ colored images of healthy and diseased leaves.

---

## 🚀 Features

- Upload any leaf image (JPG/PNG).
- Classifies into one of 38 plant disease categories.
- Beautiful and interactive Streamlit interface.
- Lightweight and fast predictions.
- Fully offline, client-side inference.

---

## 🧪 Model Training (Brief)

> *Model training details available in `plant-disease-prediction-with-cnn.ipynb`.*

- Preprocessing: Resizing, normalization
- Architecture: Custom CNN with Conv2D, MaxPooling, Dropout, Dense layers
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: 20+

---


## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/plant-disease-prediction.git
cd plant-disease-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```
streamlit run main.py

```

# 📌 Requirements
Python 3.8+

TensorFlow 2.x

Streamlit

NumPy, Pillow, JSON

# 📈 Result
Validation Accuracy: 88.04%

Lightweight CNN architecture for quick inference.

# 🙋‍♂️ Author
Brijesh Rakhasiya
AI/ML Enthusiast | Data Scientist | Problem Solver


## 📄 License

This project is licensed under the MIT License.

---
**Made ❤️ by Brijesh Rakhasiya**
