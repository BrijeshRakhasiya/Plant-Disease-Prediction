import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Page configuration
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="centered")

# Load model and class indices
model_path = r"F:\30 Days\19. Plant Disease Prediction\model\plant_disease_prediction_model.h5"
json_path = r"F:\30 Days\19. Plant Disease Prediction\class_indices.json"

model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(json_path))


# Function to load and preprocess image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array


# Prediction function
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Custom CSS
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        color: #2e8b57;
        font-weight: bold;
    }
    .subheader {
        text-align: center;
        font-size: 1.2rem;
        color: #444;
        margin-bottom: 30px;
    }
    .stButton > button {
        background-color: #2e8b57;
        color: white;
        padding: 0.6em 1.5em;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #256d47;
        color: white;
    }
    .prediction-box {
        padding: 1em;
        border-radius: 10px;
        background-color: #e6ffe6;
        color: #2e8b57;
        font-size: 1.2rem;
        text-align: center;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<div class="main-title">🌿 Plant Disease Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload a leaf image to detect plant disease using deep learning</div>', unsafe_allow_html=True)

# File Uploader
uploaded_image = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image.resize((300, 300)), caption="Uploaded Leaf Image", use_column_width=True)

    with col2:
        st.markdown("### 🔍 Click below to classify the disease")
        if st.button("Classify"):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.markdown(f'<div class="prediction-box">🧠 Prediction: {prediction}</div>', unsafe_allow_html=True)
