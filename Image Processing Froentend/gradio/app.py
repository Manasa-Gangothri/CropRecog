import gradio as gr
import numpy as np
import cv2
import joblib
import os
from skimage.feature import hog

# ===== LOAD MODEL + PREPROCESSORS =====
model = joblib.load("brfc_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# ===== LOAD CATEGORIES =====
categories = ['Cherry','lemon','cotton','jute']

# ===== PREDICTION FUNCTION =====
def predict_image(image):
    try:
        # Convert Gradio image (numpy array) to OpenCV format
        img = cv2.resize(image, (64, 64))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # HOG Feature Extraction
        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8,8),
            cells_per_block=(2,2),
            block_norm='L2-Hys'
        )

        features = features.reshape(1, -1)

        # Scaling + PCA
        features = scaler.transform(features)
        features = pca.transform(features)

        # Prediction
        pred = model.predict(features)[0]

        if pred < len(categories):
            return f"🌱 Predicted Crop: {categories[pred]}"
        else:
            return "Unknown Crop"

    except Exception as e:
        return f"Error: {str(e)}"

# ===== GRADIO UI =====
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="🌾 Crop Recognition System",
    description="Upload a crop image to predict its category"
)

interface.launch()