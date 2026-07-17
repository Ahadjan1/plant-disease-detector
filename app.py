import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import json

# Page config
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="centered")

# Load model and class labels (cached so it doesn't reload every interaction)
@st.cache_resource
def load_trained_model():
    model = load_model("plant_disease_mobilenet.keras")
    with open("class_labels.json", "r") as f:
        class_labels = json.load(f)
    return model, class_labels

model, class_labels = load_trained_model()

# UI
st.title("🌿 Plant Disease Detector")
st.write("Upload a **Tomato** or **Potato** leaf image to detect disease.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess (same as training: resize to 224x224, rescale)
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    with st.spinner("Analyzing..."):
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        predicted_class = class_labels[predicted_index]

    # Confidence threshold warning
    if confidence < 50:
        st.warning(f"⚠️ Model is not confident (confidence: {confidence:.2f}%). "
                   "Please ensure you uploaded a Tomato or Potato leaf image.")
    else:
        # Format class name nicely
        display_name = predicted_class.replace("___", " - ").replace("_", " ")
        st.success(f"**Prediction:** {display_name}")
        st.info(f"**Confidence:** {confidence:.2f}%")

        # Show top-3 predictions
        st.subheader("Top 3 Predictions")
        top3_idx = np.argsort(predictions[0])[-3:][::-1]
        for idx in top3_idx:
            name = class_labels[idx].replace("___", " - ").replace("_", " ")
            conf = predictions[0][idx] * 100
            st.write(f"- {name}: {conf:.2f}%")

st.markdown("---")
st.caption("Model: MobileNetV2 (Transfer Learning) | Trained on PlantVillage dataset (Tomato + Potato classes)")