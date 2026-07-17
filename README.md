# 🌿 Plant Disease Detector

A deep learning web app that detects diseases in Tomato and Potato leaves using Transfer Learning (MobileNetV2).

🔗 **Live Demo:** [plant-disease-detector-r.streamlit.app](https://plant-disease-detector-r.streamlit.app/)

## 📊 Overview

This project classifies leaf images into 13 categories (10 Tomato diseases/healthy + 3 Potato diseases/healthy) using a CNN model trained on the PlantVillage dataset.

## 🧠 Approach

Two models were built and compared:

| Model | Test Accuracy |
|---|---|
| Custom CNN (from scratch) | 69.62% |
| **MobileNetV2 (Transfer Learning)** | **91.91%** |

Transfer learning significantly outperformed training a CNN from scratch, since MobileNetV2 leverages features pretrained on ImageNet (14M+ images), requiring far less data to generalize well.

## 🗂️ Dataset

- **Source:** [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Classes used:** Tomato (10 classes) + Potato (3 classes) = 13 total
- **Split:** 80% train / 20% test

## 🏗️ Model Architecture

- Base: MobileNetV2 (pretrained on ImageNet, frozen)
- Custom classification head: GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(13, Softmax)
- Only 165,645 params trained (out of 2.4M total)

## 🚀 Running Locally

```bash
git clone https://github.com/Ahadjan1/plant-disease-detector.git
cd plant-disease-detector
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run app.py
```

## ⚠️ Known Limitations

- Model is trained specifically on **leaf images** — it does not detect out-of-distribution inputs (e.g., uploading an image of the actual potato/tomato fruit instead of a leaf may still produce a confident but incorrect prediction).
- A confidence threshold warning (<50%) is shown, but doesn't catch all out-of-distribution cases.

## 🛠️ Tech Stack

- TensorFlow / Keras
- MobileNetV2 (Transfer Learning)
- Streamlit (Frontend)
- Google Colab (Training environment)

## 👤 Author
- GitHub: [@Ahadjan1](https://github.com/Ahadjan1)
- LinkedIn: [ahad-jan](https://www.linkedin.com/in/ahad-jan)

**Ahad Jan**
- GitHub: [@Ahadjan1](https://github.com/Ahadjan1)
- LinkedIn: [ahad-jan](https://www.linkedin.com/in/ahad-jan)
