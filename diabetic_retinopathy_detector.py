import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import torchvision.models as models
import torch.nn.functional as F
import io
from reportlab.pdfgen import canvas
import sqlite3
import os
import datetime

# ====================== DATABASE SETUP ======================
def init_db():
    """Initialize SQLite database and create table if it doesn't exist."""
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS scans (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT,
                 age INTEGER,
                 gender TEXT,
                 contact TEXT,
                 model_used TEXT,
                 diagnosis TEXT,
                 confidence REAL,
                 image_path TEXT,
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()  # Call function to ensure DB is initialized

# ====================== MODEL LOADING ======================
@st.cache_resource
def load_models():
    """Load pre-trained models for diabetic retinopathy classification."""
#    resnet50 = models.resnet50(pretrained=False)
#    resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, 2)
#    resnet50.load_state_dict(torch.load("resnet50_diabetic_retinopathy.pth", map_location=torch.device('cpu')))
#    resnet50.eval()

    efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    efficientnet.classifier[1] = torch.nn.Linear(efficientnet.classifier[1].in_features, 5)  # Match 5 output classes

    # Load trained weights
    efficientnet.load_state_dict(torch.load("efficientnet_diabetic_retinopathy.pth", map_location=torch.device('cpu')))
    efficientnet.eval()

    #return resnet50, efficientnet
    return efficientnet

#resnet50, efficientnet = load_models()
efficientnet = load_models()

# ====================== IMAGE TRANSFORM ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ====================== PREDICTION FUNCTION ======================
def predict(image, model):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    return prediction.item(), confidence.item()

# ====================== SAVE SCAN TO DATABASE ======================
def save_scan(name, age, gender, contact, model_used, diagnosis, confidence, image_path):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("INSERT INTO scans (name, age, gender, contact, model_used, diagnosis, confidence, image_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (name, age, gender, contact, model_used, diagnosis, confidence, image_path))
    conn.commit()
    conn.close()

# ====================== STREAMLIT UI ======================
st.title("👁️ Diabetic Retinopathy Detection")

# === Patient Details Form ===
st.header("📝 Patient Details")
name = st.text_input("👤 Name")
age = st.number_input("📅 Age", min_value=1, max_value=120, step=1)
gender = st.selectbox("⚧️ Gender", ["Male", "Female", "Other"])
contact = st.text_input("📞 Contact Information")

# === Upload Retinal Scan ===
st.header("📸 Upload Retinal Scan")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Select model(s) for classification
    models_selected = st.multiselect("🧠 Select Models for Classification",
                                     ["ResNet50", "EfficientNet", "Ensemble"])

    if st.button("🔍 Classify"):
        if not name or not contact:
            st.error("❌ Please fill in patient details before proceeding.")
        else:
            results = {}
            image_path = f"uploads/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            os.makedirs("uploads", exist_ok=True)
            image.save(image_path)

            if "ResNet50" in models_selected:
                pred, conf = predict(image, resnet50)
                results["ResNet50"] = ("Diabetic Retinopathy" if pred == 1 else "No Diabetic Retinopathy", conf)

            if "EfficientNet" in models_selected:
                pred, conf = predict(image, efficientnet)
                results["EfficientNet"] = ("Diabetic Retinopathy" if pred == 1 else "No Diabetic Retinopathy", conf)

            if "Ensemble" in models_selected:
                preds, confs = [], []
                if "ResNet50" in results:
                    preds.append(1 if results["ResNet50"][0] == "Diabetic Retinopathy" else 0)
                    confs.append(results["ResNet50"][1])
                if "EfficientNet" in results:
                    preds.append(1 if results["EfficientNet"][0] == "Diabetic Retinopathy" else 0)
                    confs.append(results["EfficientNet"][1])

                ensemble_pred = 1 if sum(preds) >= len(preds) / 2 else 0
                ensemble_conf = sum(confs) / len(confs) if confs else 0
                results["Ensemble"] = ("Diabetic Retinopathy" if ensemble_pred == 1 else "No Diabetic Retinopathy", ensemble_conf)

            for model, (diagnosis, confidence) in results.items():
                st.write(f"**{model} Prediction:** {diagnosis} (Confidence: {confidence:.2f})")
                save_scan(name, age, gender, contact, model, diagnosis, confidence, image_path)

            # Generate PDF report
            if st.button("📄 Download Report"):
                buffer = io.BytesIO()
                c = canvas.Canvas(buffer)
                c.drawString(100, 750, "Diabetic Retinopathy Classification Report")
                c.drawString(100, 730, f"Name: {name}")
                c.drawString(100, 710, f"Age: {age}")
                c.drawString(100, 690, f"Gender: {gender}")
                c.drawString(100, 670, f"Contact: {contact}")
                c.drawString(100, 650, f"Models Used: {', '.join(models_selected)}")

                y = 620
                for model, (diagnosis, confidence) in results.items():
                    c.drawString(100, y, f"{model}: {diagnosis} (Confidence: {confidence:.2f})")
                    y -= 20

                c.drawImage(image_path, 100, y - 200, width=200, height=200)
                c.save()

                buffer.seek(0)
                st.download_button("📥 Download PDF Report", buffer, file_name="report.pdf", mime="application/pdf")

# === Scan History Section ===
st.header("📜 Scan History")
conn = sqlite3.connect("database.db")
c = conn.cursor()
c.execute("SELECT id, name, age, gender, contact, model_used, diagnosis, confidence, timestamp FROM scans ORDER BY timestamp DESC")
data = c.fetchall()
conn.close()

import pandas as pd
if data:
    df = pd.DataFrame(data, columns=["ID", "Name", "Age", "Gender", "Contact", "Model", "Diagnosis", "Confidence", "Timestamp"])
    st.dataframe(df)
else:
    st.write("ℹ️ No scan history available.")
