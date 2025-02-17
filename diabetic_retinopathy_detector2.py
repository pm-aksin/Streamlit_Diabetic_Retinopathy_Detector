import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import sqlite3
import os
import datetime
import time
import random
import string
import io
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
import torch.nn.functional as F

# ====================== DATABASE SETUP ======================
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS scans (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 patient_id TEXT,
                 name TEXT,
                 age INTEGER,
                 gender TEXT,
                 contact TEXT,
                 model_used TEXT,
                 diagnosis INTEGER,
                 confidence REAL,
                 image_path TEXT,
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

# ====================== GENERATE PATIENT ID ======================
def generate_patient_id():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

# ====================== ANIMATED MODEL LOADING ======================
st.title("👁️ Diabetic Retinopathy Detection")

st.header("🧠 Loading Model...")
progress_bar = st.progress(0)
status_text = st.empty()
for percent in range(0, 101, 10):
    time.sleep(0.3)
    progress_bar.progress(percent)
    status_text.text(f"Loading... {percent}%")

@st.cache_resource
def load_model():
    model = timm.create_model('efficientnet_b0', pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 5)
    model.load_state_dict(torch.load("efficientnet_diabetic_retinopathy.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

efficientnet = load_model()
progress_bar.empty()
status_text.text("✅ Model Loaded Successfully!")

# ====================== IMAGE PROCESSING ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image, model):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    return prediction.item(), confidence.item()

# ====================== STREAMLIT UI ======================
st.header("📝 Patient Details")
patient_type = st.radio("Select Patient Type", ("New Patient", "Existing Patient"))

if patient_type == "New Patient":
    patient_id = generate_patient_id()
    st.write(f"Generated Patient ID: {patient_id}")
else:
    patient_id = st.text_input("Enter Patient ID")

name = st.text_input("👤 Name")
age = st.number_input("📅 Age", min_value=1, max_value=120, step=1)
gender = st.selectbox("⚧️ Gender", ["Male", "Female", "Other"])
contact = st.text_input("📞 Contact Information")

st.header("📸 Upload Retinal Scan")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Classify"):
        if not name or not contact or not patient_id:
            st.error("❌ Please fill in patient details before proceeding.")
        else:
            prediction, confidence = predict(image, efficientnet)
            diagnosis_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
            diagnosis = diagnosis_labels[prediction]

            image_path = f"uploads/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            os.makedirs("uploads", exist_ok=True)
            image.save(image_path)

            conn = sqlite3.connect("database.db")
            c = conn.cursor()
            c.execute("INSERT INTO scans (patient_id, name, age, gender, contact, model_used, diagnosis, confidence, image_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (patient_id, name, age, gender, contact, "EfficientNet", prediction, confidence, image_path))
            conn.commit()
            conn.close()

            st.success(f"✅ Diagnosis: {diagnosis} (Confidence: {confidence:.2f})")

            if st.button("📄 Download Report"):
                buffer = io.BytesIO()
                c = canvas.Canvas(buffer)
                c.drawString(100, 750, "Diabetic Retinopathy Classification Report")
                c.drawString(100, 730, f"Patient ID: {patient_id}")
                c.drawString(100, 710, f"Name: {name}")
                c.drawString(100, 690, f"Age: {age}")
                c.drawString(100, 670, f"Gender: {gender}")
                c.drawString(100, 650, f"Diagnosis: {diagnosis} (Confidence: {confidence:.2f})")
                c.drawImage(image_path, 100, 400, width=200, height=200)
                c.save()
                buffer.seek(0)
                st.download_button("📥 Download PDF Report", buffer, file_name="report.pdf", mime="application/pdf")

# ====================== SCAN HISTORY GRAPHS ======================
st.header("📊 Scan History & Analysis")
conn = sqlite3.connect("database.db")
df = pd.read_sql("SELECT * FROM scans", conn)
conn.close()

if not df.empty:
    st.subheader("📈 Diagnosis Distribution")
    fig, ax = plt.subplots()
    df['diagnosis'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)
else:
    st.write("ℹ️ No scan history available.")
