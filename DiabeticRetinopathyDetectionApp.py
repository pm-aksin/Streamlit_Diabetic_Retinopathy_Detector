import streamlit as st
import sqlite3
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import os
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = None
if 'image_path' not in st.session_state:
    st.session_state['image_path'] = None
if 'user_data' not in st.session_state:
    st.session_state['user_data'] = None

# Initialize database
def init_db():
    conn = sqlite3.connect('patients.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patient (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    age INTEGER,
                    gender TEXT,
                    address TEXT,
                    email TEXT,
                    phone TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS test_record (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER,
                    test_date TEXT,
                    test_type TEXT,
                    result TEXT,
                    model TEXT,
                    confidence REAL,
                    FOREIGN KEY(patient_id) REFERENCES patient(id))''')
    conn.commit()
    conn.close()

# Add new patient
def add_patient(name, age, gender, address, email, phone):
    conn = sqlite3.connect('patients.db')
    c = conn.cursor()
    c.execute("INSERT INTO patient (name, age, gender, address, email, phone) VALUES (?, ?, ?, ?, ?, ?)", (name, age, gender, address, email, phone))
    conn.commit()
    conn.close()

# Fetch patients
def get_patients():
    conn = sqlite3.connect('patients.db')
    df = pd.read_sql_query("SELECT * FROM patient", conn)
    conn.close()
    return df

def save_test_record(patient_id, model, result, confidence):
    conn = sqlite3.connect('patients.db')
    c = conn.cursor()
    c.execute("INSERT INTO test_record (patient_id, test_date, test_type, result, model, confidence) VALUES (?, date('now'), ?, ?, ?, ?)", (patient_id, 'Retinopathy', result, model, confidence))
    conn.commit()
    conn.close()

# Cache model loading
@st.cache_resource
def load_model(_architecture, path, num_classes=5):
    model = _architecture(pretrained=False)
    if isinstance(model, models.ResNet):
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif isinstance(model, models.DenseNet):
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    elif isinstance(model, models.EfficientNet):
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Load models once
resnet50 = load_model(models.resnet50, 'Resnet50_8020Split_EarlyStop.pth')
densenet121 = load_model(models.densenet121, 'Densenet121_8020Split_EarlyStop.pth')
efficientnet = load_model(models.efficientnet_b0, 'EfficientNet_8020_EarlyStop.pth')
models_dict = {'ResNet50': resnet50, 'DenseNet121': densenet121, 'EfficientNet': efficientnet}

# PDF Generation
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Diabetic Retinopathy Test Result', 0, 1, 'C')

    def add_user_details(self, data, image_path=None):
        self.set_font('Arial', '', 12)
        if image_path:
            self.image(image_path, x=10, y=self.get_y(), w=40)
            self.ln(45)
        for key, value in data.items():
            self.cell(50, 10, f'{key}:', 0, 0)
            self.multi_cell(0, 10, str(value))

def export_pdf(user_data, image_path):
    pdf = PDF()
    pdf.add_page()
    pdf.add_user_details(user_data, image_path)
    pdf_output = "Diabetic_Retinopathy_Result.pdf"
    pdf.output(pdf_output)
    return pdf_output

# Transform for image processing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(image, model):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)[0]
        confidence, prediction = torch.max(probabilities, 0)
    return prediction.item(), confidence.item()

# App
st.title("👁️ Diabetic Retinopathy Detection")
init_db()

# Tabs
tab1, tab2, tab3 = st.tabs(["📝 Add Patient", "📋 Test Records", "🔍 Retinopathy Analysis"])

with tab1:
    st.header("⚕️ Patient Selection")
    name = st.text_input("👤 Name")
    age = st.number_input("🎂 Age", min_value=0)
    gender = st.selectbox("⚧️ Gender", ["Male", "Female", "Other"])
    address = st.text_area("🏠 Address")
    email = st.text_input("📧 Email")
    phone = st.text_input("📞 Phone Number")
    if st.button("➕ Add Patient"):
        add_patient(name, age, gender, address, email, phone)
        st.success(f"Patient {name} added successfully")

with tab2:
    st.header("Patient Test Records")
    patients = get_patients()
    patient_id = st.selectbox("Select Patient", patients["id"].astype(str) + " - " + patients["name"],key='patient_tab2')
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    if st.button("Show Records"):
        patient_id = int(patient_id.split(" - ")[0])
        conn = sqlite3.connect('patients.db')
        query = "SELECT * FROM test_record WHERE patient_id = ? AND test_date BETWEEN ? AND ?"
        records = pd.read_sql_query(query, conn, params=(patient_id, start_date, end_date))
        conn.close()
        st.write(records)

with tab3:
    st.header("📸 Upload Retinal Scan")
    patients = get_patients()
    patient_selection = st.selectbox("Select Patient", patients['id'].astype(str) + ' - ' + patients['name'])
    patient_id = int(patient_selection.split(' - ')[0])
    patient_data = patients[patients['id'] == patient_id].iloc[0]
    uploaded_image = st.file_uploader("Upload Retinopathy Image", type=["png", "jpg", "jpeg"])
    selected_models = st.multiselect("🧠 Select Models for Classification", list(models_dict.keys()), default=list(models_dict.keys()))
    diagnosis_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    st.subheader("🔍 Classification Results")
    if uploaded_image:
        results = []
        confidences = []
        image = Image.open(uploaded_image)
        st.session_state['image_path'] = f"uploads/{patient_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        os.makedirs("uploads", exist_ok=True)
        image.save(st.session_state['image_path'])
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze") and uploaded_image is not None:
            image = Image.open(uploaded_image).convert('RGB')
            
            model_results = {}  # Store results for JSON export
            for model_name in selected_models:
                model = models_dict[model_name]
                result, confidence = predict_image(image, model)
                results.append(result)
                confidences.append(confidence)
                save_test_record(patient_id, model_name, result, confidence)
                #st.write(f"{model_name} Prediction: {result} with Confidence: {confidence:.2f}")
                #model_results[model_name] = {"Prediction": result, "Confidence": f"{confidence:.2f}"}
                st.write(f"{model_name} Prediction: {diagnosis_labels[result]} with Confidence: {confidence:.2f}")
                model_results[model_name] = {"Prediction": diagnosis_labels[result], "Confidence": f"{confidence:.2f}"}

            avg_prediction = round(np.mean(results))
            majority_vote = max(set(results), key=results.count)
            avg_confidence = np.mean(confidences)
            #st.write(f"Average Prediction: {avg_prediction} with Confidence: {avg_confidence:.2f}")
            #st.write(f"Majority Voting Prediction: {majority_vote}")
            st.write(f"Average Prediction: {diagnosis_labels[avg_prediction]} with Confidence: {avg_confidence:.2f}")
            st.write(f"Majority Voting Prediction: {diagnosis_labels[majority_vote]}")
            save_test_record(patient_id, 'Average', avg_prediction, avg_confidence)
            save_test_record(patient_id, 'Majority Voting', majority_vote, 1.0)

            # Store analysis results
            st.session_state['user_data'] = {
                "Patient Id": patient_id,
                "Name": patient_data['name'],
                "Age": patient_data['age'],
                "Gender": patient_data['gender'],
                "Address": patient_data['address'],
                "Email": patient_data['email'],
                "Phone": patient_data['phone'],
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Results": model_results,
                "Average Prediction": f"{diagnosis_labels[avg_prediction]} with Confidence: {avg_confidence:.2f}",
                "Majority Voting Prediction": diagnosis_labels[majority_vote]
            }
            st.write(st.session_state['user_data'])
    # Export PDF Button
    if st.session_state['user_data'] and st.button("📄 Export Result as PDF"):
        pdf_path = export_pdf(st.session_state['user_data'], st.session_state['image_path'])
        with open(pdf_path, "rb") as f:
            st.download_button("⬇️ Download PDF", f, file_name=pdf_path, mime="application/pdf")
    # image_path is undefined here; use st.session_state['image_path']
            if 'image_path' in st.session_state and os.path.exists(st.session_state['image_path']):
                os.remove(st.session_state['image_path'])
            
            # Clean up session state
            for key in ['analysis_results', 'image_path', 'user_data']:
                st.session_state.pop(key, None)

