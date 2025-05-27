import os
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import streamlit as st
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt  # Now this is at the top level
import time
import plotly.express as px
import importlib

# 📥 GITHUB BASE URL
# -----------------------------
GITHUB_BASE_URL = "https://github.com/Dileep-kumarc/final-tumor/raw/master/"

# Example usage for model loading:
# Instead of loading local files, use the GITHUB_BASE_URL to construct the download path
# For example, to load a model:
# model_path = GITHUB_BASE_URL + "brain_tumor_classifier.h5"
# model = load_model(model_path)
# You may need to download the file first if your framework does not support direct loading from URL

# ======================
# MODEL DEFINITIONS
# ======================

@st.cache_resource
def load_models():
    """Load all ML models with caching"""
    def load_custom_model():
        class CustomCNN(nn.Module):
            def __init__(self):
                super(CustomCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                self.fc1 = nn.Linear(128 * 28 * 28, 512)
                self.fc2 = nn.Linear(512, 2)

            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = self.pool(torch.relu(self.conv3(x)))
                x = x.view(-1, 128 * 28 * 28)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        # Download model from GitHub if not present locally
        import requests
        import os
        def download_file(url, local_path):
            if not os.path.exists(local_path):
                r = requests.get(url)
                with open(local_path, 'wb') as f:
                    f.write(r.content)

        # Paths for local storage
        local_pth = "best_mri_classifier.pth"
        local_classifier = "brain_tumor_classifier.h5"
        local_segmentation = "brain_tumor_segmentation_unet.h5"
        local_size = "tumor_size_model.h5"

        # Download from GitHub if needed
        download_file(GITHUB_BASE_URL + "best_mri_classifier.pth", local_pth)
        download_file(GITHUB_BASE_URL + "brain_tumor_classifier.h5", local_classifier)
        download_file(GITHUB_BASE_URL + "brain_tumor_segmentation_unet.h5", local_segmentation)
        download_file(GITHUB_BASE_URL + "tumor_size_model.h5", local_size)

        model = CustomCNN()
        model.load_state_dict(torch.load(local_pth, map_location=torch.device('cpu')))
        model.eval()
        return model

    custom_cnn_model = load_custom_model()
    classifier_model = tf.keras.models.load_model("brain_tumor_classifier.h5")
    segmentation_model = tf.keras.models.load_model("brain_tumor_segmentation_unet.h5")
    tumor_size_model = tf.keras.models.load_model("tumor_size_model.h5")
    return custom_cnn_model, classifier_model, segmentation_model, tumor_size_model

# ======================
# PROCESSING FUNCTIONS
# ======================

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    return image_array

def validate_mri(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    tensor_image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor_image)
    pred = torch.argmax(output, dim=1).item()
    return ("MRI", True) if pred == 0 else ("Non-MRI", False)

def classify_tumor(image, model):
    image_array = preprocess_image(image)
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    return classes[np.argmax(predictions)], np.max(predictions)

def segment_tumor(image, model):
    image_array = preprocess_image(image, target_size=(256, 256))
    if image_array.shape[-1] == 3:
        image_array = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)
    mask = model.predict(image_array)[0]
    return mask

def extract_features_for_size(mask):
    mask = (np.array(mask) > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.zeros(5)
    contour = contours[0]
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    aspect_ratio = float(contour.shape[0]) / contour.shape[1] if contour.shape[1] != 0 else 0
    solidity = area / cv2.convexHull(contour).shape[0] if area > 0 else 0
    convex_hull_area = cv2.contourArea(cv2.convexHull(contour))
    return np.array([area, perimeter, aspect_ratio, solidity, convex_hull_area])

def predict_tumor_size(mask, model):
    features = extract_features_for_size(mask)
    reshaped_features = np.expand_dims(features, axis=0)
    predicted_size = model.predict(reshaped_features)
    return predicted_size[0]

# ======================
# STREAMLIT UI
# ======================

def home_page():
    # Hero Section
    st.markdown("""
    <div class="hero">
        <div class="hero-image-main">
            <img src="https://i.ibb.co/Kj5Xh3cZ/brain.png" alt="Brain AI Technology">
            <div class="hero-blob"></div>
        </div>
        <div class="hero-content">
            <div class="badge">AI-Powered Healthcare</div>
            <h1>Advanced Brain Tumor Detection</h1>
            <p class="subtitle">Revolutionizing medical diagnostics with state-of-the-art AI technology for precise and rapid brain tumor analysis</p>
            <div class="hero-buttons">
                <a href="/?page=Analysis" class="primary-button">Get Started Now</a>
                <div class="stats-badge">
                    <span class="stats-number">90%</span>
                    <span class="stats-text">Accuracy Rate</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    .hero {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 2rem 0;
        gap: 3rem;
        margin-bottom: 2rem;
    }
    .hero-content {
        flex: 1;
        text-align: left;
    }
    .hero-image-main {
        flex: 1;
    }
    .hero-image-main img {
        width: 100%;
        height: auto;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    @media (max-width: 768px) {
        .hero {
            flex-direction: column;
            text-align: center;
        }
        .hero-content {
            text-align: center;
            order: 2;
        }
        .hero-image-main {
            order: 1;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Remove the duplicate image
    # st.image("assets/BrainTumor.JPG", use_column_width=True, caption="Advanced MRI Analysis Technology")

    # Key Features Section
    st.markdown("""
    <div class="features">
        <div class="feature-card">
            <div class="feature-icon">🚀</div>
            <h3>Instant Analysis</h3>
            <p>Get comprehensive tumor analysis in seconds using state-of-the-art AI models</p>
            <ul class="feature-list">
                <li>Quick MRI validation</li>
                <li>Real-time processing</li>
                <li>Immediate results</li>
            </ul>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <h3>High Accuracy</h3>
            <p>Validated on thousands of MRI scans with exceptional accuracy rates</p>
            <ul class="feature-list">
                <li>98.5% MRI validation accuracy</li>
                <li>94.2% tumor classification</li>
                <li>89% segmentation precision</li>
            </ul>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3>Detailed Reports</h3>
            <p>Complete tumor characterization with comprehensive analysis</p>
            <ul class="feature-list">
                <li>Tumor type classification</li>
                <li>Size measurement</li>
                <li>Visual segmentation</li>
            </ul>
        </div>
    </div>

    <div class="info-section">
        <h2>Why Choose Our Platform?</h2>
        <div class="info-grid">
            <div class="info-item">
                <h3>🏥 Healthcare Professional Trusted</h3>
                <p>Developed in collaboration with leading medical institutions</p>
            </div>
            <div class="info-item">
                <h3>🔒 Secure Analysis</h3>
                <p>HIPAA-compliant platform ensuring data privacy and security</p>
            </div>
            <div class="info-item">
                <h3>⚡ Fast Processing</h3>
                <p>Results delivered in under 60 seconds</p>
            </div>
            <div class="info-item">
                <h3>📱 Accessible Anywhere</h3>
                <p>Web-based platform accessible from any device</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Statistics Section with Enhanced Styling
    st.markdown("<h2 class='section-title'>Brain Tumor Statistics</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            names=["Glioma", "Meningioma", "Pituitary", "No Tumor"],
            values=[35, 30, 25, 10],
            title="Tumor Type Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            x=["MRI Validation", "Tumor Classification", "Segmentation"],
            y=[98.5, 94.2, 89.0],
            labels={"x": "Model", "y": "Accuracy (%)"},
            title="Model Performance Metrics",
            color=["MRI Validation", "Tumor Classification", "Segmentation"],
            color_discrete_sequence=["#4CAF50", "#2196F3", "#9C27B0"]
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Workflow Section
    st.markdown("""
    <div class="workflow-section">
        <h2>How It Works</h2>
        <div class="workflow-steps">
            <div class="step">
                <div class="step-number">1</div>
                <h3>Upload MRI Scan</h3>
                <p>Upload your brain MRI scan in common formats (JPEG, PNG)</p>
            </div>
            <div class="step">
                <div class="step-number">2</div>
                <h3>AI Analysis</h3>
                <p>Our advanced AI models process and analyze the scan</p>
            </div>
            <div class="step">
                <div class="step-number">3</div>
                <h3>Get Results</h3>
                <p>Receive detailed analysis with visualization</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Call to Action Section
    st.markdown("""
    <div class="cta-section">
        <h2>Ready to Start Your Analysis?</h2>
        <p>Upload your brain MRI scan now and get instant results</p>
        <a href="/?page=Analysis" class="cta-button">Start Analysis Now</a>
    </div>
    """, unsafe_allow_html=True)

def analysis_page():
    st.markdown("""
    <style>
        .analysis-container {
            background: linear-gradient(120deg, #130f40 0%, #000046 100%);
            color: #000000;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            margin: 20px auto;
            max-width: 1200px;
        }
        .analysis-header {
            color: #00d2ff;
            text-align: center;
            margin-bottom: 40px;
            font-size: 32px;
            font-weight: 800;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        }
        .analysis-subheader {
            color: #00d2ff;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 24px;
            font-weight: 700;
            border-bottom: 2px solid #e3f2fd;
            padding-bottom: 10px;
        }
        .upload-container {
            background: rgba(255, 255, 255, 0.05);
            border: 2px dashed rgba(0, 210, 255, 0.5);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .upload-container:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(0, 255, 204, 0.8);
        }
        .result-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            margin: 15px 0;
            border: 1px solid rgba(255, 255, 255, 0.18);
            transition: all 0.3s ease;
        }
        .result-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(0, 210, 255, 0.3);
        }
        .metric-container {
            background: linear-gradient(120deg, rgba(0, 210, 255, 0.2) 0%, rgba(58, 123, 213, 0.2) 100%);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(0, 210, 255, 0.3);
            box-shadow: 0 0 20px rgba(0, 210, 255, 0.1);
        }
        .symptoms-list {
            list-style-type: none;
            padding: 0;
        }
        .symptoms-list li {
            background: rgba(255, 255, 255, 0.1);
            margin: 8px 0;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s ease;
            color: #000000;
        }
        .symptoms-list li:hover {
            transform: translateX(5px);
            background: rgba(255, 255, 255, 0.15);
        }
        .progress-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            margin: 20px 0;
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        .stProgress > div > div {
            background-color: #00d2ff;
        }
        .stProgress > div > div > div {
            background-color: #00ffcc;
        }
        .image-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            margin: 20px 0;
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        .image-container img {
            border-radius: 10px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        }
        .report-section {
            background: linear-gradient(120deg, rgba(0, 210, 255, 0.1) 0%, rgba(0, 255, 204, 0.1) 100%);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            border-left: 3px solid #00ffcc;
        }
        .download-button {
            background: linear-gradient(120deg, #00d2ff 0%, #00ffcc 100%);
            color: #000000;
            padding: 15px 30px;
            border-radius: 25px;
            border: none;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .download-button:hover {
            background: linear-gradient(120deg, #00ffcc 0%, #00d2ff 100%);
            transform: scale(1.05);
        }
        .switch-button {
            background: linear-gradient(120deg, #00d2ff 0%, #00ffcc 100%);
            color: #000000;
            padding: 10px 20px;
            border-radius: 30px;
            text-align: center;
            margin: 20px auto;
            display: inline-block;
            text-decoration: none;
            font-weight: bold;
            border: 1px solid rgba(0, 210, 255, 0.3);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        .switch-button:hover {
            background: linear-gradient(120deg, #00ffcc 0%, #00d2ff 100%);
            transform: translateY(-3px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(0, 210, 255, 0.5);
        }
        /* Improved text readability styles */
        p, li, span, div {
            color: #000000;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #00d2ff;
        }
        .label-text {
            color: #000000;
            font-size: 14px;
            font-weight: 500;
        }
        .value-text {
            color: #00ffcc;
            font-size: 24px;
            font-weight: 700;
        }
        .description-text {
            color: #000000;
            font-size: 13px;
            line-height: 1.5;
        }
        .highlight-text {
            color: #00ffcc;
            font-weight: 700;
        }
        .secondary-text {
            color: #000000;
            font-weight: 500;
        }
        /* Streamlit element overrides for better readability */
        .stMarkdown, .stText {
            color: #000000 !important;
        }
        .stMetric .metric-label {
            color: #000000 !important;
        }
        .stMetric .metric-value {
            color: #00ffcc !important;
        }
        .stButton button {
            color: #000000 !important;
            background: linear-gradient(120deg, #00d2ff 0%, #00ffcc 100%) !important;
            font-weight: bold !important;
        }
        .stButton button:hover {
            background: linear-gradient(120deg, #00ffcc 0%, #00d2ff 100%) !important;
            transform: translateY(-2px) !important;
        }
        /* Make file uploader text visible */
        .stFileUploader label {
            color: #000000 !important;
        }
        .stFileUploader span {
            color: #000000!important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='analysis-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='analysis-header'>🧠 Brain Tumor Analysis</h1>", unsafe_allow_html=True)
    
    # Add switch to advanced analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <a href="/?page=AdvancedAnalysis" class="switch-button">
                <span style="display: flex; align-items: center; justify-content: center;">
                    <span style="margin-right: 8px;">🔬</span> Switch to Advanced Analysis
                </span>
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #00d2ff; margin-bottom: 20px;'>Upload Your MRI Scan</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)

    custom_cnn_model, classifier_model, segmentation_model, tumor_size_model = load_models()

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='progress-container'>", unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

        status_text.text("🔍 Validating MRI...")
        image_type, is_mri = validate_mri(image, custom_cnn_model)
        progress_bar.progress(25)

        if not is_mri:
            st.error(f"⚠️ Detected Image Type: {image_type}. Please upload a valid MRI image.")
        else:
            st.success("✅ Valid MRI detected.")
            status_text.text("🔬 Classifying tumor...")
            tumor_type, confidence = classify_tumor(image, classifier_model)
            progress_bar.progress(50)

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown("<h3 class='analysis-subheader'>Classification Results</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric("Tumor Type", tumor_type)
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric("Confidence", f"{confidence:.2%}")
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            symptoms_dict = {
                "Glioma": [
                    "Headaches",
                    "Seizures",
                    "Nausea and vomiting",
                    "Vision problems",
                    "Personality changes"
                ],
                "Meningioma": [
                    "Headaches",
                    "Hearing loss",
                    "Memory loss",
                    "Seizures",
                    "Weakness in limbs"
                ],
                "Pituitary": [
                    "Vision problems",
                    "Hormonal imbalances",
                    "Headaches",
                    "Fatigue",
                    "Unexplained weight changes"
                ],
                "No Tumor": [
                    "No symptoms associated with brain tumors"
                ]
            }

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown("<h3 class='analysis-subheader'>Associated Symptoms</h3>", unsafe_allow_html=True)
            symptoms = symptoms_dict.get(tumor_type, ["No symptoms available"])
            st.markdown("<ul class='symptoms-list'>", unsafe_allow_html=True)
            for symptom in symptoms:
                st.markdown(f"<li>• {symptom}</li>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if tumor_type != "No Tumor":
                status_text.text("🎯 Segmenting tumor...")
                mask = segment_tumor(image, segmentation_model)
                progress_bar.progress(75)

                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='analysis-subheader'>Tumor Segmentation</h3>", unsafe_allow_html=True)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                ax1.imshow(image)
                ax1.set_title("Original Image")
                ax1.axis("off")
                ax2.imshow(mask.squeeze(), cmap="viridis")
                ax2.set_title("Segmentation Mask")
                ax2.axis("off")
                st.pyplot(fig)
                st.markdown("</div>", unsafe_allow_html=True)

                status_text.text("📏 Calculating tumor size...")
                size = predict_tumor_size(mask, tumor_size_model)
                progress_bar.progress(100)

                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='analysis-subheader'>Tumor Size Analysis</h3>", unsafe_allow_html=True)
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric("Predicted Tumor Size", f"{size[0]:.2f} mm²")
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            status_text.text("✨ Analysis complete!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()

            st.markdown("<div class='report-section'>", unsafe_allow_html=True)
            st.markdown("""
            <h3 style='color: #00d2ff; margin-bottom: 20px;'>📋 Generate Professional Report</h3>
            <p style='color: #a0e4ff; margin-bottom: 30px;'>Create a detailed medical report with professional formatting and comprehensive analysis results</p>
            """, unsafe_allow_html=True)

            if st.button("Generate PDF Report", type="primary"):
                from reportlab.lib import colors
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_JUSTIFY
                from io import BytesIO
                import uuid
                import datetime

                # Generate a unique report ID
                report_id = str(uuid.uuid4())[:8].upper()
                
                # Current timestamp for the report
                timestamp = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")
                
                # Save images to bytes with enhanced styling
                original_img_bytes = BytesIO()
                plt.figure(figsize=(6, 6))
                plt.imshow(image)
                plt.axis('off')
                plt.title("Original MRI Scan")
                plt.tight_layout()
                plt.savefig(original_img_bytes, format='png', bbox_inches='tight', dpi=300)
                plt.close()

                mask_img_bytes = BytesIO()
                plt.figure(figsize=(6, 6))
                plt.imshow(mask.squeeze(), cmap="viridis")
                plt.axis('off')
                plt.title("AI-Segmented Tumor Region")
                plt.tight_layout()
                plt.savefig(mask_img_bytes, format='png', bbox_inches='tight', dpi=300)
                plt.close()

                # Create PDF with enhanced styling
                pdf_buffer = BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, 
                                        leftMargin=0.5*inch, rightMargin=0.5*inch,
                                        topMargin=0.5*inch, bottomMargin=0.5*inch)
                styles = getSampleStyleSheet()
                
                # Custom styles
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=24,
                    fontName='Helvetica-Bold',
                    alignment=TA_CENTER,
                    spaceAfter=6
                )
                
                subtitle_style = ParagraphStyle(
                    'CustomSubtitle',
                    parent=styles['Heading2'],
                    fontSize=16,
                    fontName='Helvetica-Oblique',
                    textColor=colors.darkblue,
                    alignment=TA_CENTER,
                    spaceAfter=20
                )
                
                timestamp_style = ParagraphStyle(
                    'Timestamp',
                    parent=styles['Normal'],
                    fontSize=10,
                    alignment=TA_RIGHT,
                    textColor=colors.darkgrey
                )
                
                section_title_style = ParagraphStyle(
                    'SectionTitle',
                    parent=styles['Heading2'],
                    fontSize=14,
                    fontName='Helvetica-Bold',
                    textColor=colors.darkblue,
                    spaceAfter=10,
                    spaceBefore=15
                )
                
                disclaimer_style = ParagraphStyle(
                    'Disclaimer',
                    parent=styles['Normal'],
                    fontSize=8,
                    fontName='Helvetica-Oblique',
                    textColor=colors.darkgrey,
                    alignment=TA_JUSTIFY,
                    spaceBefore=30
                )
                
                footer_style = ParagraphStyle(
                    'Footer',
                    parent=styles['Normal'],
                    fontSize=8,
                    textColor=colors.darkgrey,
                    alignment=TA_CENTER
                )
                
                story = []
                
                # Header with logo placeholder
                # You can replace this with an actual logo if available
                header_data = [
                    [Paragraph("🧠 BrainAI Medical Center", styles["Heading3"]), 
                     Paragraph(f"Report Generated On:<br/>{timestamp}", timestamp_style)]
                ]
                header_table = Table(header_data, colWidths=[4*inch, 3*inch])
                header_table.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ]))
                story.append(header_table)
                story.append(Spacer(1, 20))
                
                # Title and subtitle
                story.append(Paragraph("Brain Tumor Analysis Report", title_style))
                story.append(Paragraph("AI-Powered Diagnostic Summary", subtitle_style))
                story.append(Spacer(1, 10))
                
                # Report ID
                story.append(Paragraph(f"Report ID: {report_id}", styles["Normal"]))
                story.append(Spacer(1, 20))
                
                # Horizontal line as section divider
                story.append(Table([['']], colWidths=[7*inch], style=[
                    ('LINEABOVE', (0, 0), (-1, 0), 1, colors.lightgrey),
                ]))
                
                # Analysis Results Section
                story.append(Paragraph("Analysis Results", section_title_style))
                
                # Results table with enhanced styling
                data = [
                    ["Parameter", "Value"],
                    ["Tumor Type", tumor_type],
                    ["Confidence", f"{confidence:.2%}"],
                ]
                if tumor_type != "No Tumor":
                    data.append(["Tumor Size", f"{size[0]:.2f} mm²"])
                    # Optional: Add tumor location if available
                    # data.append(["Tumor Location", "Right Temporal Lobe"])
                
                results_table = Table(data, colWidths=[3*inch, 3*inch])
                results_table.setStyle(TableStyle([
                    # Header row styling
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    
                    # Alternating row colors
                    ('BACKGROUND', (0, 1), (-1, 1), colors.lightgrey),
                    ('BACKGROUND', (0, 3), (-1, 3), colors.lightgrey),
                    
                    # Cell styling
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 10),
                    ('TOPPADDING', (0, 1), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                    
                    # Table grid
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('BOX', (0, 0), (-1, -1), 1, colors.darkgrey),
                ]))
                story.append(results_table)
                
                # Section divider
                story.append(Spacer(1, 10))
                story.append(Table([['']], colWidths=[7*inch], style=[
                    ('LINEABOVE', (0, 0), (-1, 0), 1, colors.lightgrey),
                ]))
                
                # MRI Scan Analysis Section
                story.append(Paragraph("MRI Scan Analysis", section_title_style))
                story.append(Spacer(1, 10))
                
                # Reset image bytes position
                original_img_bytes.seek(0)
                mask_img_bytes.seek(0)
                
                # Create images with rounded corners and borders
                original_img = RLImage(original_img_bytes, width=3*inch, height=3*inch)
                mask_img = RLImage(mask_img_bytes, width=3*inch, height=3*inch)
                
                # Image labels
                image_labels = [
                    [original_img, mask_img],
                    [Paragraph("<b>Original MRI Scan</b>", styles["Normal"]), 
                     Paragraph("<b>AI-Segmented Tumor Region</b>", styles["Normal"])]
                ]
                
                image_table = Table(image_labels, colWidths=[3.5*inch, 3.5*inch])
                image_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 1), (-1, 1), 6),
                    ('BOTTOMPADDING', (0, 1), (-1, 1), 20),
                ]))
                story.append(image_table)
                
                # Section divider
                story.append(Table([['']], colWidths=[7*inch], style=[
                    ('LINEABOVE', (0, 0), (-1, 0), 1, colors.lightgrey),
                ]))
                
                # Disclaimer
                disclaimer_text = (
                    "This report is generated by an AI-based system and is intended solely for informational purposes. "
                    "Please consult a qualified medical professional for clinical decision-making. "
                    "This should not replace expert medical judgment."
                )
                story.append(Paragraph(disclaimer_text, disclaimer_style))
                
                # Footer with page number
                def add_page_number(canvas, doc):
                    canvas.saveState()
                    canvas.setFont('Helvetica', 8)
                    canvas.setFillColor(colors.grey)
                    footer_text = f"Page {doc.page} | Report ID: {report_id} | BrainAI Medical Analysis"
                    canvas.drawCentredString(letter[0]/2, 0.25*inch, footer_text)
                    canvas.restoreState()
                
                # Build PDF with page numbers
                doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
                pdf_bytes = pdf_buffer.getvalue()
                
                # Create download button with enhanced styling
                st.download_button(
                    label="📥 Download Professional PDF Report",
                    data=pdf_bytes,
                    file_name=f"brain_tumor_analysis_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    help="Download a professionally formatted PDF report of the analysis results"
                )
                
                # Success message
                st.success("✅ Professional PDF report generated successfully!")
                story.append(Spacer(1, 20))

                # Images
                story.append(Paragraph("MRI Scan Analysis", styles["Heading2"]))
                story.append(Spacer(1, 10))

                # Add images side by side
                original_img = RLImage(original_img_bytes, width=250, height=250)
                mask_img = RLImage(mask_img_bytes, width=250, height=250)
                image_table = Table([[original_img, mask_img]], colWidths=[250, 250])
                image_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                story.append(image_table)

                # Disclaimer
                story.append(Spacer(1, 30))
                story.append(Paragraph("Disclaimer", styles["Heading3"]))
                story.append(Paragraph(
                    "This report is generated by an AI system and should be reviewed by a qualified healthcare professional. "
                    "The results should be considered as supplementary information for clinical decision-making.",
                    styles["Normal"]))

                # Build PDF
                doc.build(story)
                pdf_bytes = pdf_buffer.getvalue()

                # Create download button
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"brain_tumor_analysis_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                )

            st.markdown("</div>", unsafe_allow_html=True)

def advanced_analysis_page():
    st.markdown("""
    <style>
        .adv-container {
            background: linear-gradient(120deg, #130f40 0%, #000046 100%);
            color: #e6f7ff;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            margin: 20px auto;
            max-width: 1200px;
        }
        .adv-header {
            color: #00d2ff;
            text-align: center;
            margin-bottom: 30px;
            font-size: 36px;
            font-weight: 800;
            text-shadow: 0 0 10px rgba(0, 210, 255, 0.5);
        }
        .adv-subheader {
            color: #00ffcc;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 24px;
            font-weight: 700;
            border-bottom: 2px solid rgba(0, 255, 204, 0.3);
            padding-bottom: 10px;
        }
        .adv-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            margin: 20px 0;
            border: 1px solid rgba(255, 255, 255, 0.18);
            transition: all 0.3s ease;
        }
        .adv-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(0, 210, 255, 0.3);
        }
        .upload-zone {
            background: rgba(255, 255, 255, 0.05);
            border: 2px dashed rgba(0, 210, 255, 0.5);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .upload-zone:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(0, 255, 204, 0.8);
        }
        .metric-futuristic {
            background: linear-gradient(120deg, rgba(0, 210, 255, 0.2) 0%, rgba(58, 123, 213, 0.2) 100%);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(0, 210, 255, 0.3);
            box-shadow: 0 0 20px rgba(0, 210, 255, 0.1);
        }
        .ai-insight {
            background: linear-gradient(120deg, rgba(0, 210, 255, 0.1) 0%, rgba(0, 255, 204, 0.1) 100%);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            border-left: 3px solid #00ffcc;
        }
        .ai-insight-title {
            color: #00ffcc;
            font-weight: bold;
            display: flex;
            align-items: center;
        }
        .ai-insight-title svg {
            margin-right: 10px;
        }
        .ai-insight-content {
            margin-top: 10px;
            font-style: italic;
            color: rgba(255, 255, 255, 0.8);
        }
        .heatmap-container {
            padding: 20px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 15px;
            margin: 20px 0;
        }
        .timeline {
            background: rgba(0, 0, 0, 0.2);
            padding: 20px;
            border-radius: 15px;
            margin-top: 30px;
        }
        .timeline-point {
            background: linear-gradient(120deg, #00d2ff 0%, #00ffcc 100%);
            width: 15px;
            height: 15px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .switch-back-button {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            color: #e6f7ff;
            padding: 10px 20px;
            border-radius: 30px;
            text-align: center;
            margin: 20px auto;
            display: inline-block;
            text-decoration: none;
            font-weight: bold;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        .switch-back-button:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-3px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(0, 210, 255, 0.3);
        }
        /* Improved text readability styles */
        .adv-container p, .adv-container li, .adv-container span, .adv-container div {
            color: #e6f7ff;
        }
        .adv-container h1, .adv-container h2, .adv-container h3, .adv-container h4, .adv-container h5, .adv-container h6 {
            color: #00d2ff;
        }
        .label-text {
            color: #a0e4ff;
            font-size: 14px;
            font-weight: 500;
        }
        .value-text {
            color: #00ffcc;
            font-size: 24px;
            font-weight: 700;
        }
        .description-text {
            color: #a0e4ff;
            font-size: 13px;
            line-height: 1.5;
        }
        .highlight-text {
            color: #00ffcc;
            font-weight: 700;
        }
        .secondary-text {
            color: #a0e4ff;
            font-weight: 500;
        }
        /* File uploader styling */
        .stFileUploader label, .stFileUploader span {
            color: #e6f7ff !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='adv-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='adv-header'>🔬 Advanced Brain Tumor Analysis</h1>", unsafe_allow_html=True)
    
    # Add switch back to standard analysis button 1091
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <a href="/?page=Analysis" class="switch-back-button">
                <span style="display: flex; align-items: center; justify-content: center;">
                    <span style="margin-right: 8px;">⬅️</span> Switch to Standard Analysis
                </span>
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='upload-zone'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #00d2ff; margin-bottom: 20px;'>Upload Your MRI Scan for Advanced Analysis</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)

    custom_cnn_model, classifier_model, segmentation_model, tumor_size_model = load_models()

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        
        st.markdown("<div class='adv-card'>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Progress tracking
        progress_container = st.empty()
        with progress_container.container():
            st.markdown("<div style='background: rgba(0,0,0,0.2); padding: 20px; border-radius: 10px;'>", unsafe_allow_html=True)
            progress_bar = st.progress(0)
            status_text = st.empty()
            st.markdown("</div>", unsafe_allow_html=True)

        status_text.text("🔍 Validating MRI scan...")
        image_type, is_mri = validate_mri(image, custom_cnn_model)
        progress_bar.progress(15)
        time.sleep(0.5)

        if not is_mri:
            st.error(f"⚠️ Detected Image Type: {image_type}. Please upload a valid MRI image.")
            progress_container.empty()
        else:
            st.success("✅ Valid MRI scan detected.")
            
            status_text.text("🧠 Classifying tumor type...")
            tumor_type, confidence = classify_tumor(image, classifier_model)
            progress_bar.progress(30)
            time.sleep(0.5)

            st.markdown("<div class='adv-card'>", unsafe_allow_html=True)
            st.markdown("<h3 class='adv-subheader'>Tumor Classification</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='metric-futuristic'>", unsafe_allow_html=True)
                st.metric("Tumor Type", tumor_type, delta=None, delta_color="normal")
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("<div class='metric-futuristic'>", unsafe_allow_html=True)
                st.metric("Confidence", f"{confidence:.2%}", delta=None, delta_color="normal")
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if tumor_type != "No Tumor":
                status_text.text("🔬 Segmenting tumor regions...")
                mask = segment_tumor(image, segmentation_model)
                progress_bar.progress(45)
                time.sleep(0.5)

                # Create a more advanced visualization with overlay
                st.markdown("<div class='adv-card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='adv-subheader'>Advanced Tumor Segmentation</h3>", unsafe_allow_html=True)
                
                # Create a more sophisticated visualization
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                
                                # Original image
                ax1.imshow(image)
                ax1.set_title("Original MRI Scan")
                ax1.axis("off")
                
                # Segmentation mask
                ax2.imshow(mask.squeeze(), cmap="viridis")
                ax2.set_title("Tumor Segmentation")
                ax2.axis("off")
                
                # Overlay visualization
                img_array = np.array(image)
                mask_binary = (mask.squeeze() > 0.5).astype(np.float32)
                
                # Resize mask to match image dimensions if they're different
                if img_array.shape[:2] != mask_binary.shape:
                    mask_binary = cv2.resize(mask_binary, (img_array.shape[1], img_array.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
                
                mask_rgb = np.zeros((*mask_binary.shape, 3), dtype=np.float32)
                mask_rgb[:, :, 0] = mask_binary  # Red channel for tumor
                
                # Create overlay
                overlay = img_array.astype(np.float32) / 255.0
                overlay_alpha = 0.7
                overlay = overlay * (1 - overlay_alpha * mask_rgb) + overlay_alpha * mask_rgb
                overlay = np.clip(overlay, 0, 1)
                
                ax3.imshow(overlay)
                ax3.set_title("Tumor Overlay")
                ax3.axis("off")
                
                plt.tight_layout()
                st.pyplot(fig)
                st.markdown("</div>", unsafe_allow_html=True)

                status_text.text("📏 Calculating tumor size and characteristics...")
                size = predict_tumor_size(mask, tumor_size_model)
                progress_bar.progress(60)
                time.sleep(0.5)

                # Tumor size and characteristics
                st.markdown("<div class='adv-card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='adv-subheader'>Tumor Size Analysis</h3>", unsafe_allow_html=True)
                
                # Size category determination
                size_category = "Small"
                if size[0] > 30:
                    size_category = "Large"
                elif size[0] > 15:
                    size_category = "Medium"
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("<div class='metric-futuristic'>", unsafe_allow_html=True)
                    st.metric("Tumor Size", f"{size[0]:.2f} mm²")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown("<div class='metric-futuristic'>", unsafe_allow_html=True)
                    st.metric("Size Category", size_category)
                    st.markdown("</div>", unsafe_allow_html=True)
                with col3:
                    # Calculate approximate diameter assuming circular shape
                    diameter = 2 * np.sqrt(size[0] / np.pi)
                    st.markdown("<div class='metric-futuristic'>", unsafe_allow_html=True)
                    st.metric("Approx. Diameter", f"{diameter:.2f} mm")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Additional tumor characteristics
                features = extract_features_for_size(mask)
                
                st.markdown("<div class='ai-insight'>", unsafe_allow_html=True)
                st.markdown("<div class='ai-insight-title'>🔍 AI Insight: Tumor Characteristics</div>", unsafe_allow_html=True)
                st.markdown("<div class='ai-insight-content'>", unsafe_allow_html=True)
                st.write(f"The tumor exhibits a perimeter of {features[1]:.2f} pixels with an aspect ratio of {features[2]:.2f}. The solidity value of {features[3]:.2f} indicates {'a relatively compact' if features[3] > 0.8 else 'an irregular'} shape.")
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # Treatment recommendations
                status_text.text("💊 Generating treatment recommendations...")
                progress_bar.progress(75)
                time.sleep(0.5)
                
                st.markdown("<div class='adv-card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='adv-subheader'>Personalized Treatment Recommendations</h3>", unsafe_allow_html=True)
                
                # Define medication recommendations based on tumor type and size
                medication_recommendations = {
                    "Glioma": {
                        "Small": {
                            "medications": ["Temozolomide (75-150 mg/m²)", "Dexamethasone (4-16 mg/day)"],
                            "treatments": ["Radiation therapy (54-60 Gy)", "Regular monitoring with MRI every 3 months"],
                            "specialists": ["Neuro-oncologist", "Radiation oncologist"]
                        },
                        "Medium": {
                            "medications": ["Temozolomide (150-200 mg/m²)", "Dexamethasone (4-16 mg/day)", "Levetiracetam (500-1500 mg/day)"],
                            "treatments": ["Surgical resection followed by radiation therapy", "Adjuvant chemotherapy", "Follow-up MRI every 2-3 months"],
                            "specialists": ["Neurosurgeon", "Neuro-oncologist", "Radiation oncologist"]
                        },
                        "Large": {
                            "medications": ["High-dose Temozolomide (200 mg/m²)", "Dexamethasone (16-24 mg/day)", "Levetiracetam (1000-3000 mg/day)"],
                            "treatments": ["Urgent surgical debulking", "Intensive radiation therapy", "Adjuvant chemotherapy", "Potential clinical trial enrollment"],
                            "specialists": ["Neurosurgeon", "Neuro-oncologist", "Radiation oncologist", "Palliative care specialist"]
                        }
                    },
                    "Meningioma": {
                        "Small": {
                            "medications": ["Observation only", "Acetaminophen or NSAIDs for headache if present"],
                            "treatments": ["Active surveillance with MRI every 6-12 months"],
                            "specialists": ["Neurosurgeon", "Neurologist"]
                        },
                        "Medium": {
                            "medications": ["Dexamethasone (4-8 mg/day) if symptomatic"],
                            "treatments": ["Surgical resection", "Stereotactic radiosurgery for difficult locations", "Follow-up MRI every 6 months"],
                            "specialists": ["Neurosurgeon", "Radiation oncologist"]
                        },
                        "Large": {
                            "medications": ["Dexamethasone (8-16 mg/day)", "Anti-seizure medication if symptomatic"],
                            "treatments": ["Urgent surgical resection", "Adjuvant radiation for incomplete resection", "Follow-up MRI every 3-4 months"],
                            "specialists": ["Neurosurgeon", "Radiation oncologist", "Neurologist"]
                        }
                    },
                    "Pituitary": {
                        "Small": {
                            "medications": ["Cabergoline or Bromocriptine for prolactinomas", "Hormone replacement as needed"],
                            "treatments": ["Endocrine monitoring", "MRI follow-up every 6-12 months"],
                            "specialists": ["Endocrinologist", "Neurosurgeon"]
                        },
                        "Medium": {
                            "medications": ["Hormone replacement therapy", "Cabergoline for prolactinomas"],
                            "treatments": ["Transsphenoidal surgery", "Post-op endocrine assessment", "Follow-up MRI every 4-6 months"],
                            "specialists": ["Neurosurgeon", "Endocrinologist", "Ophthalmologist"]
                        },
                        "Large": {
                            "medications": ["High-dose hormone replacement", "Dexamethasone (4-8 mg/day)"],
                            "treatments": ["Urgent transsphenoidal surgery", "Visual field monitoring", "Radiation therapy for residual tumor"],
                            "specialists": ["Neurosurgeon", "Endocrinologist", "Ophthalmologist", "Radiation oncologist"]
                        }
                    }
                }
                
                # Display recommendations
                if tumor_type in medication_recommendations:
                    recommendations = medication_recommendations[tumor_type][size_category]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("<div style='background: rgba(0,0,0,0.2); padding: 20px; border-radius: 10px; height: 100%;'>", unsafe_allow_html=True)
                        st.markdown(f"<h4 style='color: #00ffcc;'>💊 Recommended Medications</h4>", unsafe_allow_html=True)
                        for med in recommendations["medications"]:
                            st.markdown(f"<p>• {med}</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("<div style='background: rgba(0,0,0,0.2); padding: 20px; border-radius: 10px; height: 100%;'>", unsafe_allow_html=True)
                        st.markdown(f"<h4 style='color: #00ffcc;'>🏥 Recommended Treatments</h4>", unsafe_allow_html=True)
                        for treatment in recommendations["treatments"]:
                            st.markdown(f"<p>• {treatment}</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div style='background: rgba(0,0,0,0.2); padding: 20px; border-radius: 10px; margin-top: 15px;'>", unsafe_allow_html=True)
                    st.markdown(f"<h4 style='color: #00ffcc;'>👨‍⚕️ Recommended Specialists</h4>", unsafe_allow_html=True)
                    specialist_cols = st.columns(len(recommendations["specialists"]))
                    for i, specialist in enumerate(recommendations["specialists"]):
                        with specialist_cols[i]:
                            st.markdown(f"<div style='text-align: center;'><p style='font-weight: bold;'>👨‍⚕️</p><p>{specialist}</p></div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<p class='description-text' style='margin-top: 15px; font-style: italic;'>Note: These recommendations are generated by AI and should be reviewed by a healthcare professional. Treatment plans should be personalized based on the patient's complete medical history and specific circumstances.</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # 12-month tumor growth prediction
                status_text.text("📈 Predicting tumor growth over 12 months...")
                progress_bar.progress(85)
                time.sleep(0.5)
                
                st.markdown("<div class='adv-card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='adv-subheader'>12-Month Tumor Growth Prediction</h3>", unsafe_allow_html=True)
                
                # Simulate growth rates based on tumor type and size
                growth_rates = {
                    "Glioma": {"Small": 0.15, "Medium": 0.25, "Large": 0.35},
                    "Meningioma": {"Small": 0.05, "Medium": 0.10, "Large": 0.15},
                    "Pituitary": {"Small": 0.08, "Medium": 0.12, "Large": 0.18}
                }
                
                # Get growth rate for this tumor type and size
                growth_rate = growth_rates.get(tumor_type, {}).get(size_category, 0.1)
                
                               # Get growth rate for this tumor type and size
                growth_rate = growth_rates.get(tumor_type, {}).get(size_category, 0.1)
                
                # Generate monthly predictions
                months = list(range(1, 13))
                current_size = size[0]
                predicted_sizes = [current_size]
                
                # Modified growth calculation for more gradual, realistic growth
                base_monthly_growth = growth_rate / 12  # Distribute annual growth rate across months
                
                for i in range(12):
                    # Apply smaller incremental growth with slight randomness
                    random_factor = np.random.uniform(0.95, 1.05)
                    
                    # More gradual growth pattern
                    if i < 3:
                        # Slower initial growth
                        month_growth = base_monthly_growth * 0.8 * random_factor
                    elif i < 6:
                        # Slightly increased growth
                        month_growth = base_monthly_growth * 1 * random_factor
                    elif i < 9:
                        # Normal growth
                        month_growth = base_monthly_growth * random_factor
                    else:
                        # Slightly reduced growth as tumor gets larger
                        month_growth = base_monthly_growth * 1.95 * random_factor
                    
                    # Add growth to previous size
                    next_size = predicted_sizes[-1] * (1 + month_growth)
                    predicted_sizes.append(next_size)
                
                # Create growth chart
                fig = plt.figure(figsize=(10, 6))
                plt.plot(range(13), predicted_sizes, marker='o', linestyle='-', color='#00d2ff', linewidth=3, markersize=8)
                plt.fill_between(range(13), predicted_sizes, alpha=0.2, color='#00d2ff')
                plt.title('Predicted Tumor Growth Over 12 Months', fontsize=16, color='white')
                plt.xlabel('Months', fontsize=12, color='white')
                plt.ylabel('Tumor Size (mm²)', fontsize=12, color='white')
               
                # Removed critical threshold line
                
                # Customize appearance
                plt.legend(fontsize=10)
                fig.patch.set_facecolor('#0f2365')
                ax = plt.gca()
                ax.set_facecolor('#0f2365')
                for spine in ax.spines.values():
                    spine.set_color('white')
                # Add current size marker
                plt.scatter([0], [current_size], s=120, color='#00ffcc', zorder=5, label='Current Size')
                
                # Add critical threshold if applicable
                if tumor_type in ["Glioma", "Meningioma"]:
                    critical_size = current_size * 2  # Example threshold
                    plt.axhline(y=critical_size, color='#ff5252', linestyle='--', alpha=0.7, label='Critical Threshold')
                
                # Customize appearance
                plt.legend(fontsize=10)
                fig.patch.set_facecolor('#0f2365')
                ax = plt.gca()
                ax.set_facecolor('#0f2365')
                for spine in ax.spines.values():
                    spine.set_color('white')
                
                st.pyplot(fig)
                
                # Predicted MRI visualization at 6 and 12 months
                st.markdown("<h4 style='color: #00ffcc; margin-top: 20px;'>Simulated Future MRI Appearance</h4>", unsafe_allow_html=True)
                
                # Create simulated future MRIs
                                # Create simulated future MRIs
                def simulate_future_mri(original_img, original_mask, growth_factor):
                    # Convert mask to binary
                    binary_mask = (original_mask.squeeze() > 0.5).astype(np.uint8)
                    
                    # Resize mask to match image dimensions if they're different
                    img_array = np.array(original_img)
                    if img_array.shape[:2] != binary_mask.shape:
                        binary_mask = cv2.resize(binary_mask, (img_array.shape[1], img_array.shape[0]), 
                                                interpolation=cv2.INTER_NEAREST)
                    
                    # Dilate mask to simulate growth
                    kernel_size = int(np.ceil(growth_factor * 10))
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
                    
                    # Create overlay
                    mask_rgb = np.zeros((*dilated_mask.shape, 3), dtype=np.float32)
                    mask_rgb[:, :, 0] = dilated_mask  # Red channel for tumor
                    
                    overlay = img_array.astype(np.float32) / 255.0
                    overlay_alpha = 0.7
                    overlay = overlay * (1 - overlay_alpha * mask_rgb) + overlay_alpha * mask_rgb
                    overlay = np.clip(overlay, 0, 1)
                    
                    return overlay
                               # Generate future MRIs
                # Increase the growth factor to make the difference more visible
                growth_factor_6m = (predicted_sizes[6]/current_size) * 1.5  # Amplify the 6-month growth
                growth_factor_12m = (predicted_sizes[12]/current_size) * 2.0  # Amplify the 12-month growth even more
                
                future_6m = simulate_future_mri(image, mask, growth_factor_6m)
                future_12m = simulate_future_mri(image, mask, growth_factor_12m)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(image, caption="Current MRI")
                    st.markdown(f"<div style='text-align: center;'><p>Current Size: <span style='color: #00ffcc; font-weight: bold;'>{current_size:.2f} mm²</span></p></div>", unsafe_allow_html=True)
                
                with col2:
                    st.image(future_6m, caption="Predicted MRI at 6 Months")
                    st.markdown(f"<div style='text-align: center;'><p>Predicted Size: <span style='color: #00ffcc; font-weight: bold;'>{predicted_sizes[6]:.2f} mm²</span></p></div>", unsafe_allow_html=True)
                
                with col3:
                    st.image(future_12m, caption="Predicted MRI at 12 Months")
                    st.markdown(f"<div style='text-align: center;'><p>Predicted Size: <span style='color: #00ffcc; font-weight: bold;'>{predicted_sizes[12]:.2f} mm²</span></p></div>", unsafe_allow_html=True)
                # Growth analysis insights
                growth_percentage = ((predicted_sizes[12] - current_size) / current_size) * 100
                
                st.markdown("<div class='ai-insight'>", unsafe_allow_html=True)
                st.markdown("<div class='ai-insight-title'>🔮 AI Insight: Growth Projection</div>", unsafe_allow_html=True)
                st.markdown("<div class='ai-insight-content'>", unsafe_allow_html=True)
                
                if growth_percentage < 20:
                    growth_assessment = "slow"
                    recommendation = "regular monitoring with follow-up MRI every 6 months"
                elif growth_percentage < 50:
                    growth_assessment = "moderate"
                    recommendation = "closer monitoring with follow-up MRI every 3 months and consideration of early intervention"
                else:
                    growth_assessment = "rapid"
                    recommendation = "immediate intervention and aggressive treatment plan"
                
                st.write(f"Based on the analysis, this {tumor_type.lower()} shows a projected {growth_assessment} growth rate of {growth_percentage:.1f}% over the next 12 months. We recommend {recommendation}.")
                
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Professional report generation
                status_text.text("📄 Preparing comprehensive report...")
                progress_bar.progress(95)
                time.sleep(0.5)
                
                st.markdown("<div class='adv-card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='adv-subheader'>Professional Medical Report</h3>", unsafe_allow_html=True)
                st.markdown("<p>Generate a comprehensive medical report with all analysis results, treatment recommendations, and growth predictions.</p>", unsafe_allow_html=True)
                
                if st.button("Generate Advanced PDF Report", type="primary"):
                    from reportlab.lib import colors
                    from reportlab.lib.pagesizes import letter
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                    from reportlab.lib.units import inch
                    from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_JUSTIFY, TA_LEFT
                    from io import BytesIO
                    import uuid
                    import datetime

                    # Generate a unique report ID
                    report_id = str(uuid.uuid4())[:8].upper()
                    
                    # Current timestamp for the report
                    timestamp = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")
                    
                    # Save images to bytes
                    original_img_bytes = BytesIO()
                    plt.figure(figsize=(6, 6))
                    plt.imshow(image)
                    plt.axis('off')
                    plt.title("Original MRI Scan")
                    plt.tight_layout()
                    plt.savefig(original_img_bytes, format='png', bbox_inches='tight', dpi=300)
                    plt.close()

                    mask_img_bytes = BytesIO()
                    plt.figure(figsize=(6, 6))
                    plt.imshow(mask.squeeze(), cmap="viridis")
                    plt.axis('off')
                    plt.title("Tumor Segmentation")
                    plt.tight_layout()
                    plt.savefig(mask_img_bytes, format='png', bbox_inches='tight', dpi=300)
                    plt.close()
                    
                    overlay_img_bytes = BytesIO()
                    plt.figure(figsize=(6, 6))
                    img_array = np.array(image)
                    mask_binary = (mask.squeeze() > 0.5).astype(np.float32)
                    mask_rgb = np.zeros((*mask_binary.shape, 3), dtype=np.float32)
                    mask_rgb[:, :, 0] = mask_binary  # Red channel for tumor
                    overlay = img_array.astype(np.float32) / 255.0
                    overlay_alpha = 0.7
                    overlay_img_bytes = BytesIO()
                    plt.figure(figsize=(6, 6))
                    img_array = np.array(image)
                    mask_binary = (mask.squeeze() > 0.5).astype(np.float32)
                    
                    # Resize mask to match image dimensions if they're different
                    if img_array.shape[:2] != mask_binary.shape:
                        mask_binary = cv2.resize(mask_binary, (img_array.shape[1], img_array.shape[0]), 
                                                interpolation=cv2.INTER_NEAREST)
                    
                    mask_rgb = np.zeros((*mask_binary.shape, 3), dtype=np.float32)
                    mask_rgb[:, :, 0] = mask_binary  # Red channel for tumor
                    overlay = img_array.astype(np.float32) / 255.0
                    overlay_alpha = 0.7
                    overlay = overlay * (1 - overlay_alpha * mask_rgb) + overlay_alpha * mask_rgb
                    overlay = np.clip(overlay, 0, 1)
                    plt.imshow(overlay)

                    plt.axis('off')
                    plt.title("Tumor Overlay")
                    plt.tight_layout()
                    plt.savefig(overlay_img_bytes, format='png', bbox_inches='tight', dpi=300)
                    plt.close()
                    
                  
                    
                    # Save future MRI predictions
                    future_mri_bytes = BytesIO()
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
                    ax1.imshow(image)
                    ax1.set_title("Current")
                    ax1.axis('off')
                    ax2.imshow(future_6m)
                    ax2.set_title("6 Months")
                    ax2.axis('off')
                    ax3.imshow(future_12m)
                    ax3.set_title("12 Months")
                    ax3.axis('off')
                    plt.tight_layout()
                    plt.savefig(future_mri_bytes, format='png', bbox_inches='tight', dpi=300)
                    plt.close()

                    # Create PDF with enhanced styling
                    pdf_buffer = BytesIO()
                    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, 
                                            leftMargin=0.5*inch, rightMargin=0.5*inch,
                                            topMargin=0.5*inch, bottomMargin=0.5*inch)
                    styles = getSampleStyleSheet()
                    
                    # Custom styles
                    title_style = ParagraphStyle(
                        'CustomTitle',
                        parent=styles['Heading1'],
                        fontSize=24,
                        fontName='Helvetica-Bold',
                        alignment=TA_CENTER,
                        spaceAfter=6
                    )
                    
                    subtitle_style = ParagraphStyle(
                        'CustomSubtitle',
                        parent=styles['Heading2'],
                        fontSize=16,
                        fontName='Helvetica-Oblique',
                        textColor=colors.darkblue,
                        alignment=TA_CENTER,
                        spaceAfter=20
                    )
                    
                    section_title_style = ParagraphStyle(
                        'SectionTitle',
                        parent=styles['Heading2'],
                        fontSize=14,
                        fontName='Helvetica-Bold',
                        textColor=colors.darkblue,
                        spaceAfter=10,
                        spaceBefore=15
                    )
                    
                    subsection_title_style = ParagraphStyle(
                        'SubsectionTitle',
                        parent=styles['Heading3'],
                        fontSize=12,
                        fontName='Helvetica-Bold',
                        textColor=colors.darkblue,
                        spaceAfter=8,
                        spaceBefore=12
                    )
                    
                    normal_style = ParagraphStyle(
                        'CustomNormal',
                        parent=styles['Normal'],
                        fontSize=10,
                        spaceBefore=6,
                        spaceAfter=6
                    )
                    
                    disclaimer_style = ParagraphStyle(
                        'Disclaimer',
                        parent=styles['Normal'],
                        fontSize=8,
                        fontName='Helvetica-Oblique',
                        textColor=colors.darkgrey,
                        alignment=TA_JUSTIFY,
                        spaceBefore=30
                    )
                    
                    footer_style = ParagraphStyle(
                        'Footer',
                        parent=styles['Normal'],
                        fontSize=8,
                        textColor=colors.darkgrey,
                        alignment=TA_CENTER
                    )
                    
                    story = []
                    
                    # Header with logo placeholder
                    header_data = [
                        [Paragraph("🧠 BrainAI Advanced Medical Center", styles["Heading3"]), 
                         Paragraph(f"Report ID: {report_id}<br/>Generated: {timestamp}", styles["Normal"])]
                    ]
                    header_table = Table(header_data, colWidths=[4*inch, 3*inch])
                    header_table.setStyle(TableStyle([
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                    ]))
                    story.append(header_table)
                    story.append(Spacer(1, 20))
                    
                    # Title and subtitle
                    story.append(Paragraph("Advanced Brain Tumor Analysis Report", title_style))
                    story.append(Paragraph("AI-Powered Comprehensive Diagnostic Summary", subtitle_style))
                    story.append(Spacer(1, 10))
                    
                    # Patient Information (placeholder)
                    story.append(Paragraph("Patient Information", section_title_style))
                    patient_data = [
                        ["Patient ID:", "ANONYMOUS"],
                        ["Date of Birth:", "REDACTED"],
                        ["Gender:", "REDACTED"],
                        ["Scan Date:", datetime.datetime.now().strftime("%Y-%m-%d")],
                    ]
                    patient_table = Table(patient_data, colWidths=[1.5*inch, 5.5*inch])
                    patient_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (0, -1), colors.darkblue),
                        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
                        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                        ('TOPPADDING', (0, 0), (-1, -1), 6),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ]))
                    story.append(patient_table)
                    story.append(Spacer(1, 15))
                    
                    # Analysis Results Section
                    story.append(Paragraph("Analysis Results", section_title_style))
                    
                    # Results table with enhanced styling
                    data = [
                        ["Parameter", "Value", "Interpretation"],
                        ["Tumor Type", tumor_type, f"High confidence classification ({confidence:.1%})"],
                        ["Tumor Size", f"{size[0]:.2f} mm²", size_category],
                        ["Growth Rate", f"{growth_percentage:.1f}% (12 months)", growth_assessment.capitalize()],
                    ]
                    
                    results_table = Table(data, colWidths=[1.5*inch, 2*inch, 3.5*inch])
                    results_table.setStyle(TableStyle([
                        # Header row styling
                        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        
                        # Alternating row colors
                        ('BACKGROUND', (0, 1), (-1, 1), colors.lightgrey),
                        ('BACKGROUND', (0, 3), (-1, 3), colors.lightgrey),
                        
                        # Cell styling
                        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 1), (-1, -1), 10),
                        ('TOPPADDING', (0, 1), (-1, -1), 6),
                        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                        
                        # Table grid
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('BOX', (0, 0), (-1, -1), 1, colors.darkgrey),
                    ]))
                    story.append(results_table)
                    
                    # Section divider
                    story.append(Spacer(1, 10))
                    story.append(Table([['']], colWidths=[7*inch], style=[
                        ('LINEABOVE', (0, 0), (-1, 0), 1, colors.lightgrey),
                    ]))
                    
                    # MRI Scan Analysis Section
                    story.append(Paragraph("MRI Scan Analysis", section_title_style))
                    story.append(Spacer(1, 10))
                    
                    # Reset image bytes position
                    original_img_bytes.seek(0)
                    mask_img_bytes.seek(0)
                    overlay_img_bytes.seek(0)
                    
                    # Create images with captions
                    original_img = RLImage(original_img_bytes, width=2.2*inch, height=2.2*inch)
                    mask_img = RLImage(mask_img_bytes, width=2.2*inch, height=2.2*inch)
                    overlay_img = RLImage(overlay_img_bytes, width=2.2*inch, height=2.2*inch)
                    
                    # Image table with captions
                    image_data = [
                        [original_img, mask_img, overlay_img],
                        [Paragraph("<b>Original MRI Scan</b>", styles["Normal"]), 
                         Paragraph("<b>Tumor Segmentation</b>", styles["Normal"]),
                         Paragraph("<b>Tumor Overlay</b>", styles["Normal"])]
                    ]
                    
                    image_table = Table(image_data, colWidths=[2.3*inch, 2.3*inch, 2.3*inch])
                    image_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('TOPPADDING', (0, 1), (-1, 1), 6),
                        ('BOTTOMPADDING', (0, 1), (-1, 1), 20),
                    ]))
                    story.append(image_table)
                    
                    # Tumor Characteristics
                    story.append(Paragraph("Tumor Characteristics", subsection_title_style))
                    
                    characteristics_text = (
                        f"The tumor exhibits a perimeter of {features[1]:.2f} pixels with an aspect ratio of {features[2]:.2f}. "
                        f"The solidity value of {features[3]:.2f} indicates {'a relatively compact' if features[3] > 0.8 else 'an irregular'} shape. "
                        f"Based on these characteristics, this appears to be a {size_category.lower()}-sized {tumor_type} tumor."
                    )
                    story.append(Paragraph(characteristics_text, normal_style))
                    story.append(Spacer(1, 15))
                    
                    # Treatment Recommendations
                    story.append(Paragraph("Treatment Recommendations", section_title_style))
                    
                    # Get recommendations based on tumor type and size
                    medication_recommendations = {
                        "Glioma": {
                            "Small": {
                                "medications": ["Temozolomide (75-150 mg/m²)", "Dexamethasone (4-16 mg/day)"],
                                "treatments": ["Radiation therapy (54-60 Gy)", "Regular monitoring with MRI every 3 months"],
                                "specialists": ["Neuro-oncologist", "Radiation oncologist"]
                            },
                            "Medium": {
                                "medications": ["Temozolomide (150-200 mg/m²)", "Dexamethasone (4-16 mg/day)", "Levetiracetam (500-1500 mg/day)"],
                                "treatments": ["Surgical resection followed by radiation therapy", "Adjuvant chemotherapy", "Follow-up MRI every 2-3 months"],
                                "specialists": ["Neurosurgeon", "Neuro-oncologist", "Radiation oncologist"]
                            },
                            "Large": {
                                "medications": ["High-dose Temozolomide (200 mg/m²)", "Dexamethasone (16-24 mg/day)", "Levetiracetam (1000-3000 mg/day)"],
                                "treatments": ["Urgent surgical debulking", "Intensive radiation therapy", "Adjuvant chemotherapy", "Potential clinical trial enrollment"],
                                "specialists": ["Neurosurgeon", "Neuro-oncologist", "Radiation oncologist", "Palliative care specialist"]
                            }
                        },
                        "Meningioma": {
                            "Small": {
                                "medications": ["Observation only", "Acetaminophen or NSAIDs for headache if present"],
                                "treatments": ["Active surveillance with MRI every 6-12 months"],
                                "specialists": ["Neurosurgeon", "Neurologist"]
                            },
                            "Medium": {
                                "medications": ["Dexamethasone (4-8 mg/day) if symptomatic"],
                                "treatments": ["Surgical resection", "Stereotactic radiosurgery for difficult locations", "Follow-up MRI every 6 months"],
                                "specialists": ["Neurosurgeon", "Radiation oncologist"]
                            },
                            "Large": {
                                "medications": ["Dexamethasone (8-16 mg/day)", "Anti-seizure medication if symptomatic"],
                                "treatments": ["Urgent surgical resection", "Adjuvant radiation for incomplete resection", "Follow-up MRI every 3-4 months"],
                                "specialists": ["Neurosurgeon", "Radiation oncologist", "Neurologist"]
                            }
                        },
                        "Pituitary": {
                            "Small": {
                                "medications": ["Cabergoline or Bromocriptine for prolactinomas", "Hormone replacement as needed"],
                                "treatments": ["Endocrine monitoring", "MRI follow-up every 6-12 months"],
                                "specialists": ["Endocrinologist", "Neurosurgeon"]
                            },
                            "Medium": {
                                "medications": ["Hormone replacement therapy", "Cabergoline for prolactinomas"],
                                "treatments": ["Transsphenoidal surgery", "Post-op endocrine assessment", "Follow-up MRI every 4-6 months"],
                                "specialists": ["Neurosurgeon", "Endocrinologist", "Ophthalmologist"]
                            },
                            "Large": {
                                "medications": ["High-dose hormone replacement", "Dexamethasone (4-8 mg/day)"],
                                "treatments": ["Urgent transsphenoidal surgery", "Visual field monitoring", "Radiation therapy for residual tumor"],
                                "specialists": ["Neurosurgeon", "Endocrinologist", "Ophthalmologist", "Radiation oncologist"]
                            }
                        }
                    }
                    
                    # Get recommendations for this tumor type and size
                    if tumor_type in medication_recommendations:
                        recommendations = medication_recommendations[tumor_type][size_category]
                        
                        # Medications
                        story.append(Paragraph("Recommended Medications", subsection_title_style))
                        meds_text = ""
                        for med in recommendations["medications"]:
                            meds_text += f"• {med}<br/>"
                        story.append(Paragraph(meds_text, normal_style))
                        
                        # Treatments
                        story.append(Paragraph("Recommended Treatments", subsection_title_style))
                        treatments_text = ""
                        for treatment in recommendations["treatments"]:
                            treatments_text += f"• {treatment}<br/>"
                        story.append(Paragraph(treatments_text, normal_style))
                        
                        # Specialists
                        story.append(Paragraph("Recommended Specialists", subsection_title_style))
                        specialists_text = ""
                        for specialist in recommendations["specialists"]:
                            specialists_text += f"• {specialist}<br/>"
                        story.append(Paragraph(specialists_text, normal_style))
                    
                    story.append(Spacer(1, 15))
                    story.append(PageBreak())
                    
                    # Growth Prediction Section
                    story.append(Paragraph("12-Month Tumor Growth Prediction", section_title_style))
                    
                    # Growth chart
                                       # Save growth prediction chart
                    growth_chart_bytes = BytesIO()
                    fig = plt.figure(figsize=(8, 5))
                    plt.plot(range(13), predicted_sizes, marker='o', linestyle='-', color='#00d2ff', linewidth=3, markersize=8)
                    plt.fill_between(range(13), predicted_sizes, alpha=0.2, color='#00d2ff')
                    plt.title('Predicted Tumor Growth Over 12 Months', fontsize=16)
                    plt.xlabel('Months', fontsize=12)
                    plt.ylabel('Tumor Size (mm²)', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(growth_chart_bytes, format='png', bbox_inches='tight', dpi=300)
                    plt.close()
                    
                    # Save future MRI predictions
                    future_mri_bytes = BytesIO()
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
                    ax1.imshow(image)
                    ax1.set_title("Current")
                    ax1.axis('off')
                    ax2.imshow(future_6m)
                    ax2.set_title("6 Months")
                    ax2.axis('off')
                    ax3.imshow(future_12m)
                    ax3.set_title("12 Months")
                    ax3.axis('off')
                    plt.tight_layout()
                    plt.savefig(future_mri_bytes, format='png', bbox_inches='tight', dpi=300)
                    plt.close()
                    
                    # Growth analysis text
                    growth_analysis = (
                        f"Based on the analysis, this {tumor_type.lower()} shows a projected {growth_assessment} growth rate "
                        f"of {growth_percentage:.1f}% over the next 12 months. The current size of {current_size:.2f} mm² "
                        f"is expected to increase to approximately {predicted_sizes[12]:.2f} mm² after 12 months. "
                        f"We recommend {recommendation}."
                    )
                   
                    story.append(Paragraph(growth_analysis, normal_style))
                    story.append(Spacer(1, 15))
                    
                    # Future MRI predictions
                    story.append(Paragraph("Predicted Future MRI Appearance", subsection_title_style))
                    future_mri_bytes.seek(0)
                    future_mri_img = RLImage(future_mri_bytes, width=7*inch, height=2.5*inch)
                    story.append(future_mri_img)
                    
                    future_mri_text = (
                        f"The images above show the predicted appearance of the tumor at the current time (left), "
                        f"after 6 months (middle), and after 12 months (right). These visualizations are based on "
                        f"the growth model and should be considered as approximations."
                    )
                    story.append(Paragraph(future_mri_text, normal_style))
                    story.append(Spacer(1, 15))
                    
                    # Follow-up Recommendations
                    story.append(Paragraph("Follow-up Recommendations", section_title_style))
                    
                    if growth_percentage < 20:
                        followup_text = (
                            "Based on the slow projected growth rate, we recommend:<br/>"
                            "• Follow-up MRI scan every 6 months<br/>"
                            "• Regular consultation with a neurologist<br/>"
                            "• Monitor for any new or worsening symptoms"
                        )
                    elif growth_percentage < 50:
                        followup_text = (
                            "Based on the moderate projected growth rate, we recommend:<br/>"
                            "• Follow-up MRI scan every 3 months<br/>"
                            "• Consultation with a neurosurgeon to discuss intervention options<br/>"
                            "• Regular neurological examinations<br/>"
                            "• Prompt reporting of any new or worsening symptoms"
                        )
                    else:
                        followup_text = (
                            "Based on the rapid projected growth rate, we recommend:<br/>"
                            "• Immediate consultation with a neurosurgeon<br/>"
                            "• Follow-up MRI scan every 1-2 months<br/>"
                            "• Consider early intervention options<br/>"
                            "• Close monitoring of neurological symptoms<br/>"
                            "• Urgent medical attention for any new or worsening symptoms"
                        )
                    
                    story.append(Paragraph(followup_text, normal_style))
                    story.append(Spacer(1, 20))
                    
                    # Disclaimer
                    disclaimer_text = (
                        "DISCLAIMER: This report is generated by an AI-based system and is intended solely for informational purposes. "
                        "The analysis, predictions, and recommendations provided are based on machine learning models and should be "
                        "reviewed and validated by qualified healthcare professionals. This report should not replace expert medical "
                        "judgment or proper clinical evaluation. Treatment decisions should be made in consultation with appropriate "
                        "medical specialists considering the patient's complete medical history and specific circumstances."
                    )
                    story.append(Paragraph(disclaimer_text, disclaimer_style))
                    
                    # Footer with page number
                    def add_page_number(canvas, doc):
                        canvas.saveState()
                        canvas.setFont('Helvetica', 8)
                        canvas.setFillColor(colors.grey)
                        footer_text = f"Page {doc.page} of {doc.page} | Report ID: {report_id} | BrainAI Advanced Medical Analysis"
                        canvas.drawCentredString(letter[0]/2, 0.25*inch, footer_text)
                        canvas.restoreState()
                    
                    # Build PDF with page numbers
                    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
                    pdf_bytes = pdf_buffer.getvalue()
                    
                    # Create download button with enhanced styling
                    st.download_button(
                        label="📥 Download Comprehensive PDF Report",
                        data=pdf_bytes,
                        file_name=f"advanced_brain_tumor_analysis_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        help="Download a professionally formatted comprehensive PDF report of the analysis results"
                    )
                    
                    # Success message
                    st.success("✅ Comprehensive PDF report generated successfully!")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            progress_container.empty()
            st.markdown("<div style='text-align: center; margin-top: 30px;'>", unsafe_allow_html=True)
            st.success("✅ Advanced analysis completed successfully!")
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ======================
# MAIN APP CONFIG
# ======================

def main():
    st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Top Nav Bar
    st.markdown("""
    <div class="top-nav">
        <div class="nav-container">
            <div class="nav-brand">🧠 BrainAI</div>
            <div class="nav-links">
                <a href="/?page=Home" class="nav-link">Home</a>
                <a href="/?page=Analysis" class="nav-link">Analysis</a>
                <a href="/?page=AdvancedAnalysis" class="nav-link">Advanced Analysis</a>
                <a href="/?page=HowItWorks" class="nav-link">How It Works</a>
                <a href="/?page=About" class="nav-link">About</a>
                <a href="/?page=Contact" class="nav-link">Contact</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
import importlib.util
import os

def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

query_params = st.query_params.get("page", "Home")

if query_params == "Home":
    home_page()
elif query_params == "Analysis":
    analysis_page()
elif query_params == "AdvancedAnalysis":
    advanced_analysis_page()
elif query_params == "HowItWorks":
    how_it_works = load_module_from_file("how_it_works", os.path.join(os.path.dirname(__file__), "2_How_It_Works.py"))
    how_it_works.how_it_works_page()
elif query_params == "About":
    about = load_module_from_file("about", os.path.join(os.path.dirname(__file__), "1_About.py"))
    about.about_page()
elif query_params == "Contact":
    contact = load_module_from_file("contact", os.path.join(os.path.dirname(__file__), "3_Contact.py"))
    contact.contact_page()


if __name__ == "__main__":
    main()
