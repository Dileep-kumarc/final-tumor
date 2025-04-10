import streamlit as st

# Page setup
st.set_page_config(
    page_title="About | BrainAI",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for style
st.markdown("""
<style>
.about-container {
    background-color: #f9f9f9;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 0 12px rgba(0,0,0,0.07);
    font-family: 'Segoe UI', sans-serif;
}
.about-container h2, h3 {
    color: #3A3A3A;
}
.about-container ul {
    line-height: 1.8;
}
.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin-top: 2rem;
    color: #30475e;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("📖 About BrainAI")

# About content
st.markdown("""
<div class="about-container">

<h2>🧠 Brain Tumor Detection System</h2>

<p>
BrainAI is a deep learning-powered application designed to assist radiologists and healthcare providers in analyzing brain MRI scans. It performs automatic detection, classification, segmentation, and size estimation of tumors, delivering fast and interpretable results.
</p>

<h3 class="section-title">🚀 Core Capabilities</h3>
<ul>
    <li><b>MRI Scan Validation</b> — Ensures uploaded images are genuine T1-weighted MRI scans.</li>
    <li><b>Tumor Classification</b> — Identifies tumor type: Glioma, Meningioma, or Pituitary.</li>
    <li><b>Segmentation</b> — Generates a mask overlay highlighting tumor boundaries.</li>
    <li><b>Size Estimation</b> — Estimates tumor area in cm² from the segmentation mask.</li>
</ul>

<h3 class="section-title">🧪 Technology Stack</h3>
<ul>
    <li><b>Deep Learning</b>: Transfer Learning with CNNs (PyTorch & TensorFlow)</li>
    <li><b>Web App</b>: Streamlit for fast interactive UI</li>
    <li><b>Visualization</b>: Plotly & Matplotlib for charts and overlays</li>
</ul>

<h3 class="section-title">📊 Model Accuracy</h3>
<ul>
    <li><b>Validation Model</b>: 98.5% accuracy</li>
    <li><b>Classification Model</b>: 94.2% accuracy across 4 tumor types</li>
    <li><b>Segmentation Model</b>: 89% Dice coefficient score</li>
</ul>

<h3 class="section-title">⚠️ Medical Disclaimer</h3>
<p>
This tool is developed for educational and research purposes and is intended to support—not replace—professional medical judgment. Always consult a licensed medical professional for diagnosis and treatment decisions.
</p>

</div>
""", unsafe_allow_html=True)
