import streamlit as st
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(
    page_title="How It Works | Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .feature-card {
        background: white;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 25px;
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .model-diagram {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin: 20px 0;
    }
    .step-container {
        display: flex;
        align-items: center;
        margin: 30px 0;
        gap: 30px;
    }
    .step-number {
        font-size: 24px;
        font-weight: bold;
        background: #3b82f6;
        color: white;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .tab-content {
        padding: 20px 0;
    }
    @media (max-width: 768px) {
        .step-container {
            flex-direction: column;
        }
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.title("How Our AI Detects Brain Tumors")
st.markdown("""
Explore the sophisticated technology behind our brain tumor detection system. 
This page reveals how artificial intelligence analyzes MRI scans to identify 
and characterize brain tumors with remarkable accuracy.
""")

# Model Architecture Section
st.header("🧠 Our AI Architecture", divider="blue")

tab1, tab2, tab3 = st.tabs(["MRI Validation", "Tumor Classification", "Segmentation"])

with tab1:
    st.markdown("""
    <div class="feature-card">
        <h3>MRI Validation Model</h3>
        <p>Ensures uploaded images are valid MRI scans before analysis</p>
        <div class="model-diagram">
            <img src="https://sdmntprwestus.oaiusercontent.com/files/00000000-0c18-5230-a683-3f8626ad601a/raw?se=2025-04-09T12%3A51%3A02Z&sp=r&sv=2024-08-04&sr=b&scid=f068c6a2-c221-57e3-84b9-bd0b4d6e839f&skoid=e872f19f-7b7f-4feb-9998-20052dec61d6&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-04-09T09%3A41%3A16Z&ske=2025-04-10T09%3A41%3A16Z&sks=b&skv=2024-08-04&sig=29tMGKTFf2g8LyDZrM8ucNGfrKz24cnrEpy%2BD0q0l74%3D" 
                 style="width:100%; border-radius:8px;" alt="CNN Architecture">
        </div>
        <h4>Key Features:</h4>
        <ul>
            <li><strong>Custom CNN</strong> built with PyTorch</li>
            <li>3 convolutional layers with max pooling</li>
            <li>2 fully connected layers for classification</li>
            <li>98.5% validation accuracy</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div class="feature-card">
        <h3>Tumor Classifier</h3>
        <p>Identifies tumor type from validated MRI scans</p>
        <div class="model-diagram">
            <img src="https://imghost.net/ib/1axVhAGyExn8Qqt_1744197375.png" 
                 style="width:100%; border-radius:8px;" alt="Classifier Architecture">
        </div>
        <h4>Classification Capabilities:</h4>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
            <div style="background: #f0f7ff; padding: 10px; border-radius: 8px;">
                <h5>Glioma</h5>
                <p>Most common primary brain tumor</p>
            </div>
            <div style="background: #f0f7ff; padding: 10px; border-radius: 8px;">
                <h5>Meningioma</h5>
                <p>Typically benign tumors</p>
            </div>
            <div style="background: #f0f7ff; padding: 10px; border-radius: 8px;">
                <h5>Pituitary</h5>
                <p>Affects hormone regulation</p>
            </div>
            <div style="background: #f0f7ff; padding: 10px; border-radius: 8px;">
                <h5>No Tumor</h5>
                <p>Healthy brain scan</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with tab3:
    st.markdown("""
    <div class="feature-card">
        <h3>Segmentation Model</h3>
        <p>Our segmentation system is powered by a transfer learning-based CNN model, trained using paired MRI scans and corresponding tumor mask images.</p>
        <div class="model-diagram">
            <img src="https://imghost.net/ib/uAlcRZ3PNjT7apC_1744199073.png" 
                 style="width:100%; border-radius:8px;" alt="CNN Transfer Learning Segmentation">
        </div>
        <h4>Segmentation Workflow:</h4>
        <ul>
            <li>Uses pre-trained CNN (e.g., ResNet50) for high-level feature extraction</li>
            <li>Trained on MRI scans with ground-truth tumor masks</li>
            <li>Fine-tuned on additional data to improve performance</li>
            <li>Achieved <strong>89% Dice Score</strong> on validation set</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# Interactive Pipeline Visualization
st.header("🔍 Our Analysis Pipeline", divider="blue")

steps = [
    {
        "title": "Image Upload & Validation",
        "description": "The system first verifies that uploaded images are valid MRI scans using our validation model.",
        "visual": "https://cdn-icons-png.flaticon.com/512/3652/3652191.png"
    },
    {
        "title": "Preprocessing",
        "description": "Images are resized, normalized, and converted to appropriate formats for each model.",
        "visual": "https://cdn-icons-png.flaticon.com/512/3344/3344376.png"
    },
    {
        "title": "Feature Extraction",
        "description": "The models analyze textures, shapes, and patterns in the MRI scans.",
        "visual": "https://cdn-icons-png.flaticon.com/512/3344/3344387.png"
    },
    {
        "title": "Classification & Segmentation",
        "description": "The system identifies tumor type and creates precise segmentation masks.",
        "visual": "https://cdn-icons-png.flaticon.com/512/2933/2933245.png"
    },
    {
        "title": "Results Visualization",
        "description": "Findings are presented with clear visualizations and metrics.",
        "visual": "https://cdn-icons-png.flaticon.com/512/2936/2936886.png"
    }
]

for i, step in enumerate(steps, 1):
    with st.container():
        st.markdown(f"""
        <div class="step-container">
            <div class="step-number">{i}</div>
            <div style="flex-grow:1">
                <h3>{step['title']}</h3>
                <p>{step['description']}</p>
            </div>
            <div>
                <img src="{step['visual']}" width="80" alt="Step {i}">
            </div>
        </div>
        """, unsafe_allow_html=True)

# Performance Metrics
st.header("📊 Model Performance", divider="blue")

col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=98.5,
        title={'text': "MRI Validation Accuracy"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#3b82f6"},
            'steps': [
                {'range': [0, 60], 'color': "#fee2e2"},
                {'range': [60, 80], 'color': "#fef3c7"},
                {'range': [80, 100], 'color': "#dcfce7"}
            ]
        },
        number={'suffix': '%'}
    ))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=94.2,
        title={'text': "Tumor Classification Accuracy"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#3b82f6"},
            'steps': [
                {'range': [0, 60], 'color': "#fee2e2"},
                {'range': [60, 80], 'color': "#fef3c7"},
                {'range': [80, 100], 'color': "#dcfce7"}
            ]
        },
        number={'suffix': '%'}
    ))
    st.plotly_chart(fig, use_container_width=True)

# Interpretation Guide
st.header("🔬 Understanding Your Results", divider="blue")

st.markdown("""
<div class="feature-card">
    <h3>Reading the Analysis Report</h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
        <div>
            <h4>📈 Confidence Scores</h4>
            <p>Our models provide confidence percentages for each prediction. Higher values indicate greater certainty.</p>
            <div style="background: #f0f7ff; padding: 15px; border-radius: 8px;">
                <p><strong>Interpretation Guide:</strong></p>
                <p>✅ <strong>90-100%</strong>: Very high confidence</p>
                <p>🟢 <strong>75-90%</strong>: High confidence</p>
                <p>🟡 <strong>60-75%</strong>: Moderate confidence</p>
                <p>🔴 <strong>Below 60%</strong>: Low confidence</p>
            </div>
        </div>
        <div>
            <h4>🎨 Segmentation Colors</h4>
            <p>The tumor mask uses a color scale to show probability:</p>
            <img src="https://matplotlib.org/stable/_images/sphx_glr_colormap_reference_003.png" 
                 style="width:100%; border-radius:8px; margin-top:10px;" alt="Color Scale">
            <p>Warmer colors indicate higher probability of tumor tissue.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# FAQ Section
st.header("❓ Frequently Asked Questions", divider="blue")

with st.expander("How accurate is the tumor detection system?"):
    st.markdown("""
    Our system achieves:
    - 98.5% accuracy in MRI validation
    - 94.2% accuracy in tumor classification
    - 89% Dice score in tumor segmentation
    
    These metrics are based on testing with thousands of MRI scans from diverse sources.
    """)


with st.expander("How is tumor size calculated?"):
    st.markdown("""
    Tumor size is calculated by:
    1. Creating a precise segmentation mask
    2. Identifying all tumor pixels
    3. Calculating area based on scan resolution
    4. Converting to real-world measurements (mm²)
    
    The size estimation accounts for slice thickness and scan parameters.
    """)

# Call to Action
st.markdown("""
<div style="text-align: center; margin: 40px 0; padding: 30px; background: #f0f7ff; border-radius: 12px;">
    <h2>Ready to analyze your MRI scans?</h2>
    <p>Experience our advanced brain tumor detection system firsthand</p>
    <a href="/?page=Analysis" target="_self" style="
        display: inline-block;
        padding: 12px 24px;
        background: #3b82f6;
        color: white;
        border-radius: 8px;
        text-decoration: none;
        font-weight: bold;
        margin-top: 15px;
    ">Start Analysis Now</a>
</div>
""", unsafe_allow_html=True)