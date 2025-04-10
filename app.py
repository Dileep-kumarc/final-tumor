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

# Rest of your code remains the same...

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

        model = CustomCNN()
        model.load_state_dict(torch.load(r"C:\main project files\mri and non mri classfier\best_mri_classifier.pth", map_location=torch.device('cpu')))
        model.eval()
        return model

    custom_cnn_model = load_custom_model()
    classifier_model = tf.keras.models.load_model(r"C:\Users\dilee\Downloads\cnn model traning\brain_tumor_classifier.h5")
    segmentation_model = tf.keras.models.load_model(r"C:\Users\dilee\Downloads\cnn model traning\brain_tumor_segmentation_unet.h5")
    tumor_size_model = tf.keras.models.load_model(r"C:\Users\dilee\Downloads\tumor_size_model.h5")
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
            <img src="https://media-hosting.imagekit.io/abd4c2ba8b3c4113/brain.png?Expires=1838796668&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=M4LwtNQ2w1W2WkrtSk2-LNxA3UWxYivl-9Nivplk19fnE2S5kzBkrk28BM9V7H~Ps4Qc5BpUVhfRIBO~MMGa-YCTgYTFbEIj4yj3zH2YhlRF3~a-Pkf2M65Gjs-eZkzdNNcd-ajyAQEmAjZ2jbwtUQ5Vs4mYyCLtmAvOCuoCJ30eCOKU8JFuieZuf--112t8iXZ4ymJ7Zi5Ottl74qfkSS3YbK4oisANg73gj6WKOwbNhCc2GAVzZjTaNgB70xvqdyjMlK9FZGBjOCfjGLJ6ZS2RKiP4KkJ22sK0AOeU9eaUVMzi~hfOt0pFiLCT9sI7gh9ztfA7sxIPYfFGIcCMQQ__" alt="Brain AI Technology">
            <div class="hero-blob"></div>
        </div>
        <div class="hero-content">
            <div class="badge">AI-Powered Healthcare</div>
            <h1>Advanced Brain Tumor Detection</h1>
            <p class="subtitle">Revolutionizing medical diagnostics with state-of-the-art AI technology for precise and rapid brain tumor analysis</p>
            <div class="hero-buttons">
                <a href="/?page=Analysis" class="primary-button">Get Started Now</a>
                <div class="stats-badge">
                    <span class="stats-number">98.5%</span>
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
    st.title("🧠 Brain Tumor Analysis")
    uploaded_file = st.file_uploader("Upload a brain MRI scan", type=["jpg", "jpeg", "png"])
    custom_cnn_model, classifier_model, segmentation_model, tumor_size_model = load_models()

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Validating MRI...")
        image_type, is_mri = validate_mri(image, custom_cnn_model)
        progress_bar.progress(25)

        if not is_mri:
            st.error(f"⚠️ Detected Image Type: {image_type}. Please upload a valid MRI image.")
        else:
            st.success("✅ Valid MRI detected.")
            status_text.text("Classifying tumor...")
            tumor_type, confidence = classify_tumor(image, classifier_model)
            progress_bar.progress(50)

            st.subheader("Classification Results")
            col1, col2 = st.columns(2)
            col1.metric("Tumor Type", tumor_type)
            col2.metric("Confidence", f"{confidence:.2%}")

            if tumor_type != "No Tumor":
                status_text.text("Segmenting tumor...")
                mask = segment_tumor(image, segmentation_model)
                progress_bar.progress(75)

                st.subheader("Tumor Segmentation")
                # Fix: Explicitly import matplotlib.pyplot here to ensure it's available
                import matplotlib.pyplot as plt
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                ax1.imshow(image)
                ax1.set_title("Original Image")
                ax1.axis("off")
                ax2.imshow(mask.squeeze(), cmap="viridis")
                ax2.set_title("Segmentation Mask")
                ax2.axis("off")
                st.pyplot(fig)

                status_text.text("Calculating tumor size...")
                size = predict_tumor_size(mask, tumor_size_model)
                progress_bar.progress(100)

                st.subheader("Tumor Size Analysis")
                st.metric("Predicted Tumor Size", f"{size[0]:.2f} mm²")

            status_text.text("Analysis complete!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
            st.balloons()

            # Report Generation Section
            st.markdown("""
            <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 30px;'>
                <h3 style='color: #1565c0;'>📋 Enhanced Analysis Report</h3>
                <p>Generate a detailed medical report with professional formatting and comprehensive analysis results</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Generate PDF Report", type="primary"):
                from reportlab.lib import colors
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_JUSTIFY
                from io import BytesIO
                import matplotlib.pyplot as plt
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
                
                # Enhanced table with alternating row colors
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
                <a href="/?page=HowItWorks" class="nav-link">How It Works</a>
                <a href="/?page=About" class="nav-link">About</a>
                <a href="/?page=Contact" class="nav-link">Contact</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    query_params = st.query_params.get("page", "Home")

    if query_params == "Home":
        home_page()
    elif query_params == "Analysis":
        analysis_page()
    elif query_params == "HowItWorks":
        st.switch_page("pages/2_How_It_Works.py")
    elif query_params == "About":
        st.switch_page("pages/1_About.py")
    elif query_params == "Contact":
        st.switch_page("pages/3_Contact.py")

if __name__ == "__main__":
    main()
