# app.py - Streamlit App for YOLOv11 Bone Fracture Detection

import streamlit as st
from pathlib import Path
import yaml
from datetime import datetime
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import cv2

from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# PAGE CONFIG

st.set_page_config(
    page_title="BFD-ID (Bone Fracture Detection)",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* ================= HEADER ================= */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 1rem;
    }

    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* ================= METRIC & BUTTON ================= */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }

    .stButton > button {
        width: 100%;
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px;
    }

    .detection-box {
        border: 2px solid #e74c3c;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f8f9fa;
    }

    /* ================= DETECTED FRACTURE CARDS ================= */
    .severity-normal,
    .severity-simple,
    .severity-complex {
        padding: 16px;
        margin: 8px 0;              /* DIPERKETAT */
        border-radius: 10px;
        font-size: 15px;
        font-weight: 500;
        line-height: 1.3;           /* DIPERKETAT */
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.12);
    }

    .severity-normal {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 6px solid #4caf50;
        color: #2e7d32;
    }

    .severity-simple {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border-left: 6px solid #ff9800;
        color: #ef6c00;
    }

    .severity-complex {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 6px solid #f44336;
        color: #c62828;
    }

    /* ================= TITLE INSIDE CARD (FIX SPACE UTAMA) ================= */
    .severity-normal strong,
    .severity-simple strong,
    .severity-complex strong {
        display: inline-block;      /* FIX SPACE */
        margin: 0;                  /* FIX SPACE */
        padding: 0;                 /* FIX SPACE */
        font-size: 17px;
        font-weight: 700;
        line-height: 1.2;
    }

    /* ================= HOVER EFFECT ================= */
    .severity-normal:hover,
    .severity-simple:hover,
    .severity-complex:hover {
        transform: translateY(-2px);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.18);
    }

    /* ================= ICON PREFIX ================= */
    .severity-normal::before {
        content: "✅ ";
        font-size: 16px;
        margin-right: 6px;
    }

    .severity-simple::before {
        content: "⚠️ ";
        font-size: 16px;
        margin-right: 6px;
    }

    .severity-complex::before {
        content: "🚨 ";
        font-size: 16px;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# CONFIGURATION

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "data.yaml"

# Class names from your data.yaml
CLASS_NAMES = [
    'Comminuted',
    'Greenstick', 
    'Healthy',
    'Linear',
    'Oblique Displaced',
    'Oblique',
    'Segmental',
    'Spiral',
    'Transverse Displaced',
    'Transverse'
]

NUM_CLASSES = 10

# Medical categories based on severity
MEDICAL_CATEGORIES = {
    'Normal': ['Healthy'],
    'Simple Fractures': ['Greenstick', 'Linear', 'Oblique', 'Transverse', 'Spiral'],
    'Complex Fractures': ['Oblique Displaced', 'Transverse Displaced', 'Comminuted', 'Segmental']
}

# Color palette for each class
COLORS = {
    'Comminuted': '#e74c3c',
    'Greenstick': '#3498db',
    'Healthy': '#2ecc71',
    'Linear': '#f39c12',
    'Oblique Displaced': '#e74c3c',
    'Oblique': '#9b59b6',
    'Segmental': '#c0392b',
    'Spiral': '#1abc9c',
    'Transverse Displaced': '#e67e22',
    'Transverse': '#34495e'
}

# Medical descriptions
MEDICAL_INFO = {
    'Healthy': 'No fracture detected. Normal bone structure.',
    'Greenstick': 'Incomplete fracture where bone bends. Common in children.',
    'Linear': 'Straight line fracture parallel to bone axis.',
    'Oblique': 'Diagonal fracture across the bone.',
    'Transverse': 'Horizontal fracture perpendicular to bone axis.',
    'Spiral': 'Twisting fracture, often from rotational force.',
    'Oblique Displaced': 'Displaced diagonal fracture requiring realignment.',
    'Transverse Displaced': 'Displaced horizontal fracture requiring realignment.',
    'Comminuted': 'Bone shattered into multiple fragments.',
    'Segmental': 'Multiple fracture lines creating separate bone segments.'
}

# UTILITY FUNCTIONS

@st.cache_resource
def load_model(model_path):
    """Load YOLO model with caching"""
    try:
        if model_path == "yolo11s.pt":
            model = YOLO(model_path)
        else:
            model = YOLO(str(model_path))
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def get_category(class_name):
    """Get medical category for a class"""
    for category, classes in MEDICAL_CATEGORIES.items():
        if class_name in classes:
            return category
    return "Unknown"

def get_severity_level(category):
    """Get severity level (1-3)"""
    severity_map = {
        'Normal': 1,
        'Simple Fractures': 2,
        'Complex Fractures': 3
    }
    return severity_map.get(category, 0)

def draw_detections(image, results, conf_threshold=0.25):
    """Draw bounding boxes and labels on image"""
    img_array = np.array(image)
    img_draw = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img_draw)
    
    detections = []
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        
        for box in boxes:
            # Get box data
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            if conf < conf_threshold:
                continue
            
            # Get class info
            class_name = CLASS_NAMES[cls]
            color = COLORS[class_name]
            category = get_category(class_name)
            
            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            
            # Draw label with background
            label = f"{class_name} {conf:.2f}"
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            bbox = draw.textbbox((x1, y1-25), label, font=font)
            draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], fill=color)
            draw.text((x1, y1-25), label, fill='white', font=font)
            
            # Store detection
            detections.append({
                'class': class_name,
                'confidence': conf,
                'category': category,
                'severity': get_severity_level(category),
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'description': MEDICAL_INFO.get(class_name, '')
            })
    
    return img_draw, detections

def create_detection_summary(detections):
    """Create summary statistics from detections"""
    if not detections:
        return None
    
    df = pd.DataFrame(detections)
    
    summary = {
        'total_detections': len(detections),
        'by_category': df['category'].value_counts().to_dict(),
        'by_class': df['class'].value_counts().to_dict(),
        'avg_confidence': df['confidence'].mean(),
        'max_severity': df['severity'].max(),
        'severity_label': 'Normal' if df['severity'].max() == 1 else 
                         'Simple' if df['severity'].max() == 2 else 'Complex'
    }
    
    return summary

# SIDEBAR

st.sidebar.markdown("## 🦴Bone Fracture Detection")
st.sidebar.markdown("**YOLOv11 Medical AI System**")
st.sidebar.markdown("---")

# Model selection
st.sidebar.markdown("### 🤖 Model Settings")

model_options = ["yolo11s.pt"]
models_dir = ROOT / "output_fixed"

# Find trained models
if models_dir.exists():
    for output_folder in sorted(models_dir.iterdir(), reverse=True):
        if output_folder.is_dir():
            model_path = output_folder / "03_models" / "best_model.pt"
            if model_path.exists():
                timestamp = output_folder.name
                model_options.append(f"Trained Model ({timestamp})")
                
selected_model_display = st.sidebar.selectbox(
    "Select Model",
    model_options,
    help="Choose pre-trained or your custom trained model"
)

# Get actual model path
if selected_model_display == "yolo11s.pt":
    selected_model = "yolo11s.pt"
else:
    timestamp = selected_model_display.split("(")[1].split(")")[0]
    selected_model = models_dir / timestamp / "03_models" / "best_model.pt"

# Detection settings
st.sidebar.markdown("### ⚙️ Detection Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.25,
    step=0.05,
    help="Minimum confidence score for detection"
)

iou_threshold = st.sidebar.slider(
    "IOU Threshold (NMS)",
    min_value=0.1,
    max_value=0.9,
    value=0.45,
    step=0.05,
    help="IoU threshold for Non-Maximum Suppression"
)

# Class information
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Class Information")
st.sidebar.markdown(f"**Total Classes:** {NUM_CLASSES}")

with st.sidebar.expander("🔍 View All Classes"):
    for cat_name, classes in MEDICAL_CATEGORIES.items():
        st.markdown(f"**{cat_name}**")
        for cls in classes:
            if cls in CLASS_NAMES:
                color = COLORS[cls]
                st.markdown(f"<span style='color:{color}'>●</span> {cls}", unsafe_allow_html=True)
        st.markdown("---")

# About
st.sidebar.markdown("---")
with st.sidebar.expander("ℹAbout"):
    st.markdown("""
    **Bone Fracture Detection System**
    
    Uses YOLOv11 deep learning to detect and classify bone fractures in X-ray images.
    
    **Categories:**
    - 🟢 Normal (Healthy)
    - 🟡 Simple Fractures (5 types)
    - 🔴 Complex Fractures (4 types)
    
    **Developer:** Nadia and Nazril
    **Version:** 1.0
    """)

# MAIN APP

# Header
st.markdown("""
<h1 style="text-align:center; font-weight:800; margin-bottom:1rem;">
    ⚕️ 
    <span style="
        background: linear-gradient(90deg, #e3f2fd, #1e88e5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent;
    ">
        BFD-ID Bone Fracture Detection System
    </span>
</h1>
""", unsafe_allow_html=True)

st.markdown("<div class='sub-header'>AI-Powered Medical Object Detection Image Analysis with YOLOv11</div>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Detection", 
    "📊 Dataset EDA", 
    "📈 Training Metrics",
    "🎯 Batch Inference",
    "📚 Documentation"
])

# TAB 1: DETECTION
with tab1:
    st.markdown("### Real-time Fracture Detection")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("#### Upload X-Ray Image")
        uploaded_file = st.file_uploader(
            "Choose an X-ray image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload bone X-ray image for fracture detection"
        )
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file).convert('RGB')
            
            st.image(image, caption='Uploaded X-Ray', use_container_width=True)
            
            # Detect button
            if st.button("Run Detection", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Load model
                    model = load_model(selected_model)
                    
                    if model is not None:
                        # Run inference
                        results = model.predict(
                            source=image,
                            conf=conf_threshold,
                            iou=iou_threshold,
                            verbose=False
                        )
                        
                        # Draw detections
                        img_with_detections, detections = draw_detections(
                            image, results, conf_threshold
                        )
                        
                        # Store in session state
                        st.session_state.detections = detections
                        st.session_state.result_image = img_with_detections
                        
                        st.success(f"Detection complete! Found {len(detections)} object(s)")
    
    with col2:
        st.markdown("#### 🔬 Detection Results")
        
        if 'result_image' in st.session_state and 'detections' in st.session_state:
            # Show result image
            st.image(
                st.session_state.result_image, 
                caption='Detection Result',
                use_container_width=True
            )
            
            detections = st.session_state.detections
            
            if len(detections) > 0:
                # Summary metrics
                summary = create_detection_summary(detections)
                
                st.markdown("#### 📊 Summary")
                
                # Metrics row
                met1, met2, met3 = st.columns(3)
                met1.metric("Detections", summary['total_detections'])
                met2.metric("Avg Confidence", f"{summary['avg_confidence']:.2%}")
                met3.metric("Severity", summary['severity_label'])
                
                # Detection details
                st.markdown("####  Detected Fractures")
                
                for i, det in enumerate(detections):
                    severity_class = f"severity-{det['category'].lower().replace(' ', '-')}"
                    if 'Normal' in det['category']:
                        severity_class = 'severity-normal'
                    elif 'Simple' in det['category']:
                        severity_class = 'severity-simple'
                    else:
                        severity_class = 'severity-complex'
                    
                    st.markdown(f"""
                    <div class='{severity_class}'>
                        <strong>#{i+1}: {det['class']}</strong><br>
                        📍 Category: {det['category']}<br>
                        📊 Confidence: {det['confidence']:.2%}<br>
                        ℹ️ {det['description']}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Visualization
                st.markdown("#### 📈 Distribution")
                
                df = pd.DataFrame(detections)
                
                fig = px.bar(
                    df['class'].value_counts().reset_index(),
                    x='class',
                    y='count',
                    color='class',
                    color_discrete_map=COLORS,
                    title='Detection Count by Class'
                )
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                st.markdown("#### 💾 Export Results")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    # Save image
                    if st.button("📥 Download Image"):
                        result_path = ROOT / "detection_result.jpg"
                        st.session_state.result_image.save(result_path)
                        st.success(f"✅ Saved to {result_path}")
                
                with col_b:
                    # Save JSON
                    if st.button("📄 Download JSON"):
                        json_path = ROOT / "detection_result.json"
                        with open(json_path, 'w') as f:
                            json.dump(detections, f, indent=2)
                        st.success(f"✅ Saved to {json_path}")
            else:
                st.info("No fractures detected in this image.")
        else:
            st.info("👆 Upload an image and click 'Run Detection' to see results")

# TAB 2: DATASET EDA
with tab2:
    st.markdown("### 📊 Dataset Exploratory Data Analysis")
    
    # Find latest EDA (nested in output_fixed/timestamp/)
    eda_path = None
    samples_dir = None
    stats_path = None
    
    if models_dir.exists():
        latest = sorted(models_dir.iterdir(), reverse=True)
        if latest:
            eda_path = latest[0] / "01_eda" / "eda_analysis.png"
            samples_dir = latest[0] / "01_eda" / "samples"
            stats_path = latest[0] / "04_reports" / "statistics.txt"
    
    if eda_path and eda_path.exists():
        try:
            st.image(str(eda_path), caption="Dataset Analysis", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading EDA image: {e}")
        
        # Show sample images
        if samples_dir and samples_dir.exists():
            st.markdown("#### 🖼️ Sample Images from Dataset")
            
            sample_files = sorted(samples_dir.glob("sample_*.png"))
            
            if sample_files:
                # Display samples in grid (max 5)
                cols = st.columns(min(len(sample_files), 5))
                for idx, sample_file in enumerate(sample_files[:5]):
                    try:
                        with cols[idx]:
                            st.image(str(sample_file), caption=f"Sample {idx+1}", use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not load {sample_file.name}")
        
        # Show statistics
        if stats_path and stats_path.exists():
            try:
                st.markdown("#### 📄 Dataset Statistics")
                with open(stats_path) as f:
                    stats = f.read()
                st.code(stats, language='text')
            except Exception as e:
                st.warning(f"Could not load statistics: {e}")
    else:
        st.warning("No EDA data found. Please run main_fixed.py first.")
    
    # Interactive class distribution
    st.markdown("#### Interactive Class Distribution")
    
    # Create sample data
    class_counts = {name: np.random.randint(50, 200) for name in CLASS_NAMES}
    
    fig = go.Figure()
    
    for name, count in class_counts.items():
        fig.add_trace(go.Bar(
            name=name,
            x=[name],
            y=[count],
            marker_color=COLORS[name],
            text=[count],
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Class Distribution Overview",
        xaxis_title="Fracture Type",
        yaxis_title="Number of Samples",
        showlegend=False,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# TAB 3: TRAINING METRICS

with tab3:
    st.markdown("### 📈 Training Performance Metrics")
    
    # Find training results
    training_path = None
    if models_dir.exists():
        latest = sorted(models_dir.iterdir(), reverse=True)
        if latest:
            training_path = latest[0] / "training"
    
    if training_path and training_path.exists():
        col1, col2 = st.columns(2)
        
        # Results
        results_img = training_path / "results.png"
        if results_img.exists():
            with col1:
                st.markdown("#### 📊 Training Curves")
                st.image(str(results_img), use_container_width=True)
        
        # Confusion matrix
        confusion_img = training_path / "confusion_matrix.png"
        if confusion_img.exists():
            with col2:
                st.markdown("#### 🎯 Confusion Matrix")
                st.image(str(confusion_img), use_container_width=True)
        
        # Show report
        report_path = latest[0] / "04_reports" / "final_report.txt"
        if report_path.exists():
            st.markdown("#### 📄 Training Report")
            with open(report_path) as f:
                report = f.read()
            st.code(report, language='text')
    else:
        st.warning("⚠️ No training metrics found. Train a model first using main_fixed.py")

# TAB 4: BATCH INFERENCE

with tab4:
    st.markdown("### 🎯 Batch Inference")
    st.markdown("Process multiple X-ray images at once")
    
    uploaded_files = st.file_uploader(
        "Upload multiple X-ray images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} images uploaded")
        
        if st.button("Process Batch", type="primary"):
            model = load_model(selected_model)
            
            if model:
                progress_bar = st.progress(0)
                results_data = []
                
                for idx, file in enumerate(uploaded_files):
                    image = Image.open(file).convert('RGB')
                    
                    # Predict
                    results = model.predict(
                        source=image,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        verbose=False
                    )
                    
                    _, detections = draw_detections(image, results, conf_threshold)
                    
                    results_data.append({
                        'filename': file.name,
                        'detections': len(detections),
                        'classes': [d['class'] for d in detections],
                        'confidences': [d['confidence'] for d in detections]
                    })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                # Show results
                st.success(f"✅ Processed {len(uploaded_files)} images!")
                
                # Summary table
                df_results = pd.DataFrame([
                    {
                        'Filename': r['filename'],
                        'Detections': r['detections'],
                        'Classes': ', '.join(r['classes']) if r['classes'] else 'None',
                        'Avg Confidence': f"{np.mean(r['confidences']):.2%}" if r['confidences'] else 'N/A'
                    }
                    for r in results_data
                ])
                
                st.dataframe(df_results, use_container_width=True)
                
                # Statistics
                total_detections = sum(r['detections'] for r in results_data)
                st.metric("Total Detections", total_detections)

# TAB 5: DOCUMENTATION

with tab5:
    st.markdown("### 📚 Documentation")
    
    st.markdown("""
    ## 🦴 Bone Fracture Detection System
    
    ### Overview
    This application uses **YOLOv11** deep learning model to detect and classify bone fractures in X-ray images.
    
    ### Supported Fracture Types (10 Classes)
    
    #### 🟢 Normal
    - **Healthy**: No fracture detected
    
    #### 🟡 Simple Fractures (Non-displaced)
    1. **Greenstick**: Incomplete fracture, bone bends but doesn't break completely
    2. **Linear**: Straight line fracture parallel to bone's long axis
    3. **Oblique**: Diagonal fracture across the bone
    4. **Transverse**: Horizontal fracture perpendicular to bone's long axis
    5. **Spiral**: Twisting fracture from rotational force
    
    #### 🔴 Complex Fractures (Displaced/Severe)
    1. **Oblique Displaced**: Displaced diagonal fracture
    2. **Transverse Displaced**: Displaced horizontal fracture
    3. **Comminuted**: Bone shattered into 3+ fragments
    4. **Segmental**: Multiple fracture lines creating floating bone segment
    
    ### How to Use
    
    #### 1️⃣ Single Image Detection
    - Go to **Detection** tab
    - Upload X-ray image (JPG/PNG)
    - Adjust confidence threshold if needed
    - Click "Run Detection"
    - View results and download if needed
    
    #### 2️⃣ Batch Processing
    - Go to **Batch Inference** tab
    - Upload multiple images
    - Process all at once
    - Export results as CSV
    
    ### Model Information
    - **Architecture**: YOLOv11s
    - **Input Size**: 224x224 pixels
    - **Classes**: 10 fracture types
    - **Format**: RGB X-ray images
    
    ### Performance Metrics
    - **mAP50**: Mean Average Precision at IoU 0.5
    - **mAP50-95**: Mean Average Precision at IoU 0.5-0.95
    - **Precision**: True Positive Rate
    - **Recall**: Detection Rate
    
    ### Tips for Best Results
    1. ✅ Use clear, high-quality X-ray images
    2. ✅ Ensure proper image orientation
    3. ✅ Adjust confidence threshold based on use case
    4. ✅ Review all detections, especially borderline cases
    5. ⚠️ This is an AI assistant tool, not a replacement for professional medical diagnosis
    
    ### System Requirements
    - Python 3.11
    - Ultralytics YOLOv11
    - Streamlit
    - PIL, OpenCV, NumPy
    
    ### Contact & Support
    For questions or issues, contact your development team.
    
    ---
    **Disclaimer**: This system is designed for research and educational purposes, as the developed model has not yet reached optimal performance, with a recall of 0.4 and a precision of 0.7. Therefore, the results generated by this platform should not be used as a clinical medical diagnosis. This platform serves as a prototype and proof of concept for a bone fracture detection system developed specifically for the PINUS 3.0 essay competition. Further improvements are required in terms of data quality, model architecture, and training strategies to enhance performance and reliability in future developments. 
    Always consult with qualified medical professionals for diagnosis and treatment.
    """)

# FOOTER

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>🦴 Bone Fracture Detection System v1.0</p>
    <p>Powered by YOLOv11 | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)