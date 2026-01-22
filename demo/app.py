#!/usr/bin/env python3
"""
Streamlit demo app for anomaly detection using diffusion model
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots  # <--- MỚI THÊM
import pandas as pd
import torch
import json
from collections import defaultdict
import time
import tempfile
import shutil
import glob

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from inference import preprocess_image, predict, load_checkpoint, defaultdict_from_json, denormalize_image, min_max_norm, cvt2heatmap, show_cam_on_image, predict_batch, predict_single_image_array
    from models import UNetModel, SegmentationSubNetwork, GaussianDiffusionModel, get_beta_schedule
except ImportError as e:
    st.error(f"Failed to import modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Diffusion Anomaly Detection Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .anomaly-score {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .normal-score {
        color: #28a745;
    }
    .anomaly-score-high {
        color: #dc3545;
    }
    /* Điều chỉnh style cho plotly chart nếu cần */
</style>
""", unsafe_allow_html=True)

def get_available_models(root_dir="outputs"):
    """Recursively find .pt files in the outputs directory"""
    model_files = []
    if os.path.exists(root_dir):
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".pt"):
                    # Create a relative path for display
                    full_path = os.path.join(root, file)
                    model_files.append(full_path)
    return sorted(model_files)

@st.cache_resource
def load_models(model_path, device_type="auto"):
    """Load trained diffusion models from a specific path"""
    
    # Determine device
    device = torch.device("cpu")
    if device_type == "cuda" or (device_type == "auto" and torch.cuda.is_available()):
        try:
            # Check if current PyTorch version actually supports the GPU's compute capability
            torch.randn(1, device="cuda")
            device = torch.device("cuda")
        except Exception as e:
            if device_type == "cuda":
                st.error(f"CUDA device requested but it is incompatible with current PyTorch installation: {e}")
            else:
                st.sidebar.warning(f"GPU detected but incompatible with PyTorch (Capability mismatch). Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    
    # Load model checkpoint
    ckpt_state = None
    if os.path.exists(model_path):
        try:
            ckpt_state = load_checkpoint(model_path, device)
        except Exception as e:
            st.error(f"Failed to load model from {model_path}: {e}")
            return None
    else:
        st.error(f"Model path does not exist: {model_path}")
        return None
    
    if ckpt_state is None:
        st.error("No model checkpoint found!")
        return None

    # Args loaded from checkpoint state_dict
    args = None
    if 'args' in ckpt_state:
        args = ckpt_state['args']
        # Convert dict to defaultdict if necessary
        if not isinstance(args, defaultdict):
            try:
                args = defaultdict_from_json(args)
            except Exception as e:
                st.warning(f"Args in checkpoint could not be converted to defaultdict: {e}")
    else:
        st.error("No args found in checkpoint state!")
        return None

    try:
        # Initialize models
        unet_model = UNetModel(
            args['img_size'][0], 
            args['base_channels'], 
            channel_mults=args['channel_mults'], 
            dropout=args["dropout"], 
            n_heads=args["num_heads"], 
            n_head_channels=args["num_head_channels"],
            in_channels=args["channels"]
        ).to(device)

        seg_model = SegmentationSubNetwork(in_channels=args["channels"] * 2, out_channels=1).to(device)

        # Initialize DDPM
        betas = get_beta_schedule(args['T'], args['beta_schedule'])
        ddpm = GaussianDiffusionModel(
            args['img_size'], 
            betas, 
            loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], 
            noise=args["noise_fn"], 
            img_channels=args["channels"]
        )
        
        # Load state dicts
        unet_model.load_state_dict(ckpt_state['unet_model_state_dict'])
        seg_model.load_state_dict(ckpt_state['seg_model_state_dict'])
        unet_model.eval()
        seg_model.eval()
        
        return {
            'unet_model': unet_model,
            'seg_model': seg_model,
            'ddpm': ddpm,
            'args': args,
            'device': device,
            'model_path': model_path
        }
        
    except Exception as e:
        st.error(f"Failed to initialize models: {e}")
        return None

def predict_single_image(models, image_array, heatmap_threshold=0.6):
    """Predict anomaly for single image using diffusion model"""
    unet_model = models['unet_model']
    seg_model = models['seg_model']
    ddpm = models['ddpm']
    args = models['args']
    device = models['device']
    
    # Use the function from inference.py
    return predict_single_image_array(unet_model, seg_model, ddpm, image_array, args, device, heatmap_threshold)

def predict_batch_images(models, image_arrays, batch_size=8, heatmap_threshold=0.6, progress_bar=None, status_text=None):
    """Predict anomalies for a batch of images with parallel processing"""
    unet_model = models['unet_model']
    seg_model = models['seg_model']
    ddpm = models['ddpm']
    args = models['args']
    device = models['device']
    
    # Progress callback function
    def progress_callback(progress, status):
        if progress_bar is not None and status_text is not None:
            progress_bar.progress(progress)
            status_text.text(status)
    
    # Use predict_batch from inference.py
    results = predict_batch(
        unet_model, seg_model, ddpm, image_arrays, args, device, 
        heatmap_threshold=heatmap_threshold, 
        batch_size=batch_size,
        progress_callback=progress_callback if progress_bar is not None else None
    )
    
    return results

def display_prediction_results(result, method, image):
    """Display prediction results with interactive Plotly visualization"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Anomaly Score",
            value=f"{result['anomaly_score']:.4f}",
            delta=None
        )
    
    with col2:
        is_anomaly = result['is_anomaly']
        status = "Anomaly" if is_anomaly else "Normal"
        color = "red" if is_anomaly else "green"
        st.markdown(f"<div style='color: {color}; font-size: 1.2em; font-weight: bold;'>{status}</div>", 
                   unsafe_allow_html=True)
    
    with col3:
        confidence = abs(result['anomaly_score'])
        st.metric(
            label="Confidence",
            value=f"{confidence:.4f}",
            delta=None
        )
    
    with col4:
        if 'inference_time' in result:
            st.metric(
                label="Inference Latency",
                value=f"{result['inference_time']*1000:.2f} ms",
                delta=None
            )
        else:
            st.metric(
                label="Inference Latency",
                value="N/A",
                delta=None
            )
    
    # Display visualizations for diffusion model using Plotly for zoom capability
    if 'heatmap' in result:
        st.subheader("Visual Analysis (Interactive Zoom/Pan)")
        
        # Define titles and image keys
        titles = ["Original", "Reconstruction", "Recon (Noisier)", "Anomaly Mask", "Heatmap Overlay"]
        keys = ['original_image', 'reconstructed_image', 'recon_noisier', 'anomaly_mask', 'heatmap_overlay']
        
        # Create subplots: 1 row, 5 columns
        fig = make_subplots(
            rows=1, cols=5,
            subplot_titles=titles,
            horizontal_spacing=0.01,
            vertical_spacing=0.02
        )
        
        # Helper to ensure image format is correct for Plotly
        def process_for_plotly(img):
            # Ensure it's numpy array
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            # If float 0-1, convert to 0-255 uint8 for consistent display
            if img.dtype.kind == 'f' and img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                img = img.astype(np.uint8)
            return img

        # Add each image to the figure
        for i, key in enumerate(keys):
            if key in result:
                img_data = process_for_plotly(result[key])
                fig.add_trace(
                    go.Image(z=img_data, hoverinfo='skip'), # skip hover to make it cleaner
                    row=1, col=i+1
                )
        
        # Update layout properties
        fig.update_layout(
            height=350,  # Adjust height
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            dragmode='zoom'  # Enable zoom by default
        )
        
        # Hide axis ticks for all subplots to look like cleaner images
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        
        # Render the interactive chart
        st.plotly_chart(fig, use_container_width=True)

def create_zoomable_img(img, title=None):
    """Create a zoomable plotly figure for a single image"""
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    
    # If float 0-1, convert to 0-255 uint8
    if img.dtype.kind == 'f' and img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
        
    # Handle grayscale (2D) images by converting to RGB
    if len(img.shape) == 2:
        img = np.stack((img,)*3, axis=-1)
        
    fig = go.Figure()
    fig.add_trace(go.Image(z=img, hoverinfo='skip'))
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        dragmode='zoom',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        autosize=True
    )
    return fig

def batch_analysis_page(models):
    """Batch analysis page"""
    st.header("Batch Analysis")
    
    if models is None:
        st.warning("Please load a model from the sidebar first.")
        return

    uploaded_files = st.file_uploader(
        "Upload multiple images for batch analysis",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        col1, col2 = st.columns(2)
        with col1:
            threshold = st.slider("Anomaly Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
        with col2:
            batch_size = st.slider("Batch Size", min_value=1, max_value=32, value=8, step=1)
        
        if st.button("Analyze Batch"):
            # Convert uploaded files to image arrays
            image_arrays = []
            filenames = []
            
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                image_arrays.append(image_array)
                filenames.append(uploaded_file.name)
            
            # Process in batches
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text(f'Initializing... Processing {len(uploaded_files)} images in batches of {batch_size}...')
            
            results = predict_batch_images(models, image_arrays, batch_size=batch_size, heatmap_threshold=threshold, 
                                         progress_bar=progress_bar, status_text=status_text)
            
            # Add filenames to results
            for i, result in enumerate(results):
                result['filename'] = filenames[i]
                result['confidence'] = abs(result['anomaly_score'])
            
            progress_bar.progress(1.0)
            status_text.text(f'Analysis complete! Processed {len(results)} images.')
            
            # Display results
            df = pd.DataFrame(results)
            
            # Summary statistics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Images", len(results))
            with col2:
                anomaly_count = sum(r['is_anomaly'] for r in results)
                st.metric("Anomalies Found", anomaly_count)
            with col3:
                normal_count = len(results) - anomaly_count
                st.metric("Normal Images", normal_count)
            with col4:
                anomaly_rate = anomaly_count / len(results) * 100 if results else 0
                st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
            with col5:
                avg_latency = df['inference_time'].mean() * 1000  # Convert to ms
                st.metric("Avg Latency", f"{avg_latency:.2f} ms")
            
            # Results table
            st.subheader("Detailed Results")
            # Format latency column for display
            df_display = df.copy()
            df_display['inference_time_ms'] = df_display['inference_time'].apply(lambda x: f"{x*1000:.2f} ms")
            df_display = df_display[['filename', 'anomaly_score', 'is_anomaly', 'confidence', 'inference_time_ms']]
            st.dataframe(df_display)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Score Distribution")
                fig = px.histogram(df, x='anomaly_score', color='is_anomaly',
                                 title='Anomaly Score Distribution',
                                 labels={'anomaly_score': 'Anomaly Score', 'count': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Latency Distribution")
                fig = px.histogram(df, x='inference_time', 
                                 title='Inference Time Distribution',
                                 labels={'inference_time': 'Inference Time (seconds)', 'count': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="anomaly_detection_results.csv",
                mime="text/csv"
            )

def live_monitor_page(models):
    """Live Folder Monitor page"""
    st.header("Live Folder Monitor")
    
    if models is None:
        st.warning("Please load a model from the sidebar first.")
        return

    st.markdown("Monitors a folder for new images, automatically crops them to `(751, 0, 3072, 2048)`, and performs anomaly detection.")

    # Monitor configuration
    col1, col2 = st.columns([3, 1])
    with col1:
        # Default to current directory or previously used
        default_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
        if not os.path.exists(default_dir):
            default_dir = os.getcwd()
            
        input_dir = st.text_input("Folder path to monitor:", value=default_dir)
    with col2:
        threshold = st.number_input("Anomaly Threshold", 0.0, 1.0, 0.6, step=0.01)

    # State init
    if 'monitoring' not in st.session_state:
        st.session_state.monitoring = False
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    if 'monitor_start_time' not in st.session_state:
        st.session_state.monitor_start_time = time.time()

    # Controls
    col_start, col_stop = st.columns([1, 1])
    with col_start:
        start_disabled = st.session_state.monitoring
        if st.button("Start Monitoring", type="primary", disabled=start_disabled):
            if not os.path.exists(input_dir):
                st.error(f"Directory not found: {input_dir}")
            else:
                st.session_state.monitoring = True
                st.session_state.monitor_start_time = time.time()
                
                # Pre-populate with existing files so we don't process them
                current_files = []
                image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
                for ext in image_extensions:
                    current_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
                    current_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
                
                st.session_state.processed_files = set(current_files)
                st.rerun()
    
    with col_stop:
        stop_disabled = not st.session_state.monitoring
        if st.button("Stop Monitoring", type="secondary", disabled=stop_disabled):
            st.session_state.monitoring = False
            st.rerun()

    st.divider()

    # Layout for results (Persistent containers)
    status_container = st.empty()
    metrics_container = st.container()
    
    # Grid for images - Create the layout once
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    # Placeholders
    with row1_col1:
        st.markdown("**Original Crop**")
        img_place_1 = st.empty()
    with row1_col2:
        st.markdown("**Anomaly Mask**")
        img_place_2 = st.empty()
    with row2_col1:
        st.markdown("**Reconstruction**")
        img_place_3 = st.empty()
    with row2_col2:
        st.markdown("**Heat Map**")
        img_place_4 = st.empty()

    # Monitoring Loop
    if st.session_state.monitoring:
        status_container.info(f"Monitoring: {input_dir}...")
        
        # Crop box from requirements
        CROP_BOX = (751, 0, 3072, 2048)
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        
        try:
             # Scan loop
            while st.session_state.monitoring:
                # Check for stop signal from UI interaction (although difficult in loop without rerun)
                # We rely on the rerun from stop button to break this, but a small sleep helps responsiveness
                
                # Find new files
                candidate_files = []
                for ext in image_extensions:
                    candidate_files.extend(glob.glob(os         .path.join(input_dir, f"*{ext}")))
                    candidate_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
                
                new_files = []
                for fpath in candidate_files:
                    if fpath not in st.session_state.processed_files:
                        # Found a new file!
                        new_files.append((fpath, os.path.getmtime(fpath)))
                
                # Sort by time to process oldest new file first
                new_files.sort(key=lambda x: x[1])
                
                if new_files:
                    target_file, _ = new_files[0]
                    status_container.warning(f"Processing: {os.path.basename(target_file)}...")
                    first_size = -1
                    while True:
                        curr_size = os.path.getsize(target_file)
                        if curr_size == first_size and curr_size > 0:
                            break
                        first_size = curr_size
                        time.sleep(0.05)
                    try:
                        # Process
                        img = Image.open(target_file)
                        cropped_img = img.crop(CROP_BOX)
                        image_array = np.array(cropped_img)
                        
                        # Handle channels
                        if image_array.ndim == 2:
                             image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
                        elif image_array.shape[2] == 4:
                             image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
                        
                        # Predict
                        result = predict_single_image(models, image_array, heatmap_threshold=threshold)
                        
                        # Update Metrics
                        is_anomaly = result['is_anomaly']
                        status_text = "ANOMALY" if is_anomaly else "NORMAL"
                        color = "red" if is_anomaly else "green"
                        
                        # Clear previous metrics and write new ones
                        with metrics_container:
                            # We need to recreate columns inside the container to overwrite cleanly
                            mc1, mc2, mc3, mc4 = st.columns(4)
                            mc1.metric("Filename", os.path.basename(target_file))
                            mc2.metric("Score", f"{result['anomaly_score']:.4f}")
                            mc3.markdown(f"<h3 style='color: {color}; margin:0;'>{status_text}</h3>", unsafe_allow_html=True)
                            mc4.metric("Latency", f"{result.get('inference_time', 0)*1000:.0f} ms")

                        # Update Images - reusing create_zoomable_img
                        img_place_1.plotly_chart(create_zoomable_img(image_array), use_container_width=True)
                        
                        if 'anomaly_mask' in result:
                            img_place_2.plotly_chart(create_zoomable_img(result['anomaly_mask']), use_container_width=True)
                        if 'reconstructed_image' in result:
                            img_place_3.plotly_chart(create_zoomable_img(result['reconstructed_image']), use_container_width=True)
                        if 'heatmap_overlay' in result:
                            img_place_4.plotly_chart(create_zoomable_img(result['heatmap_overlay']), use_container_width=True)
                        
                        # Mark as processed
                        st.session_state.processed_files.add(target_file)
                        
                        status_container.success(f"Processed: {os.path.basename(target_file)}")
                        
                    except Exception as e:
                         status_container.error(f"Error processing {os.path.basename(target_file)}: {e}")
                         # Mark as processed so we don't get stuck on one bad file
                         st.session_state.processed_files.add(target_file) 
                
                else:
                    status_container.info(f"Monitoring {input_dir}... Waiting for new images...")
                    time.sleep(0.05)
                
        except Exception as e:
            st.error(f"Monitoring Loop Error: {e}")
            st.session_state.monitoring = False

def main():
    """Main application"""
    # Title
    st.markdown("<h1 class='main-header'>Diffusion Anomaly Detection Demo</h1>", unsafe_allow_html=True)
    
    # Initialize session state for models
    if 'models' not in st.session_state:
        st.session_state.models = None
    
    # Sidebar: Model Selection
    st.sidebar.title("Model Configuration")
    
    input_method = st.sidebar.radio("Model Source:", ["Select from list", "Upload file"])
    
    selected_model_path = None
    
    if input_method == "Select from list":
        available_models = get_available_models()
        if not available_models:
            st.sidebar.warning("No models found in 'outputs/' directory.")
        else:
            selected_model_path = st.sidebar.selectbox(
                "Choose a model (.pt):", 
                available_models,
                index=0
            )
    else:
        uploaded_model = st.sidebar.file_uploader("Upload .pt checkpoint", type=['pt'])
        if uploaded_model is not None:
            # Save uploaded file to a temporary location because load_checkpoint expects a path
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, uploaded_model.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_model.getbuffer())
            selected_model_path = temp_path
            st.sidebar.info(f"Uploaded: {uploaded_model.name}")

    # Device selection
    available_devices = ["auto", "cpu"]
    if torch.cuda.is_available():
        available_devices.insert(1, "cuda")
    
    device_preference = st.sidebar.selectbox(
        "Compute Device:",
        available_devices,
        index=0,
        help="Select 'cuda' for GPU acceleration or 'cpu' if the GPU is incompatible."
    )

    # Load Model Button
    if selected_model_path:
        if st.sidebar.button("Load Model", type="primary"):
            with st.spinner(f"Loading model from {os.path.basename(selected_model_path)}..."):
                st.session_state.models = load_models(selected_model_path, device_preference)
            
            if st.session_state.models:
                st.sidebar.success("Model loaded successfully!")
    # Threshold slider
    threshold = st.sidebar.slider("Anomaly Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
    
    # Display current model info
    if st.session_state.models:
        models = st.session_state.models
        st.sidebar.markdown("---")
        st.sidebar.subheader("Current Model Info")
        st.sidebar.text(f"Device: {models['device']}")
        st.sidebar.text(f"Image Size: {models['args']['img_size']}")
        st.sidebar.text(f"Steps (T): {models['args']['T']}")
        st.sidebar.text(f"File: {os.path.basename(models['model_path'])}")
    else:
        st.warning("No model loaded. Please select and load a model from the sidebar to proceed.")

    # Main Page Content
    st.sidebar.markdown("---")
    page = st.sidebar.selectbox("Select Page", ["Single Image Analysis", "Batch Analysis", "Live Folder Monitor"])
    
    if page == "Single Image Analysis":
        if not st.session_state.models:
            st.info("Please load a model from the sidebar to start analysis.")
            st.stop()
        
        models = st.session_state.models
        
        # Main content
        st.header("Single Image Analysis")
        
        
        
        # Image upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload an image for anomaly detection",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            # Layout for 2x2 grid
            # Create placeholders for the images
            st.subheader("Analysis Results")
            
            # Create metrics container (to be filled after inference)
            metrics_container = st.container()
            
            # Create 2x2 grid for images
            row1_col1, row1_col2 = st.columns(2)
            row2_col1, row2_col2 = st.columns(2)
            
            # Display Input Image immediately in Top-Left
            with row1_col1:
                st.markdown("**Input Image**")
                # Use Plotly for zoom
                st.plotly_chart(create_zoomable_img(image), use_container_width=True, config={'displayModeBar': True})
            
            # Placeholders for other images (to keep layout stable or just empty)
            with row1_col2:
                st.markdown("**Anomaly Mask**") # Moved to Top-Right
                mask_placeholder = st.empty()
                mask_placeholder.info("Waiting for inference...")
                
            with row2_col1:
                st.markdown("**Reconstruction**") # Moved to Bottom-Left
                recon_placeholder = st.empty()
                recon_placeholder.info("Waiting for inference...")
                
            with row2_col2:
                st.markdown("**Heat Map**")
                heatmap_placeholder = st.empty()
                heatmap_placeholder.info("Waiting for inference...")

            # Prediction
            # Revert to standard button since run_btn is not defined in this scope
            if st.button("Detect Anomaly", type="primary"):
                # Convert to numpy array for inference
                image_array = np.array(image)
                
                with st.spinner("Analyzing..."):
                    result = predict_single_image(models, image_array, heatmap_threshold=threshold)
                    
                    # Update Metrics in the container above
                    with metrics_container:
                        # Use tighter columns for metrics
                        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                        with m_col1:
                            st.metric("Score", f"{result['anomaly_score']:.4f}")
                        with m_col2:
                            is_anomaly = result['is_anomaly']
                            status = "ANOMALY" if is_anomaly else "NORMAL"
                            color = "red" if is_anomaly else "green"
                            st.markdown(f"<h3 style='color: {color}; margin: 0; padding: 0;'>{status}</h3>", 
                                       unsafe_allow_html=True)
                        with m_col3:
                            confidence = abs(result['anomaly_score'])
                            st.metric("Conf.", f"{confidence:.4f}")
                        with m_col4:
                            latency = result.get('inference_time', 0) * 1000
                            st.metric("Time", f"{latency:.0f} ms")

                    # Update Images in the grid with Zoomable Plotly Charts
                    
                    if 'anomaly_mask' in result:
                        mask_placeholder.plotly_chart(
                            create_zoomable_img(result['anomaly_mask']), 
                            use_container_width=True, 
                            config={'displayModeBar': True}
                        )
                        
                    if 'reconstructed_image' in result:
                        recon_placeholder.plotly_chart(
                            create_zoomable_img(result['reconstructed_image']), 
                            use_container_width=True, 
                            config={'displayModeBar': True}
                        )

                    if 'heatmap_overlay' in result:
                        heatmap_placeholder.plotly_chart(
                            create_zoomable_img(result['heatmap_overlay']), 
                            use_container_width=True, 
                            config={'displayModeBar': True}
                        )
                        
                    
    
    elif page == "Batch Analysis":
        batch_analysis_page(st.session_state.models)
        
    elif page == "Live Folder Monitor":
        live_monitor_page(st.session_state.models)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Diffusion Anomaly Detection Demo | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()