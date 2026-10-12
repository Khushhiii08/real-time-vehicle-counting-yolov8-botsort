# app.py
# =============================================================================
# TRAFFIC INTELLIGENCE DASHBOARD
# This Streamlit app provides:
# 1. Dataset statistics and exploration
# 2. Model training visualizations
# 3. Model evaluation metrics
# 4. Live vehicle detection and counting
# =============================================================================

import streamlit as st  # Web app framework
import tempfile  # For creating temporary files
import os  # Operating system operations
import cv2  # OpenCV for image/video processing
import numpy as np  # NumPy for numerical operations
import time  # For timing operations
from pathlib import Path  # Better path handling
from ultralytics import YOLO  # YOLOv8 model
from typing import Iterable  # Type hints
from collections import Counter  # For counting class occurrences

# Import custom modules for dataset analysis and evaluation
from preprocessing.dataseteda import analyze_split
from evaluation import (
    run_yolov8_filtered_evaluation,
    visualize_yolov8_eval,
    visualize_rf_detr_eval,
    YOLO_DATA_YAML_PATH,
    run_yolov12_filtered_evaluation,
    visualize_yolov12_eval,
    RF_DETR_RESULTS_JSON_PATH,
    class_names  # List of vehicle classes: ['car', 'truck', 'bus', etc.]
)
from virtual_line import line_counting_botsort_ultralytics  # Our counting function

# =============================================================================
# CONFIGURATION: Define all important paths
# =============================================================================

# Get project base directory from environment variable or use current directory
BASE_DIR = Path(os.getenv("VEHICLE_COUNTING_PROJECT", Path.cwd()))

# Model paths
YOLOv8_MODEL_PATH = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"  # Trained YOLOv8 model
YOLOv8_PLOTS_PATH = BASE_DIR / "runs" / "detect" / "train" / "results.png"  # YOLOv8 training plots
YOLOv12_PLOTS_PATH = BASE_DIR / "runs" / "detect" / "yolo12m_optimized" / "results.png"  # YOLOv8 training plots
RF_DETR_MODEL_PATH = BASE_DIR / "rf-detr-base.pth"  # RF-DETR model 
RF_DETR_PLOTS_PATH = BASE_DIR / "rfdetr_optimized" / "metrics_plot.png"  # RF-DETR training plots
YOLOv12_MODEL_PATH = BASE_DIR / "runs" / "detect" / "yolo12m_optimized" / "weights" / "best.pt"  # Trained YOLOv12 model
# Dataset paths
LABELS_BASE = BASE_DIR / "IDD_Detection_yolo" / "labels"  # Annotation files
IMAGES_BASE = BASE_DIR / "IDD_Detection_yolo" / "images"  # Image files

# Dataset splits
splits = ['train', 'val', 'test']  # Training, validation, and test sets
num_classes = len(class_names)  # Total number of vehicle classes

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_resource  # Cache the model so it's only loaded once
def load_model(model_path, model_choice):
    """
    Load the selected model (YOLOv8 or RF-DETR).
    
    Args:
        model_path: Path to the model weights file
        model_choice: String indicating which model to load ("YOLOv8" or "RF-DETR")
    
    Returns:
        Loaded model object, or None if loading fails
    
    Note: RF-DETR loading is not fully implemented and will show an error
    """
    try:
        if model_choice == "RF-DETR":
            # RF-DETR requires custom loading logic (not implemented here)
            st.error("❌ RF-DETR Loading Error: DETR models like RF-DETR (.pth) require custom loading logic.")
            # Return a dummy function that raises an error if called
            def dummy_rfdetr_model(*args, **kwargs):
                raise NotImplementedError("RF-DETR model loading failed.")
            return dummy_rfdetr_model
        
        elif model_choice == "YOLOv8":
            # Load YOLOv8 model using Ultralytics library
            return YOLO(str(model_path))
    
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

def save_uploaded_file(uploaded_file):
    """
    Save an uploaded file (from Streamlit's file_uploader) to a temporary location.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        str: Path to the saved temporary file
    
    Why? Streamlit's uploaded files are in memory. We need them on disk for OpenCV.
    """
    # Get file extension (e.g., ".mp4", ".jpg")
    suffix = f".{uploaded_file.name.split('.')[-1]}"
    
    # Create temporary file with appropriate extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())  # Write uploaded content to temp file
        temp_file_path = temp_file.name  # Get path to temp file
    
    return temp_file_path

# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================

st.set_page_config(page_title="YOLOv8 VS RF-DETR", layout="wide")  # Wide layout for better use of space
st.title("🚦 Traffic Intelligence Dashboard: YOLOv8 vs RF-DETR + BOT-SORT ")
st.markdown("**VIDEO MODE:** Line crossing with BOT-SORT tracking | **IMAGE MODE:** Count all detected objects")
st.divider()  # Visual separator

# =============================================================================
# SECTION I: DATASET SUMMARY
# =============================================================================

st.header("I. Dataset Summary (IDD Detection)")

# Create 3 columns (one for each split: train, val, test)
eda_cols = st.columns(len(splits))

# Display metrics for each split
for i, split in enumerate(splits):
    img_dir = IMAGES_BASE / split  # e.g., "images/train"
    label_dir = LABELS_BASE / split  # e.g., "labels/train"
    
    # Count number of images and labels
    n_imgs = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
    n_labels = len(list(label_dir.glob("*"))) if label_dir.exists() else 0
    
    # Display in the corresponding column
    with eda_cols[i]:
        st.metric(f"**{split.upper()} Split**", f"{n_imgs} Images | {n_labels} Labels")

# Expandable section for detailed dataset analysis
with st.expander("🔬 View Detailed Dataset Analysis", expanded=False):
    for split in splits:
        try:
            # Call custom function to analyze and visualize dataset statistics
            analyze_split(split, LABELS_BASE, IMAGES_BASE, class_names, num_classes)
        except Exception as e:
            st.warning(f"Could not run detailed EDA for {split}: {e}")

st.divider()

# =============================================================================
# SECTION II: MODEL TRAINING VISUALIZATIONS
# =============================================================================

st.header("II. Model Training Visualizations")

# Create 2 columns for side-by-side comparison
col1, col2 = st.columns(2)

# --- YOLOv8 Training Plots ---
with col1:
    st.subheader("YOLOv8 Training Plots")
    if YOLOv8_PLOTS_PATH.exists():
        # Display training metrics image (loss, mAP, precision, recall, etc.)
        st.image(str(YOLOv8_PLOTS_PATH), caption="YOLOv8 Training Metrics")
    else:
        st.warning(f"YOLOv8 training plots not found at: {YOLOv8_PLOTS_PATH}.")

# --- RF-DETR Training Plots ---
with col2:
    st.subheader("RF-DETR Training Plots")
    if RF_DETR_PLOTS_PATH.exists():
        st.image(str(RF_DETR_PLOTS_PATH), caption="RF-DETR Training Metrics")
    else:
        st.warning(f"RF-DETR training plots not found at: {RF_DETR_PLOTS_PATH}.")

# --- YOLOv12 Training Plots ---
with col1:
    st.subheader("YOLOv12 Training Plots")
    if YOLOv12_PLOTS_PATH.exists():
        # Display training metrics image (loss, mAP, precision, recall, etc.)
        st.image(str(YOLOv12_PLOTS_PATH), caption="YOLOv12 Training Metrics")
    else:
        st.warning(f"YOLOv12 training plots not found at: {YOLOv12_PLOTS_PATH}.")

st.divider()

# =============================================================================
# SECTION III: MODEL EVALUATION DASHBOARD
# =============================================================================

st.header("III. Model Evaluation Dashboard")

# --- Initialize session state for storing evaluation results ---
# Session state persists across Streamlit reruns
if 'yolo8_eval_results' not in st.session_state: 
    st.session_state.yolo8_eval_results = None
if 'rfdetr_eval_results' not in st.session_state: 
    st.session_state.rfdetr_eval_results = None
if 'yolo12_eval_results' not in st.session_state: 
    st.session_state.yolo12_eval_results = None

# --- Button to trigger evaluation ---
run_eval_button = st.button("🔴 Run Comprehensive Model Evaluations", type="primary")

if run_eval_button:
    # --- Evaluate YOLOv8 on test set ---
    with st.spinner("⏳ Running YOLOv8 Evaluation on TEST set..."):
        yolo_results = run_yolov8_filtered_evaluation(
            model_path=YOLOv8_MODEL_PATH,
            images_dir=IMAGES_BASE / "test",  # Test images
            labels_dir=LABELS_BASE / "test",  # Test labels
            data_yaml_path=YOLO_DATA_YAML_PATH,  # Dataset config
            class_names_list=class_names  # Class names
        )
        st.session_state.yolo8_eval_results = yolo_results  # Store results

    with st.spinner("⏳ Running YOLOv12 Evaluation on TEST set..."):
        yolo_results = run_yolov12_filtered_evaluation(
            model_path=YOLOv12_MODEL_PATH,
            images_dir=IMAGES_BASE / "test",  # Test images
            labels_dir=LABELS_BASE / "test",  # Test labels
            data_yaml_path=YOLO_DATA_YAML_PATH,  # Dataset config
            class_names_list=class_names  # Class names
        )
        st.session_state.yolo12_eval_results = yolo_results  # Store results
    
    # --- Check for RF-DETR results ---
    if RF_DETR_RESULTS_JSON_PATH.exists():
        st.session_state.rfdetr_eval_results = True  # Results file exists
    else:
        st.session_state.rfdetr_eval_results = {"error": f"**Results file not found:** `{RF_DETR_RESULTS_JSON_PATH.name}`"}
    
    st.success("✅ Evaluation runs complete! Results are stored below.")
    st.rerun()  # Refresh page to display results

# --- Display evaluation results side-by-side ---
col_eval1, col_eval2, col_eval3 = st.columns(3)

# YOLOv8 Evaluation Results
with col_eval1:
    with st.expander("📊 YOLOv8 Evaluation Results (mAP)", expanded=True):
        if st.session_state.yolo8_eval_results:
            visualize_yolov8_eval(st.session_state.yolo8_eval_results)
        else:
            st.info("Click 'Run Comprehensive Model Evaluations' to generate results.")

# RF-DETR Evaluation Results
with col_eval2:
    with st.expander("📊 RF-DETR Evaluation Results (mAP)", expanded=True):
        if st.session_state.rfdetr_eval_results:
            if isinstance(st.session_state.rfdetr_eval_results, dict) and 'error' in st.session_state.rfdetr_eval_results:
                st.error(st.session_state.rfdetr_eval_results['error'])
            else:
                visualize_rf_detr_eval()
        else:
            st.info("Click 'Run Comprehensive Model Evaluations' to check for results file.")

# YOLOv12 Evaluation Results
with col_eval2:
    with st.expander("📊 YOLOv12 Evaluation Results (mAP)", expanded=True):
        if st.session_state.yolo12_eval_results:
            visualize_yolov12_eval(st.session_state.yolo12_eval_results)
        else:
            st.info("Click 'Run Comprehensive Model Evaluations' to generate results.")

st.divider()

# =============================================================================
# SECTION IV: LIVE VEHICLE COUNTING AND TRACKING
# =============================================================================

st.header("IV. Live Vehicle Counting and Tracking (YOLOv8 + BOT-SORT)")

# --- Model Selection (currently fixed to YOLOv8) ---
model_choice = "YOLOv8"
selected_model_path = YOLOv8_MODEL_PATH
st.info("**YOLOv8** is recommended. VIDEO: line crossing with tracking | IMAGE: all detections counted")

# =============================================================================
# SIDEBAR SETTINGS
# =============================================================================

st.sidebar.header("⚙️ Live Tracking Settings")
st.sidebar.markdown(f"**Model Selected:** `{model_choice}`")

# --- Confidence Threshold Slider ---
# Lower values detect more objects but may include false positives
# Higher values are more selective but may miss some objects
conf = st.sidebar.slider(
    "Detection Confidence Threshold", 
    0.01,  # Minimum value
    1.0,   # Maximum value
    0.35,  # Default value
    0.01,  # Step size
    key="conf_slider"
)

# --- Counting Line Position Slider ---
# Controls where the virtual line is drawn (0% = top, 100% = bottom)
line_pct = st.sidebar.slider(
    "Virtual Line Position (% from top)", 
    0,     # Top of frame
    100,   # Bottom of frame
    90,    # Default at 90% (near bottom)
    key="line_pct_slider"
) / 100.0  # Convert to 0.0-1.0 range

# =============================================================================
# INPUT SOURCE SELECTION
# =============================================================================

# Radio button to choose input type
source_type_live = st.radio(
    "Select Input Source", 
    ["Upload Video", "Upload Image", "RTSP/HTTP URL"], 
    key="input_source_live"
)

# Initialize variables
uploaded_live = None  # Uploaded file object
src = None  # Source path/URL
input_is_image = False  # Flag: True if input is an image
temp_file_path = None  # Path to temporary file (if uploaded)

# --- UPLOAD VIDEO ---
if source_type_live == "Upload Video":
    uploaded_live = st.file_uploader(
        "Upload Video File", 
        type=['mp4', 'mov', 'avi', 'mkv'],  # Accepted formats
        key="video_upload_live"
    )
    if uploaded_live:
        temp_file_path = save_uploaded_file(uploaded_live)  # Save to temp location
        src = temp_file_path  # Use temp file as source

# --- UPLOAD IMAGE ---
elif source_type_live == "Upload Image":
    uploaded_live = st.file_uploader(
        "Upload Image File", 
        type=['jpg', 'jpeg', 'png', 'bmp'],  # Accepted formats
        key="image_upload_live"
    )
    input_is_image = True  # Set image flag
    if uploaded_live:
        temp_file_path = save_uploaded_file(uploaded_live)
        src = temp_file_path

# --- RTSP/HTTP URL ---
elif source_type_live == "RTSP/HTTP URL":
    url_val = st.text_input("Enter RTSP/HTTP URL", key="rtsp_url_live")
    if url_val.strip():  # If URL is not empty
        src = url_val.strip()

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
# Session state persists values across Streamlit reruns

if "detecting" not in st.session_state: 
    st.session_state.detecting = False  # Is detection currently running?

if "temp_file_path" not in st.session_state: 
    st.session_state.temp_file_path = None  # Path to temp file

if "cumulative_counts" not in st.session_state: 
    st.session_state.cumulative_counts = Counter()  # Running count of detected classes

if "last_source" not in st.session_state:
    st.session_state.last_source = None  # Last processed source (to detect changes)

# =============================================================================
# START/STOP BUTTONS
# =============================================================================

col_start, col_stop = st.columns(2)

# --- START BUTTON ---
start_live = col_start.button("▶️ Start Detection", type="primary", key="start_live_btn")

# --- STOP BUTTON ---
stop_live = col_stop.button("⏹️ Stop Detection", type="secondary", key="stop_live_btn")

# --- Handle START button click ---
if start_live and src is not None and (not isinstance(src, str) or src.strip() != ""):
    st.session_state.detecting = True  # Enable detection flag
    
    if temp_file_path: 
        st.session_state.temp_file_path = temp_file_path  # Store temp file path
    
    # Reset counts only if source changed (new video/image)
    if st.session_state.last_source != src:
        st.session_state.cumulative_counts = Counter()  # Reset counts
        st.session_state.last_source = src  # Update last source

elif start_live:
    # User clicked Start but no source selected
    st.warning("⚠️ Please select a valid input source before starting.")

# --- Handle STOP button click ---
if stop_live:
    st.session_state.detecting = False  # Disable detection
    
    # Clean up temporary file
    if st.session_state.temp_file_path and os.path.exists(st.session_state.temp_file_path):
        os.unlink(st.session_state.temp_file_path)  # Delete temp file
        st.session_state.temp_file_path = None

# =============================================================================
# DISPLAY PLACEHOLDERS
# =============================================================================
# These are empty containers that will be updated during detection

frame_window = st.empty()  # Placeholder for video frame display
count_placeholder = st.empty()  # Placeholder for count statistics

# =============================================================================
# CORE DETECTION LOOP
# =============================================================================

# Only run if detection is enabled AND we have a valid source
if st.session_state.detecting and src is not None and (not isinstance(src, str) or src.strip() != ""):
    
    # --- Load the model ---
    model = load_model(selected_model_path, model_choice)
    
    if model is None:
        st.error("Model failed to load. Cannot start detection.")
        st.session_state.detecting = False
    
    else:
        try:
            # --- Display mode information ---
            mode_display = "IMAGE MODE (counting all detections)" if input_is_image else "VIDEO MODE (line crossing)"
            st.info(f"🟢 Stream active. Model: **{model_choice}** | Mode: **{mode_display}**")
            
            # --- Initialize tracking variables ---
            frame_count = 0  # Count frames processed
            start_time = time.time()  # For FPS calculation
            
            # --- Create generator for processing frames ---
            counter_gen = line_counting_botsort_ultralytics(
                src=src,  # Input source
                model=model,  # YOLOv8 model
                conf_thresh=conf,  # Confidence threshold from slider
                class_names=class_names,  # List of class names
                line_pct=line_pct,  # Line position from slider
                botsort_config_path=str(BASE_DIR / "botsort.yaml"),  # BoT-SORT config
                r1_mode=True  # Enable persistent tracking IDs
            )
            
            # =================================================================
            # PROCESS FRAMES ONE BY ONE
            # =================================================================
            for annotated_frame, current_frame_counts in counter_gen:
                # annotated_frame: Frame with bounding boxes and labels drawn
                # current_frame_counts: Counter object with class counts
                
                # --- Update session state with new counts ---
                st.session_state.cumulative_counts = current_frame_counts
                
                # --- Display the annotated frame ---
                frame_window.image(
                    cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),  # Convert BGR to RGB for display
                    channels="RGB", 
                    use_container_width=True  # Scale to fit container
                )
                
                frame_count += 1  # Increment frame counter
                
                # --- Calculate total count ---
                total_count = sum(st.session_state.cumulative_counts.values())
                
                # --- Build count display text (Markdown format) ---
                if input_is_image:
                    # IMAGE MODE: Show "Detected" instead of "Counted"
                    count_md = "### 📊 Image Detection Results\n\n"
                    count_md += f"**TOTAL OBJECTS DETECTED: {total_count}**\n\n"
                else:
                    # VIDEO MODE: Show "Counted" (crossed line)
                    count_md = "### 🚦 Cumulative Vehicle Crossings\n\n"
                    count_md += f"**TOTAL VEHICLES COUNTED: {total_count}**\n\n"
                
                # --- Display per-class counts ---
                if total_count > 0:
                    # Loop through all class names
                    for cls_name in class_names:
                        count = st.session_state.cumulative_counts.get(cls_name, 0)
                        if count > 0:  # Only show classes with detections
                            count_md += f"**{cls_name}:** {count}\n\n"
                else:
                    # No detections/counts yet
                    if input_is_image:
                        count_md += "*No objects detected in this image.*\n"
                    else:
                        count_md += "*No vehicles counted yet. Waiting for line crossings...*\n"
                
                # --- Update count display ---
                count_placeholder.markdown(count_md)
                
                # --- Check stop conditions ---
                if not st.session_state.detecting:
                    break  # User clicked Stop button
                
                if input_is_image or source_type_live == "Upload Image":
                    # For images, process once then stop
                    time.sleep(2)  # Display for 2 seconds
                    break
                
                # Small delay for video processing (prevents CPU overload)
                time.sleep(0.001)
            
            # =================================================================
            # DETECTION COMPLETE
            # =================================================================
            st.session_state.detecting = False  # Disable detection flag
            
            # --- Calculate final statistics ---
            elapsed = time.time() - start_time  # Total time elapsed
            fps = frame_count / elapsed if elapsed > 0 else 0  # Frames per second
            final_total = sum(st.session_state.cumulative_counts.values())  # Final count
            
            # --- Display completion message ---
            result_msg = f"Total Detected: {final_total}" if input_is_image else f"Total Counted: {final_total}"
            st.success(
                f"🛑 Processing complete. Frames: {frame_count}, "
                f"Average FPS: {fps:.1f}, {result_msg}"
            )
        
        # =================================================================
        # ERROR HANDLING
        # =================================================================
        except RuntimeError as e:
            st.error(f"🔴 Stream Error: **{e}**. Please check your source, model path, or data files.")
            st.session_state.detecting = False
        
        except Exception as e:
            st.error(f"🔴 An unexpected error occurred: **{e}**")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")  # Show full error trace
            st.session_state.detecting = False
        
        # =================================================================
        # CLEANUP
        # =================================================================
        finally:
            # Always clean up temporary files
            if st.session_state.temp_file_path and os.path.exists(st.session_state.temp_file_path):
                try:
                    os.unlink(st.session_state.temp_file_path)  # Delete temp file
                    st.session_state.temp_file_path = None
                except Exception as e:
                    st.warning(f"Could not delete temp file: {e}")

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.markdown("Developed by Khushi Rawat - Vehicle Counting with YOLOv8 vs RF-DETR + BOT-SORT.")
