# We are analyzing the dataset to understand its characteristics better.
# This includes checking for missing files, empty labels, class distributions, 
# bounding box statistics, and visualizations. This helps ensure data quality 
# and informs training strategies.

import os
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import numpy as np

# Define BASE_DIR relative to the script's location for consistency
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Path Configuration ---
# NOTE: The paths defined in app.py are now passed directly to the function.
# We remove the hardcoded local paths here to make the script portable.
# labels_base = '/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/labels' <- REMOVED
# images_base = '/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/images' <- REMOVED

# ---- EDA FUNCTION ----
def analyze_split(split, labels_base, images_base, class_names, num_classes):
    """
    Performs a detailed exploratory data analysis (EDA) on a dataset split,
    displaying key statistics and visualizations.
    
    Args:
        split (str): The name of the dataset split ('train', 'val', or 'test').
        labels_base (Path): The base Path object for label files (e.g., .../IDD_Detection_yolo/labels).
        images_base (Path): The base Path object for image files (e.g., .../IDD_Detection_yolo/images).
        class_names (list): A list of class names.
        num_classes (int): The total number of classes in the dataset.
    """
    st.markdown(f"## 📊 EDA for the `{split.capitalize()}` Split", unsafe_allow_html=True)
    st.markdown("---")

    # Construct the split-specific folder paths
    label_folder = labels_base / split
    image_folder = images_base / split

    # Use a try-except block to handle cases where folders don't exist
    try:
        # We look for all files recursively, but primarily focus on the top level
        label_files = list(label_folder.rglob('*.txt'))
        # Using rglob to catch files in potential subdirectories, and filtering by extension
        image_files = [p for p in image_folder.rglob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    except Exception:
        st.error(f"❌ Error: Could not access the `{split}` folders. Please check your `BASE_DIR` configuration in `app.py`.")
        return

    st.markdown(f"### 📁 Data Integrity Check")
    st.info(f"**Total images found:** `{len(image_files)}`")
    st.info(f"**Total label files found:** `{len(label_files)}`")

    # Match images and labels by stem (filename without extension)
    image_names = {f.stem for f in image_files}
    label_names = {f.stem for f in label_files}

    missing_images = label_names - image_names
    missing_labels = image_names - label_names

    if missing_labels:
        st.warning(f"⚠️ **{len(missing_labels)} images have no corresponding labels.** They will be excluded from analysis.")
    if missing_images:
        st.warning(f"⚠️ **{len(missing_images)} labels have no corresponding images.** They will be excluded from analysis.")
    
    st.markdown("---")

    # Data collection for analysis
    class_counts = Counter()
    empty_label_files_count = 0
    bbox_widths, bbox_heights = [], []
    bbox_centers_x, bbox_centers_y = [], []
    image_widths, image_heights = [], []
    labels_per_image = []

    label_files_to_process = [f for f in label_files if f.stem in image_names]
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, label_file in enumerate(label_files_to_process):
        base_name = label_file.stem
        
        # Find the corresponding image
        image_path = next((p for p in image_folder.rglob(f"{base_name}.*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']), None)

        if not image_path:
            continue  # Should not happen if filtered, but safe check

        # Process image dimensions
        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
                image_widths.append(img_w)
                image_heights.append(img_h)
        except Exception as e:
            st.error(f"❌ Could not read image `{image_path}`. Error: {e}")
            continue

        # Process labels
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                labels_per_image.append(len(lines))

            if not lines:
                empty_label_files_count += 1
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    # Use index 0 for class ID to avoid float conversion issues on the first element
                    cls_id = int(parts[0]) 
                    class_counts[cls_id] += 1
                    bbox_centers_x.append(float(parts[1]))
                    bbox_centers_y.append(float(parts[2]))
                    bbox_widths.append(float(parts[3]))
                    bbox_heights.append(float(parts[4]))

        except Exception as e:
            st.error(f"❌ Error parsing label file `{label_file}`. Error: {e}")
            continue

        # Update progress bar
        if len(label_files_to_process) > 0:
            progress = (i + 1) / len(label_files_to_process)
            progress_bar.progress(progress)
            status_text.text(f"Processing... {int(progress * 100)}%")

    progress_bar.empty()
    status_text.empty()
    
    st.markdown("---")
    st.markdown("### 📈 Dataset Statistics")

    st.write(f"**Objects with no annotations:** `{empty_label_files_count}` images")
    st.write("**Object Distribution by Class:**")
    
    # Display class counts as a bar chart for better visualization
    if class_counts:
        # Sort counts by class ID for consistent plotting
        # Ensure we only plot classes that are present and mapped to names
        present_classes = sorted([cid for cid, count in class_counts.items() if count > 0])
        counts = [class_counts[cid] for cid in present_classes]
        
        # Use class names for labels, ensuring the ID maps correctly
        class_labels = [class_names[cid] if cid < len(class_names) else f"Class {cid}" for cid in present_classes]
        
        fig_classes, ax_classes = plt.subplots(figsize=(10, 5))
        ax_classes.bar(range(len(present_classes)), counts, color='teal')
        ax_classes.set_xticks(range(len(present_classes)))
        ax_classes.set_xticklabels(class_labels, rotation=45, ha='right')
        ax_classes.set_ylabel('Number of Instances')
        ax_classes.set_title('Object Counts per Class')
        st.pyplot(fig_classes)
    else:
        st.warning("No objects found in this split for analysis.")

    st.markdown("---")
    st.markdown("### 🖼️ Image and Bounding Box Analysis")
    
    if image_widths:
        # Display Max/Min/Median/Average dimensions for better insight
        avg_w = np.mean(image_widths)
        avg_h = np.mean(image_heights)
        
        st.info(f"**Image Dimensions:** Max W: `{max(image_widths)}` px, Max H: `{max(image_heights)}` px. Avg W: `{avg_w:.0f}` px, Avg H: `{avg_h:.0f}` px.")
    
    if bbox_centers_x:
        # Create a visually appealing two-column layout for the plots
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### **Bounding Box Center Distribution**")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist([bbox_centers_x, bbox_centers_y], bins=30, label=['Center X', 'Center Y'], alpha=0.7)
            ax.set_xlabel('Normalized Coordinate (0 to 1)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Bounding Box Centers')
            ax.legend()
            st.pyplot(fig)
            st.markdown("This plot reveals if objects are biased towards specific regions of the images.")

        with col2:
            st.markdown("#### **Bounding Box Size Distribution**")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist([bbox_widths, bbox_heights], bins=30, label=['Width', 'Height'], alpha=0.7)
            ax.set_xlabel('Normalized Size (0 to 1)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Bounding Box Sizes')
            ax.legend()
            st.pyplot(fig)
            st.markdown("This plot helps understand the typical scale of objects in the dataset (small objects dominate if bars are near 0).")
            
    st.markdown("---")
    st.markdown("### 📦 Object Density Analysis")

    if labels_per_image:
        fig_density, ax_density = plt.subplots(figsize=(8, 5))
        
        # Calculate max+1 for histogram range for nice binning
        max_labels = max(labels_per_image) if labels_per_image else 0
        bins = range(max_labels + 2) 
        
        ax_density.hist(labels_per_image, bins=bins, align='left', edgecolor='black', color='dodgerblue')
        ax_density.set_xticks(range(max_labels + 1))
        ax_density.set_xlabel('Number of Objects per Image')
        ax_density.set_ylabel('Number of Images')
        ax_density.set_title('Object Density per Image')
        st.pyplot(fig_density)
        st.markdown("This histogram shows how many objects are typically present in each image, indicating dataset sparsity or density.")
        
    st.markdown("---")
    st.success("✅ **EDA Complete!**")
