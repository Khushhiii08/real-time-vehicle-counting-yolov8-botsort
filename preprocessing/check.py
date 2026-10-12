#Checking if all images and labels in the train, test and val splits are present in the respective folders
import os
import re

def read_digit_list(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    digit_list = [re.search(r'(\d+)$', line.strip()).group(1) for line in lines if re.search(r'(\d+)$', line.strip())]
    return set(digit_list)  # use a set for efficient lookup

def validate_split(txt_file, images_dir, labels_dir):
    missing_images = []
    missing_labels = []
    
    digits = read_digit_list(txt_file)
    print(f"Verified digits from file: {digits}")
    
    # Check images and labels for only those with digit names in the digit list
    all_img_files = [f for f in os.listdir(images_dir) if any(f.endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])]
    
    for img_file in all_img_files:
        img_digit = re.search(r'(\d+)', img_file)
        if img_digit and img_digit.group(1) in digits:
            full_img_path = os.path.join(images_dir, img_file)
            label_file = os.path.splitext(img_file)[0] + '.txt'
            full_label_path = os.path.join(labels_dir, label_file)
            if not os.path.exists(full_img_path):
                missing_images.append(full_img_path)
            if not os.path.exists(full_label_path):
                missing_labels.append(full_label_path)
    
    return missing_images, missing_labels

train_txt = '/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/train.txt'
images_dir = '/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/images'
labels_dir = '/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/labels'

miss_img, miss_lbl = validate_split(train_txt, images_dir, labels_dir)
print(f"Missing images: {miss_img}")
print(f"Missing labels: {miss_lbl}")
