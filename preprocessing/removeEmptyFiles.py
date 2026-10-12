#Removing empty label files and their corresponding images
import os

def remove_empty_labels_and_images(images_dir, labels_dir):
    removed_files = []
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            label_path = os.path.join(labels_dir, label_file)
            if os.path.getsize(label_path) == 0:  # empty label file
                os.remove(label_path)  # remove label file
                image_name = os.path.splitext(label_file)[0]
                for ext in ['.jpg', '.jpeg', '.png']:
                    image_path = os.path.join(images_dir, image_name + ext)
                    if os.path.exists(image_path):
                        os.remove(image_path)  # remove corresponding image file
                        removed_files.append((image_path, label_path))
                        break
    return removed_files

# Set the correct absolute or relative directory paths
images_train = '/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/images/train'
labels_train = '/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/labels/train'
images_test = '/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/images/test'
labels_test = '/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/labels/test'
images_val = '/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/images/val'
labels_val = '/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/labels/val'

removed = remove_empty_labels_and_images(images_train, labels_train)
print(f"Removed {len(removed)} image-label pairs with empty labels in train split.")
for img, lbl in removed:
    print(f"Removed image: {img}, label: {lbl}")

removed = remove_empty_labels_and_images(images_test, labels_test)
print(f"Removed {len(removed)} image-label pairs with empty labels in test split.")
for img, lbl in removed:
    print(f"Removed image: {img}, label: {lbl}")

removed = remove_empty_labels_and_images(images_val, labels_val)
print(f"Removed {len(removed)} image-label pairs with empty labels in val split.")
for img, lbl in removed:
    print(f"Removed image: {img}, label: {lbl}")
