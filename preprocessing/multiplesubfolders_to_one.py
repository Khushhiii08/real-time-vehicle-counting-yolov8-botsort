#Walk through multiple subfolders and copy all files to a single folder
import os
import shutil

base_dir = '/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo'

annotations_dir = os.path.join(base_dir, 'Annotations')
jpegimages_dir = os.path.join(base_dir, 'JPEGImages')

labels_dir = os.path.join(base_dir, 'labels')
images_dir = os.path.join(base_dir, 'images')

os.makedirs(labels_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

for subdir, dirs, files in os.walk(annotations_dir):
    for file in files:
        if file.endswith('.xml'):
            src_path = os.path.join(subdir, file)
            dst_path = os.path.join(labels_dir, file)
            shutil.copy2(src_path, dst_path)

for subdir, dirs, files in os.walk(jpegimages_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg')):
            src_path = os.path.join(subdir, file)
            dst_path = os.path.join(images_dir, file)
            shutil.copy2(src_path, dst_path)

print('Copying completed successfully.')