#moves all the files in the main images and labels folder to a new folder called 'og' inside the labels and images folders to keep them safe while training
#we can use them in the future if we come across any issues in the preprocessed data
import os
import shutil

images_dir = '/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/images'
labels_dir = '/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/labels'
og_labels_dir = os.path.join(labels_dir, 'oglabels')
og_images_dir = os.path.join(images_dir, 'ogimages')

# Create 'og' folder if it doesn't exist
os.makedirs(og_images_dir, exist_ok=True)
os.makedirs(og_labels_dir, exist_ok=True)
for item in os.listdir(labels_dir):
    item_path = os.path.join(labels_dir, item)
    
    if item in ['train', 'test', 'val', 'og']:
        continue

    if os.path.isfile(item_path):
        dest_path = os.path.join(og_labels_dir, item)
        # Remove if file exists in destination to avoid error
        if os.path.exists(dest_path):
            os.remove(dest_path)
        shutil.move(item_path, og_labels_dir)
        print(f'Moved {item} to og folder')

for item in os.listdir(images_dir):
    item_path = os.path.join(images_dir, item)
    
    if item in ['train', 'test', 'val', 'og']:
        continue

    if os.path.isfile(item_path):
        dest_path = os.path.join(og_images_dir, item)
        # Remove if file exists in destination to avoid error
        if os.path.exists(dest_path):
            os.remove(dest_path)
        shutil.move(item_path, og_images_dir)
        print(f'Moved {item} to og folder')

print('Moving complete')
