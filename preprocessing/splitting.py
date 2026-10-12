#Splitting the images and labels into train, test and val folders based on the text files provided
import os
import shutil
import re

base_dir = "/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo"
images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')

train_txt = os.path.join(base_dir, 'train.txt')
test_txt = os.path.join(base_dir, 'test.txt')
val_txt = os.path.join(base_dir, 'val.txt')

def read_digit_list(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    # Extract digits after last '/' of each line
    digit_list = [re.search(r'(\d+)$', line.strip()).group(1) for line in lines if re.search(r'(\d+)$', line.strip())]
    return digit_list

def make_dirs(base_dir):
    for split in ['train', 'test', 'val']:
        os.makedirs(os.path.join(base_dir, split), exist_ok=True)

def copy_files_by_digits(digit_list, images_src, labels_src, split):
    images_dst = os.path.join(images_dir, split)
    labels_dst = os.path.join(labels_dir, split)

    # Get all filenames in source folders
    images_files = os.listdir(images_src)
    labels_files = os.listdir(labels_src)

    for digit in digit_list:
        # Find image file whose basename ends with digit
        img_matches = [f for f in images_files if f.endswith(f'{digit}.jpg')]
        label_matches = [f for f in labels_files if f.endswith(f'{digit}.txt')]

        for img_file in img_matches:
            shutil.copy2(os.path.join(images_src, img_file), os.path.join(images_dst, img_file))
        for label_file in label_matches:
            shutil.copy2(os.path.join(labels_src, label_file), os.path.join(labels_dst, label_file))

def main():
    make_dirs(images_dir)
    make_dirs(labels_dir)

    train_digits = read_digit_list(train_txt)
    test_digits = read_digit_list(test_txt)
    val_digits = read_digit_list(val_txt)

    copy_files_by_digits(train_digits, images_dir, labels_dir, 'train')
    copy_files_by_digits(test_digits, images_dir, labels_dir, 'test')
    copy_files_by_digits(val_digits, images_dir, labels_dir, 'val')

    print('Splitting completed.')

if __name__ == "__main__":
    main()
