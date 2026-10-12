#Converting XML annotations to YOLO format and visualizing them

from xml.dom import minidom
import os
import glob
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# LUT dictionary for class to class_id mapping
lut = {
    'animal': 0,
    'autorickshaw': 1,
    'bicycle': 2,
    'bus': 3,
    'car': 4,
    'caravan': 5,
    'motorcycle': 6,
    'person': 7,
    'rider': 8,
    'traffic light': 9,
    'traffic sign': 10,
    'trailer': 11,
    'train': 12,
    'truck': 13,
    'vehicle fallback': 14
}

def convert_coordinates(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_xml2yolo(lut, xml_folder):
    xml_files = glob.glob(os.path.join(xml_folder, '*.xml'))
    
    for fname in xml_files:
        xmldoc = minidom.parse(fname)
        # Save .txt in same folder as .xml
        fname_out = fname[:-4] + '.txt'
        with open(fname_out, "w") as f:
            itemlist = xmldoc.getElementsByTagName('object')
            size = xmldoc.getElementsByTagName('size')[0]
            width = int(size.getElementsByTagName('width')[0].firstChild.data)
            height = int(size.getElementsByTagName('height')[0].firstChild.data)

            for item in itemlist:
                classid = item.getElementsByTagName('name')[0].firstChild.data
                if classid in lut:
                    label_str = str(lut[classid])
                else:
                    label_str = "-1"
                    print("warning: label '%s' not in look-up table" % classid)

                xmin = float(item.getElementsByTagName('xmin')[0].firstChild.data)
                ymin = float(item.getElementsByTagName('ymin')[0].firstChild.data)
                xmax = float(item.getElementsByTagName('xmax')[0].firstChild.data)
                ymax = float(item.getElementsByTagName('ymax')[0].firstChild.data)

                b = (xmin, ymin, xmax, ymax)
                bb = convert_coordinates((width, height), b)
                f.write(label_str + " " + " ".join([f"{a:.6f}" for a in bb]) + '\n')

        print(f"wrote {fname_out}")

def display_yolo_annotations(image_folder, yolo_folder, lut, sample_count=3):
    yolo_files = glob.glob(os.path.join(yolo_folder, '*.txt'))
    os.makedirs(yolo_folder, exist_ok=True)
    sampled_files = random.sample(yolo_files, min(sample_count, len(yolo_files)))

    # Building reverse LUT for label name lookup
    rev_lut = {v: k for k, v in lut.items()}

    for yolo_file in sampled_files:
        img_file = os.path.basename(yolo_file)[:-4] + '.jpg'  # assuming jpg images
        img_path = os.path.join(image_folder, img_file)

        if not os.path.exists(img_path):
            print(f"Image file missing for label: {img_file}")
            continue

        image = Image.open(img_path)
        width, height = image.size

        fig, ax = plt.subplots(1)
        ax.imshow(image)

        with open(yolo_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                x_center = float(parts[1]) * width
                y_center = float(parts[2]) * height
                w = float(parts[3]) * width
                h = float(parts[4]) * height

                xmin = x_center - w / 2
                ymin = y_center - h / 2

                rect = patches.Rectangle((xmin, ymin), w, h, linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect)
                label = rev_lut.get(class_id, 'Unknown')
                ax.text(xmin, ymin - 5, label, color='red', fontsize=8)

        plt.title(f"YOLO annotation preview: {img_file}")
        plt.axis('off')
        plt.show()

def main():
    xml_folder = "/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/labels"
    image_folder = "/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/images"
    
    convert_xml2yolo(lut, xml_folder)
    display_yolo_annotations(image_folder, xml_folder, lut, sample_count=3)

if __name__ == '__main__':
    main()
