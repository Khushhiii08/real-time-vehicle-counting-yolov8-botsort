#Extracting unique class names from XML label files before printing them in YAML

import os
import xml.etree.ElementTree as ET

labels_dir = '/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/labels_xml'  # adjust as needed
unique_classes = set()

for filename in os.listdir(labels_dir):
    if filename.endswith('.xml'):
        filepath = os.path.join(labels_dir, filename)
        tree = ET.parse(filepath)
        root = tree.getroot()
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            unique_classes.add(class_name)

# Sort classes alphabetically or as needed
sorted_classes = sorted(unique_classes)

# Print YAML style classes list
print("classes:")
for cls in sorted_classes:
    print(f"'{cls}'")
