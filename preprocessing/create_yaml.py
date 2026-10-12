#Creating a dataset.yaml file for YOLOv8 training
import os
base_dir = "/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_rfdetr"
yaml_file = os.path.join(base_dir, "dataset.yaml")

classes = ['animal','autorickshaw','bicycle','bus','car','caravan',
'motorcycle','person',
'rider','traffic light','traffic sign','trailer','train','truck','vehicle fallback']

with open(yaml_file, "w", encoding="utf-8") as f:
    f.write(f"path: {os.path.abspath(base_dir)}\n\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n\n")
    f.write(f"nc: {len(classes)}\n")
    f.write(f"names: {classes}\n")

print("✅ dataset.yaml created at:", yaml_file) 