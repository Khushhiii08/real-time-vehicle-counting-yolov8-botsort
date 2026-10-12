#Checking for invalid bounding box coordinates and class IDs in YOLO format label files
import os

def check_box_coords(labels_dir):
    issues = []
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, x_c, y_c, w, h = parts
                        try:
                            x_c, y_c, w, h = map(float, [x_c, y_c, w, h])
                            if not (0 <= x_c <= 1 and 0 <= y_c <= 1 and w > 0 and h > 0 and w <= 1 and h <= 1):
                                issues.append((label_file, i + 1, line.strip()))
                        except:
                            issues.append((label_file, i + 1, line.strip()))
    return issues

# Usage example
issues = check_box_coords('/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/labels')
if issues:
    print('Bounding box format issues found:')
    for issue in issues:
        print(issue)
else:
    print('All bounding boxes appear valid.')

def find_invalid_class_ids(labels_dir, expected_classes):
    invalid = []
    for file in os.listdir(labels_dir):
        if file.endswith('.txt'):
            path = os.path.join(labels_dir, file)
            with open(path) as f:
                for i, line in enumerate(f):
                    cls_id = line.strip().split(' ')[0]
                    try:
                        cls_id = int(float(cls_id))
                        if cls_id not in expected_classes:
                            invalid.append((file, i+1, cls_id))
                    except:
                        invalid.append((file, i+1, 'parse error'))
    return invalid

labels_path = '/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/labels'
invalid_classes = find_invalid_class_ids(labels_path, set(range(15)))
print(invalid_classes)

