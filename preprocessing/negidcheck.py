# Scan through label files to identify any negative class IDs
import os

def scan_labels_for_negative_classes(labels_dir, class_range=(0, 14)):
    issues = []
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            path = os.path.join(labels_dir, label_file)
            with open(path, 'r') as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        try:
                            cls_id = int(float(parts[0]))
                            if cls_id < class_range[0] or cls_id > class_range[1]:
                                issues.append((label_file, i+1, cls_id, line.strip()))
                        except Exception as e:
                            issues.append((label_file, i+1, 'ERROR', line.strip()))
    return issues

# Usage example:
dirs = [
    '/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/labels/train',
    '/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/labels/val',
    '/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/labels/test'
    # Add other label directories as needed
]

for d in dirs:
    if os.path.exists(d):
        issues = scan_labels_for_negative_classes(d)
        if issues:
            print(f"Issues found in directory '{d}':")
            for issue in issues:
                print(issue)
        else:
            print(f"No issues in directory '{d}'.")
