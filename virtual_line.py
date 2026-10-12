# virtual_line.py - Revised for clarity in image handling
import cv2
import numpy as np
from collections import Counter
import time
from typing import Optional, Iterable

# NOTE: draw_virtual_line is only used for video processing, 
# and is correctly skipped for image input in _process_single_frame_ultralytics.
def draw_virtual_line(frame: np.ndarray, line_y: int, color=(0,0,255), thickness=3) -> np.ndarray:
    """Draws horizontal counting line (VIDEO MODE ONLY)"""
    h, w = frame.shape[:2]
    cv2.line(frame, (0, line_y), (w, line_y), color, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.putText(frame, "Counting Line", (10, max(20, line_y - 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return frame

def _extract_boxes_and_ids_from_results(res) -> list:
    """Extract bounding boxes and (potential) track IDs from YOLO results"""
    out = []
    if hasattr(res, "boxes") and res.boxes is not None:
        for box in res.boxes:
            try:
                xyxy = np.array(box.xyxy).reshape(-1)
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            except Exception:
                continue
            try:
                conf = float(box.conf)
            except Exception:
                conf = 0.0
            try:
                cls = int(box.cls)
            except Exception:
                cls = -1
            
            track_id = -1
            for attr in ("id", "track_id", "box_id", "track_ids"):
                val = getattr(box, attr, None)
                if val is None:
                    continue
                try:
                    if isinstance(val, (list, tuple)) and len(val) > 0:
                        val = val = val[0]
                    track_id = int(val)
                    break
                except Exception:
                    continue
            
            out.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": conf, "cls": cls, "track_id": track_id
            })
    return out

def line_counting_botsort_ultralytics(
    src,
    model,
    conf_thresh: float,
    class_names: Iterable[str],
    line_pct: float = 0.9,
    botsort_config_path: Optional[str] = None,
    allowed_class_idxs: Optional[Iterable[int]] = None,
    r1_mode: bool = True
):
    """Vehicle counting: IMAGE=all detections, VIDEO=line crossing"""
    
    # 1. Check if src is an in-memory image (np.ndarray)
    if isinstance(src, np.ndarray):
        frame = src.copy()
        is_image_input = True

    # 2. Check if src is a path to an image file (attempt to read)
    else:
        # Attempt to open as video first
        cap = cv2.VideoCapture(src)
        if cap.isOpened() and (cap.get(cv2.CAP_PROP_FRAME_COUNT) > 1 or cap.get(cv2.CAP_PROP_FRAME_COUNT) == 0):
            # It's a valid multi-frame video stream (or live feed)
            is_image_input = False
        else:
            # If not a video, try reading as an image
            cap.release()
            frame = cv2.imread(src)
            if frame is None:
                raise RuntimeError(f"Unable to open source at: {src}")
            is_image_input = True
            
    # --- Image Processing Path (Meets the user's requirement: count and display, NO line logic) ---
    if is_image_input:
        cumulative_counts = Counter()
        # _process_single_frame_ultralytics handles the counting and display for images
        yield from _process_single_frame_ultralytics(
            frame, model, conf_thresh, class_names, line_pct, botsort_config_path,
            allowed_class_idxs, cumulative_counts, set(), {}, r1_mode, is_image_input=True
        )
        return

    # --- Video Processing Path (Uses line-crossing and tracking logic) ---
    cumulative_counts = Counter()
    counted_track_ids = set()
    track_previous_y = {}
    
    # Re-open the video capture as it was closed after the check
    cap = cv2.VideoCapture(src) 
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        yield from _process_single_frame_ultralytics(
            frame, model, conf_thresh, class_names, line_pct, botsort_config_path,
            allowed_class_idxs, cumulative_counts, counted_track_ids, track_previous_y,
            r1_mode, is_image_input=False
        )
    cap.release()

def _process_single_frame_ultralytics(
    frame, model, conf_thresh, class_names, line_pct, botsort_config_path,
    allowed_class_idxs, cumulative_counts, counted_track_ids, track_previous_y, r1_mode,
    is_image_input: bool = False
):
    """Process frame with different logic for image vs video"""
    fh, fw = frame.shape[:2]
    line_y = int(fh * line_pct)
    annotated = frame.copy()
    prev_time = time.time()

    # Run model
    try:
        if is_image_input:
            # IMAGE: Use predict (no tracking) - Complies with "NO virtual line NOR follow virtual line logic"
            results = model.predict(source=frame, conf=conf_thresh, verbose=False)
        else:
            # VIDEO: Use track with BoT-SORT (requires tracking for line crossing)
            if botsort_config_path:
                results = model.track(
                    source=frame, conf=conf_thresh, tracker=botsort_config_path,
                    persist=True, verbose=False
                )
            else:
                results = model.track(
                    source=frame, conf=conf_thresh, persist=True, verbose=False
                )
    except Exception as e:
        error_msg = str(e)
        if "tracker" in error_msg.lower() or "botsort" in error_msg.lower():
            raise RuntimeError(f"BoT-SORT error: {error_msg}") from e
        raise

    res = results[0] if isinstance(results, (list, tuple)) else results
    detections = _extract_boxes_and_ids_from_results(res)

    # ===========================================================================
    # IMAGE MODE: Count ALL detections, NO line, display on frame
    # (This section already implements the user's request)
    # ===========================================================================
    if is_image_input:
        
        # Reset counts for the single image frame to correctly reflect current detections
        cumulative_counts.clear() 
        
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            cls_id = det['cls']
            conf = det['conf']
            
            # Get class name
            if 0 <= cls_id < len(class_names):
                cls_name = class_names[cls_id]
            else:
                continue
            
            # Apply class filter
            if allowed_class_idxs is not None and cls_id not in allowed_class_idxs:
                continue
            
            # COUNT THIS DETECTION
            cumulative_counts[cls_name] += 1
            
            # Draw bounding box
            color = (255, 1, 245) 
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = f"{cls_name} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - lh - 12), (x1 + lw + 8, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # NO COUNTING LINE IS DRAWN HERE
        
        # Display total on image
        total = sum(cumulative_counts.values())
        cv2.putText(annotated, f"Total Detected: {total}", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 1, 245), 3, cv2.LINE_AA)
        
        # Display per-class counts
        y_offset = 85
        for cls_name in class_names:
            count = cumulative_counts.get(cls_name, 0)
            if count > 0:
                cv2.putText(annotated, f"{cls_name}: {count}", (15, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 1, 245), 2, cv2.LINE_AA)
                y_offset += 40
        
        yield annotated, cumulative_counts.copy()
        return # Exit IMAGE MODE

    # ===========================================================================
    # VIDEO MODE: Line crossing logic (Executed only if is_image_input is False)
    # ===========================================================================
    
    # Draw counting line for VIDEO only
    annotated = draw_virtual_line(annotated, line_y)
    
    for det in detections:
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        cls_id = det['cls']
        conf = det['conf']
        track_id = det['track_id']
        
        if 0 <= cls_id < len(class_names):
            cls_name = class_names[cls_id]
        else:
            continue
        
        if allowed_class_idxs is not None and cls_id not in allowed_class_idxs:
            continue
        
        if track_id < 0:
            continue
        
        # Color based on track ID
        color_id = track_id % 255
        color = (int(color_id), int(255 - color_id), 100)
        
        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Label
        label = f"ID:{track_id} {cls_name}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - lh - 10), (x1 + lw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Track bottom point
        center_x = int((x1 + x2) / 2)
        bottom_y = int(y2)
        cv2.circle(annotated, (center_x, bottom_y), 5, (255, 255, 0), -1)
        
        # Line crossing logic (Only for video mode)
        prev_y = track_previous_y.get(track_id)
        if prev_y is not None and track_id not in counted_track_ids:
            if prev_y < line_y and bottom_y >= line_y:
                cumulative_counts[cls_name] += 1
                counted_track_ids.add(track_id)
                cv2.putText(annotated, "COUNTED!", (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 1, 245), 2, cv2.LINE_AA)
                cv2.circle(annotated, (center_x, bottom_y), 10, (255, 1, 245), 3)
        
        track_previous_y[track_id] = bottom_y
    
    # Display FPS and total for video
    now = time.time()
    fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
    cv2.putText(annotated, f"FPS: {fps:.1f}", (fw - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
    total = sum(cumulative_counts.values())
    cv2.putText(annotated, f"Total: {total}", (fw - 150, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    
    yield annotated, cumulative_counts.copy()