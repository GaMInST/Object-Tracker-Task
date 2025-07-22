#!/usr/bin/env python3
"""
Unified Real-time Object Tracker
- Multi-object tracking (YOLO + Deep SORT)
- User-selected single-object tracking (bounding box selection)
- Mode selection and runtime switching
"""

import cv2
import numpy as np
from ultralytics import YOLO
from Yolov5_StrongSORT_OSNet.boxmot.trackers.strongsort.strongsort import StrongSort
import argparse
import time
from pathlib import Path
import re
import torch

class ObjectSelector:
    def __init__(self):
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.selected_bbox = None
        self.selected_class_id = None
        self.selected_class_name = None
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            x1, y1 = min(self.start_point[0], self.end_point[0]), min(self.start_point[1], self.end_point[1])
            x2, y2 = max(self.start_point[0], self.end_point[0]), max(self.start_point[1], self.end_point[1])
            self.selected_bbox = [x1, y1, x2, y2]

class UnifiedObjectTracker:
    def __init__(self, model_path="yolov8s.pt", confidence_threshold=0.3):
        self.confidence_threshold = confidence_threshold
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA (GPU) is not available! Please install CUDA and a compatible PyTorch.")
        self.yolo_model = YOLO(model_path)
        self.yolo_model.to('cuda')
        self.tracker = StrongSort(
            reid_weights=Path('osnet_x0_25_msmt17.pt'),
            device=0,
            half=True,
            max_age=120,  
            n_init=1,     
            max_iou_dist=0.5, 
            max_cos_dist=0.2,  
            nn_budget=200,
        )
        try:
            self.yolo_seg_model = YOLO("yolov8s-seg.pt")
            self.yolo_seg_model.to('cuda')
        except Exception:
            self.yolo_seg_model = None
        self.selector = ObjectSelector()
        self.target_track_id = None
        self.target_bbox = None
        self.target_class_id = None
        self.target_class_name = None
        self.trails = {}
        self.max_trail_length = 30
        self.persistent_id = None
        self.persistent_features = None
        self.fps = 0.0
        self.last_known_bbox = None
        self.lost_counter = 0
        self.max_lost_frames = 60
        
    def get_tight_bbox_from_mask(self, mask):
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return [0, 0, 0, 0]
        x1, y1, x2, y2 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
        return [int(x1), int(y1), int(x2), int(y2)]

    def select_object(self, frame):
        window_name = "Select Object to Track"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.selector.mouse_callback)
        while True:
            display_frame = frame.copy()
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0, display_frame)
            if self.selector.start_point and self.selector.end_point:
                cv2.rectangle(display_frame, self.selector.start_point, self.selector.end_point, (0, 255, 0), 2)
            if self.selector.selected_bbox:
                x1, y1, x2, y2 = self.selector.selected_bbox
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(display_frame, "Selected Object", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "OBJECT SELECTION", (display_frame.shape[1]//2 - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_frame, "Draw a box around the object to track", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(display_frame, "Press ENTER to confirm, 'r' to reset, 'q' to quit, 'n' for next frame", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and self.selector.selected_bbox:
                cv2.destroyWindow(window_name)
                return self.selector.selected_bbox
            elif key == ord('r'):
                self.selector.selected_bbox = None
                self.selector.start_point = None
                self.selector.end_point = None
            elif key == ord('n'):
                cv2.destroyWindow(window_name)
                return {'action': 'next_frame'}
            elif key == ord('q'):
                cv2.destroyWindow(window_name)
                return None
        cv2.destroyWindow(window_name)
        return None

    def calculate_iou(self, bbox1, bbox2):
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def find_best_detection(self, frame, target_bbox):
        results = self.yolo_model(frame, verbose=False)
        best_match = None
        best_iou = 0.0
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    if confidence >= self.confidence_threshold:
                        iou = self.calculate_iou([x1, y1, x2, y2], target_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            class_name = self.yolo_model.names.get(class_id, f"Class_{class_id}")
                            best_match = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': class_name,
                                'iou': iou
                            }
        return best_match if best_iou > 0.1 else None

    def shrink_bbox(self, bbox, shrink_factor=0.05):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        dx = int(w * shrink_factor)
        dy = int(h * shrink_factor)
        return [int(x1 + dx), int(y1 + dy), int(x2 - dx), int(y2 - dy)]

    def is_spatially_close(self, bbox1, bbox2, max_dist=80):
        c1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        c2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        dist = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
        return dist < max_dist

    def draw_segmentation_mask(self, frame, mask):
        colored_mask = np.zeros_like(frame)
        colored_mask[mask > 0] = (255, 255, 255)
        return cv2.addWeighted(frame, 1.0, colored_mask, 0.4, 0)

    def process_frame_single(self, frame):
        results = self.yolo_model(frame, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    if confidence >= self.confidence_threshold and class_id == self.target_class_id:
                        detections.append([x1, y1, x2, y2, confidence, class_id])
        dets = np.array(detections) if len(detections) > 0 else np.empty((0, 6))
        tracks = self.tracker.update(dets, frame)
        annotated_frame = frame.copy()
        best_track = None
        min_dist = float('inf')
        ref_cx, ref_cy = None, None
        if self.last_known_bbox is not None:
            ref_cx = (self.last_known_bbox[0] + self.last_known_bbox[2]) / 2
            ref_cy = (self.last_known_bbox[1] + self.last_known_bbox[3]) / 2
        elif self.target_bbox is not None:
            ref_cx = (self.target_bbox[0] + self.target_bbox[2]) / 2
            ref_cy = (self.target_bbox[1] + self.target_bbox[3]) / 2
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id, conf = track[:7].astype(int)
            if class_id != self.target_class_id:
                continue
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if ref_cx is not None and ref_cy is not None:
                dist = ((cx - ref_cx) ** 2 + (cy - ref_cy) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    best_track = track
        if best_track is None and len(tracks) > 0:
            for track in tracks:
                x1, y1, x2, y2, track_id, class_id, conf = track[:7].astype(int)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                if ref_cx is not None and ref_cy is not None:
                    dist = ((cx - ref_cx) ** 2 + (cy - ref_cy) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        best_track = track
        target_track = best_track
        if target_track is not None:
            x1, y1, x2, y2, track_id, class_id, conf = target_track[:7].astype(int)
            color = (0, 0, 255)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if self.persistent_id is None:
                self.persistent_id = track_id
            if self.persistent_id not in self.trails:
                self.trails[self.persistent_id] = []
            self.trails[self.persistent_id].append((cx, cy))
            if len(self.trails[self.persistent_id]) > self.max_trail_length:
                self.trails[self.persistent_id] = self.trails[self.persistent_id][-self.max_trail_length:]
            pts = self.trails[self.persistent_id]
            for j in range(1, len(pts)):
                cv2.line(annotated_frame, pts[j-1], pts[j], color, 2)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            label = f"TARGET - ID: {self.persistent_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self.last_known_bbox = [x1, y1, x2, y2]
            self.lost_counter = 0
        if target_track is None:
            if self.last_known_bbox is not None and self.lost_counter < self.max_lost_frames:
                x1, y1, x2, y2 = self.last_known_bbox
                faded_color = (128, 128, 128)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), faded_color, 2)
                cv2.putText(annotated_frame, "Target lost - Press 'R' to re-select", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if self.persistent_id in self.trails:
                    pts = self.trails[self.persistent_id]
                    for j in range(1, len(pts)):
                        cv2.line(annotated_frame, pts[j-1], pts[j], faded_color, 2)
                self.lost_counter += 1
            else:
                self.last_known_bbox = None
                self.lost_counter = 0
        return annotated_frame

    def run(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        cv2.namedWindow("Unified Tracker", cv2.WINDOW_NORMAL)
        first_frame = True
        frame_count = 0
        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera.")
                break
            if first_frame:
                while True:
                    selected = self.select_object(frame)
                    if selected is None:
                        print("No object selected. Exiting...")
                        return
                    if isinstance(selected, dict) and selected.get('action') == 'next_frame':
                        ret, frame = cap.read()
                        if not ret:
                            print("Error: Could not read frame from camera.")
                            return
                        continue
                    selected_bbox = selected
                    break
                best_detection = self.find_best_detection(frame, selected_bbox)
                if best_detection:
                    self.target_bbox = best_detection['bbox']
                    self.target_class_id = best_detection['class_id']
                    self.target_class_name = best_detection['class_name']
                    initial_detection = [self.target_bbox[0], self.target_bbox[1], self.target_bbox[2], self.target_bbox[3], best_detection['confidence'], best_detection['class_id']]
                    tracks = self.tracker.update(np.array([initial_detection]), frame)
                    for track in tracks:
                        if hasattr(track, 'is_confirmed') and track.is_confirmed():
                            self.target_track_id = track.track_id
                            break
                    first_frame = False
                else:
                    print("Could not detect the selected object. Please try again.")
                    continue
            else:
                processed_frame = self.process_frame_single(frame)
                frame_count += 1
                cv2.imshow("Unified Tracker", processed_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif (key == ord('r') or key == ord('R')):
                    first_frame = True
                    self.last_known_bbox = None
                    self.lost_counter = 0
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Unified Object Tracker (YOLOv8 + StrongSORT)")
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='Path to YOLOv8 model (default: yolov8s.pt)')
    parser.add_argument('--camera', type=int, default=0, help='Webcam device ID (default: 0)')
    parser.add_argument('--confidence', type=float, default=0.3, help='Confidence threshold for detections')
    args = parser.parse_args()
    tracker = UnifiedObjectTracker(model_path=args.model, confidence_threshold=args.confidence)
    tracker.run(args.camera)

if __name__ == "__main__":
    main() 