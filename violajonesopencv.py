import cv2
import os
import numpy as np
from glob import glob

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou

def load_dataset(image_dir, label_dir):
    data = []
    image_paths = glob(os.path.join(image_dir, "*.jpg"))
    
    for img_path in image_paths:
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # Load corresponding label
        label_path = os.path.join(label_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
        gt_boxes = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    
                    # Convert YOLO format to pixel coordinates
                    img_h, img_w = image.shape[:2]
                    x_center *= img_w
                    y_center *= img_h
                    width *= img_w
                    height *= img_h
                    
                    x1 = int(x_center - width/2)
                    y1 = int(y_center - height/2)
                    x2 = int(x_center + width/2)
                    y2 = int(y_center + height/2)
                    
                    gt_boxes.append([x1, y1, x2, y2])
        
        data.append((image, gt_boxes))
    
    return data

def evaluate_viola_jones(test_dir, label_dir, output_dir):
    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset(test_dir, label_dir)
    total_iou = 0
    total_detections = 0
    processed_images = 0
    
    for idx, (image, gt_boxes) in enumerate(dataset):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Convert to (x1, y1, x2, y2) format
        det_boxes = [(x, y, x+w, y+h) for (x, y, w, h) in faces]
        
        # Calculate IoU for each detection
        max_ious = []
        for det_box in det_boxes:
            max_iou = 0
            for gt_box in gt_boxes:
                iou = calculate_iou(det_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
            if max_iou > 0:
                max_ious.append(max_iou)
        
        # Update metrics
        if len(max_ious) > 0:
            total_iou += sum(max_ious)
            total_detections += len(max_ious)
        
        # Draw results
        result_img = image.copy()
        
        # Draw ground truth (blue)
        for box in gt_boxes:
            cv2.rectangle(result_img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            
        # Draw detections (green)
        for (x1, y1, x2, y2) in det_boxes:
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Save output
        output_path = os.path.join(output_dir, f"result_{idx}.jpg")
        cv2.imwrite(output_path, result_img)
        processed_images += 1
    
    # Calculate final metrics
    avg_iou = total_iou / total_detections if total_detections > 0 else 0
    print(f"Processed {processed_images} images")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Total detections: {total_detections}")

if __name__ == "__main__":

    TEST_DIR = "images/val"
    LABEL_DIR = "labels/val"
    OUTPUT_DIR = "detection_results"

    evaluate_viola_jones(TEST_DIR, LABEL_DIR, OUTPUT_DIR)