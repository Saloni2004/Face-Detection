import os
import glob
import cv2
import numpy as np


class ViolaJonesDetector:
    def __init__(self, window_size=128):
        """
        Initializes the Viola-Jones detector with square window.

        Parameters:
            window_size (int): Size of the square window used for detection.
        """
        self.window_size = window_size
        self.data = []
        self.images = []
        self.labels = []

    def get_integral_image(self, image):
        """
        Computes the integral image.

        Parameters:
            image (np.ndarray): Input grayscale or BGR image.

        Returns:
            np.ndarray: Integral image.
        """
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.cumsum(axis=0).cumsum(axis=1)

    def evaluate_haar_features(self, integral_image, x, y, window_size=None):
        """
        Evaluates Haar-like features for a given square window in the integral image.

        Parameters:
            integral_image (np.ndarray): Computed integral image.
            x (int): Top-left x-coordinate for the window.
            y (int): Top-left y-coordinate for the window.
            window_size (int, optional): Size of the square window. Defaults to self.window_size.

        Returns:
            np.ndarray: 1D array of five Haar feature values.
        """
        if window_size is None:
            window_size = self.window_size

        def rect_sum(ii, top, left, bottom, right):
            """
            Calculate sum of pixels in a rectangle using integral image.
            
            Parameters:
                ii (np.ndarray): Integral image
                top, left, bottom, right: Rectangle coordinates
                
            Returns:
                float: Sum of pixel values in the rectangle
            """
            # Handle boundary cases carefully to prevent overflow
            A = ii[top - 1, left - 1] if top > 0 and left > 0 else 0.0
            B = ii[top - 1, right] if top > 0 else 0.0
            C = ii[bottom, left - 1] if left > 0 else 0.0
            D = ii[bottom, right]
            
            # Use float to prevent overflow
            return float(D - B - C + A)

        h, w = integral_image.shape
        half_size = window_size // 2
        quarter_size = window_size // 4
        
        # Check if window fits within the image
        if x + window_size > w or y + window_size > h:
            raise ValueError("Window exceeds image bounds")

        features = []

        # Two-rectangle horizontal feature (left-right split)
        left_val = rect_sum(integral_image, y, x, y + window_size - 1, x + half_size - 1)
        right_val = rect_sum(integral_image, y, x + half_size, y + window_size - 1, x + window_size - 1)
        features.append(right_val - left_val)

        # Two-rectangle vertical feature (top-bottom split)
        top_val = rect_sum(integral_image, y, x, y + half_size - 1, x + window_size - 1)
        bottom_val = rect_sum(integral_image, y + half_size, x, y + window_size - 1, x + window_size - 1)
        features.append(bottom_val - top_val)

        # Three-rectangle horizontal feature (left-middle-right)
        left_rect = rect_sum(integral_image, y, x, y + window_size - 1, x + quarter_size - 1)
        mid_rect = rect_sum(integral_image, y, x + quarter_size, y + window_size - 1, x + 3 * quarter_size - 1)
        right_rect = rect_sum(integral_image, y, x + 3 * quarter_size, y + window_size - 1, x + window_size - 1)
        features.append(mid_rect - (left_rect + right_rect))

        # Three-rectangle vertical feature (top-middle-bottom)
        top_rect = rect_sum(integral_image, y, x, y + quarter_size - 1, x + window_size - 1)
        mid_rect = rect_sum(integral_image, y + quarter_size, x, y + 3 * quarter_size - 1, x + window_size - 1)
        bottom_rect = rect_sum(integral_image, y + 3 * quarter_size, x, y + window_size - 1, x + window_size - 1)
        features.append(mid_rect - (top_rect + bottom_rect))

        # Four-rectangle feature (checkerboard)
        top_left = rect_sum(integral_image, y, x, y + half_size - 1, x + half_size - 1)
        top_right = rect_sum(integral_image, y, x + half_size, y + half_size - 1, x + window_size - 1)
        bottom_left = rect_sum(integral_image, y + half_size, x, y + window_size - 1, x + half_size - 1)
        bottom_right = rect_sum(integral_image, y + half_size, x + half_size, y + window_size - 1, x + window_size - 1)
        features.append((top_left + bottom_right) - (top_right + bottom_left))

        return np.array(features)

    def check_face_overlap(self, image, labels, x, y, window_size=None, overlap_threshold=0.85):
        """
        Checks if a detection window overlaps with any face.
        
        Parameters:
            image (np.ndarray): Input image.
            labels (list): List of labels with pixel coordinates [class, x1, y1, x2, y2].
            x (int): Top-left x-coordinate for the window.
            y (int): Top-left y-coordinate for the window.
            window_size (int, optional): Size of the square window. Defaults to self.window_size.
            overlap_threshold (float): Minimum IoU for positive match.
            
        Returns:
            int: 1 if overlap detected, else 0.
        """
        if window_size is None:
            window_size = self.window_size
            
        # Define detection window
        win_x1, win_y1 = x, y
        win_x2, win_y2 = x + window_size, y + window_size
        win_area = window_size * window_size
        
        img_h, img_w = image.shape[:2]
        
        for label in labels:
            # Parse label based on format
            if len(label) == 5 and label[0] == 0:  # YOLO format [class, x_center, y_center, width, height]
                _, xc, yc, w, h = label
                
                # Check if values are normalized (between 0 and 1)
                if 0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                    # Convert normalized YOLO format to pixel coordinates
                    face_x1 = int((xc - w / 2) * img_w)
                    face_y1 = int((yc - h / 2) * img_h)
                    face_x2 = int((xc + w / 2) * img_w)
                    face_y2 = int((yc + h / 2) * img_h)
                else:
                    # Already in pixel coordinates but in center format
                    face_x1 = int(xc - w / 2)
                    face_y1 = int(yc - h / 2)
                    face_x2 = int(xc + w / 2)
                    face_y2 = int(yc + h / 2)
            
            # Compute intersection
            inter_x1 = max(win_x1, face_x1)
            inter_y1 = max(win_y1, face_y1)
            inter_x2 = min(win_x2, face_x2)
            inter_y2 = min(win_y2, face_y2)
            
            # Check if there is overlap
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                face_area = (face_x2 - face_x1) * (face_y2 - face_y1)
                # Calculate IoU (Intersection over Union)
                iou = inter_area / (win_area + face_area - inter_area)
                if iou >= overlap_threshold:
                    return 1
        return 0

    def train_ada_boost(self, steps=3):
        """
        Performs AdaBoost feature selection on training data.

        Parameters:
            steps (int): Number of boosting iterations.

        Returns:
            list: Boosted classifiers with their weights.
        """
        if not self.data:
            raise ValueError("No training data loaded. Call load_data_from_paths first.")
            
        boosted_classifiers = []
        
        # Process each image and extract features
        all_features = []
        all_labels = []
        
        print("Extracting features from images...")
        for img_idx, (img, yolo_labels) in enumerate(self.data):
            # Get integral image
            temp_feat = []
            temp_label = []
            integral_img = self.get_integral_image(img)
            h, w = integral_img.shape
            
            # Skip if image is too small for our window size
            if h < self.window_size or w < self.window_size:
                print(f"Skipping image {img_idx}: too small for feature extraction")
                continue
                
            # Calculate maximum valid positions for window
            max_x = w - self.window_size
            max_y = h - self.window_size
            
            # Ensure we have valid bounds
            if max_x <= 0 or max_y <= 0:
                print(f"Skipping image {img_idx}: insufficient size for window")
                continue
                
            print(f"Processing image {img_idx+1}/{len(self.data)}, size: {w}x{h}")
            
            # Sample windows to reduce computational load
            step_size = max(1, min(max_x, max_y) // 20)  # Adjust sampling density based on image size
            
            # Track positive and negative samples
            pos_count = 0
            neg_count = 0
            max_neg_samples = 1000  # Limit negative samples per image to balance dataset
            
            for window_x in range(0, max_x, step_size):
                for window_y in range(0, max_y, step_size):
                    # Get label (1 if window overlaps with face, 0 otherwise)
                    label = self.check_face_overlap(img, yolo_labels, window_x, window_y)
                    
                    # Get features for this window
                    features = self.evaluate_haar_features(integral_img, window_x, window_y)
                    
                    # Store features and label
                    temp_feat.append(features)
                    temp_label.append(label)
                    
                    # Count samples by type
                    if label == 1:
                        pos_count += 1
                    else:
                        neg_count += 1
            if pos_count > 0:
                # Store features and labels for this image
                for i in range (len(temp_feat)):
                    all_features.append(temp_feat[i])
                    all_labels.append(temp_label[i])
                print(f"Extracted {pos_count} positive and {neg_count} negative samples from image {img_idx+1}")
        
        # Convert to numpy arrays for faster processing
        if not all_features:
            raise ValueError("No valid features extracted. Check image sizes and window size parameters.")
        print(len(all_features))
        print(all_features[0].shape)
        features_array = np.array(all_features)
        labels_array = np.array(all_labels)
        
        num_instances = len(features_array)
        # Initialize weights - balance class weights
        weights = np.ones(num_instances) / num_instances
        
        # AdaBoost iterations
        for step in range(steps):
            print(f"AdaBoost iteration {step+1}/{steps}")
            
            best_classifier = None
            min_error = float('inf')
            
            # For each feature dimension
            for feature_idx in range(features_array.shape[1]):
                feature_values = features_array[:, feature_idx]
                
                # Find min/max for this feature
                if len(feature_values) == 0:
                    continue
                    
                f_min = np.min(feature_values)
                f_max = np.max(feature_values)
                
                # Sample threshold values
                thresholds = np.linspace(f_min, f_max, num=10)
                
                for threshold in thresholds:
                    for polarity in [-1, 1]:
                        # Apply weak classifier
                        predictions = np.ones(num_instances)
                        predictions[polarity * (feature_values - threshold) <= 0] = 0
                        
                        # Calculate weighted error
                        error = np.sum(weights * (predictions != labels_array))
                        
                        if error < min_error:
                            min_error = error
                            best_classifier = (feature_idx, threshold, polarity)
            
            # If we couldn't find a good classifier, break
            if best_classifier is None:
                print("No suitable classifier found. Stopping AdaBoost.")
                break
                
            # Calculate classifier weight
            error = max(min_error, 1e-10)  # Prevent division by zero
            alpha = 0.5 * np.log((1 - error) / error)
            
            # Update sample weights
            feature_idx, threshold, polarity = best_classifier
            predictions = np.ones(num_instances)
            predictions[polarity * (features_array[:, feature_idx] - threshold) <= 0] = 0
            
            # Update weights
            weights *= np.exp(-alpha * (2 * labels_array - 1) * (2 * predictions - 1))
            weights /= np.sum(weights)  # Normalize
            
            # Save this classifier
            boosted_classifiers.append((best_classifier, alpha))
            print(f"Selected classifier: feature {best_classifier[0]}, threshold {best_classifier[1]:.2f}, polarity {best_classifier[2]}, weight {alpha:.4f}")
        
        return boosted_classifiers

    def load_images_and_labels(self, folder, label_folder, img_ext='jpg', label_ext='txt'):
        """
        Loads images and their corresponding YOLO labels from the given directories.

        Parameters:
            folder (str): Path to images.
            label_folder (str): Path to label files.
            img_ext (str): Image file extension.
            label_ext (str): Label file extension.

        Returns:
            list: List of tuples (image, labels).
        """
        data = []
        image_paths = glob.glob(os.path.join(folder, f'*.{img_ext}'))
        
        if not image_paths:
            print(f"No images found in {folder} with extension .{img_ext}")
            return data
            
        # print(f"Found {len(image_paths)} images")
        
        for img_path in image_paths:
            filename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_folder, f'{filename}.{label_ext}')
            
            try:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                    
                labels = []
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        for line in f:
                            labels.append([float(x) for x in line.strip().split()])
                else:
                    print(f"No label file found for {filename}")
                    
                data.append((image, labels))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
        return data

    def load_data_from_paths(self, image_folder, labels_folder, img_ext='jpg', label_ext='txt'):
        """
        Directly loads data by specifying the image and label directories.

        Parameters:
            image_folder (str): Path containing images.
            labels_folder (str): Path containing YOLO label files.
            img_ext (str): Image file extension.
            label_ext (str): Label file extension.
        """
        print(f"Loading data from {image_folder} and {labels_folder}")
        self.data = self.load_images_and_labels(image_folder, labels_folder, img_ext, label_ext)
        self.images = [img for img, _ in self.data]
        self.labels = [lbl for _, lbl in self.data]
        print(f"Loaded {len(self.data)} image-label pairs")

    def draw_detections(self, image, detections):
        """
        Draw detection squares on the image.
        
        Parameters:
            image (np.ndarray): Input image
            detections (list): List of [x, y, size] detections
            
        Returns:
            np.ndarray: Image with drawn detection squares
        """
        output = image.copy()
        for x, y, size in detections:
            cv2.rectangle(output, (x, y), (x + size, y + size), (0, 255, 0), 2)
        return output

    def apply_classifier(self, image, classifiers, window_size=None):
        """
        Apply trained classifier to detect faces in an image.
        
        Parameters:
            image (np.ndarray): Input image
            classifiers (list): List of boosted classifiers
            window_size (int): Window size for detection, defaults to self.window_size
            
        Returns:
            list: Detected face regions as [x, y, size]
        """
        if window_size is None:
            window_size = self.window_size
            
        detections = []
        integral_img = self.get_integral_image(image)
        h, w = integral_img.shape
        
        # Calculate maximum valid window positions
        max_x = w - window_size
        max_y = h - window_size
        
        if max_x <= 0 or max_y <= 0:
            print(f"Image too small for detection with window size {window_size}")
            return detections
        
        # Step size for scanning (smaller step = more overlapping windows = slower but better detection)
        step_size = max(1, window_size // 8)
        
        print(f"Scanning image with window size {window_size}, step size {step_size}")
        
        for window_y in range(0, max_y, step_size):
            for window_x in range(0, max_x, step_size):
                try:
                    # Extract features for this window
                    features = self.evaluate_haar_features(integral_img, window_x, window_y, window_size)
                    
                    # Apply boosted classifier
                    weighted_sum = 0
                    for (feature_idx, threshold, polarity), alpha in classifiers:
                        # Ensure the feature index is valid
                        if feature_idx >= len(features):
                            continue
                            
                        # Apply weak classifier
                        h_val = 1 if polarity * (features[feature_idx] - threshold) > 0 else 0
                        weighted_sum += alpha * h_val

                    # If classified as face, add to detections
                    if weighted_sum > 0:
                        detections.append([window_x, window_y, window_size])
                except (ValueError, IndexError) as e:
                    continue
                    
        print(f"Found {len(detections)} potential faces")
        return detections

    def non_max_suppression(self, detections, overlap_threshold=0.25):
        """
        Apply non-maximum suppression to remove overlapping detections.
        
        Parameters:
            detections (list): List of [x, y, size] detections
            overlap_threshold (float): IoU threshold for suppression
            
        Returns:
            list: Filtered detections
        """
        if not detections:
            return []
            
        # Convert to [x1, y1, x2, y2, score] format
        # For simplicity, we use window size as score since all detections have same confidence
        boxes = []
        for x, y, size in detections:
            boxes.append([x, y, x + size, y + size, size])
        
        boxes = np.array(boxes)
        
        # Extract coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]
        
        # Calculate area of each box
        area = (x2 - x1) * (y2 - y1)
        
        # Sort by size (larger windows generally better with VJ)
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while indices.size > 0:
            # Keep the current highest score
            i = indices[0]
            keep.append(i)
            
            # Calculate intersection with all other boxes
            xx1 = np.maximum(x1[i], x1[indices[1:]])
            yy1 = np.maximum(y1[i], y1[indices[1:]])
            xx2 = np.minimum(x2[i], x2[indices[1:]])
            yy2 = np.minimum(y2[i], y2[indices[1:]])
            
            # Calculate intersection area
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            # Calculate IoU
            union = area[i] + area[indices[1:]] - intersection
            iou = intersection / union
            
            # Keep boxes with IoU below threshold
            inds = np.where(iou <= overlap_threshold)[0] + 1
            indices = indices[inds]
        
        # Return to original format [x, y, size]
        filtered_detections = []
        for i in keep:
            filtered_detections.append([int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][4])])
            
        return filtered_detections

    def detect_faces(self, classifiers, output_dir='output'):
        """
        Apply trained classifier to detect faces in all loaded images.
        
        Parameters:
            classifiers (list): List of boosted classifiers
            output_dir (str): Directory to save results
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for i, image in enumerate(self.images):
            print(f"Processing image {i+1}/{len(self.images)}")
            
            # Detect faces
            detections = self.apply_classifier(image, classifiers)
            
            # Apply non-maximum suppression to remove overlapping detections
            filtered_detections = self.non_max_suppression(detections)
            
            # Draw and save results
            result_image = self.draw_detections(image, filtered_detections)
            cv2.imwrite(f'{output_dir}/detected_{i}.jpg', result_image)
            print(f"Saved detection result to {output_dir}/detected_{i}.jpg")

    def train(self, image_folder, labels_folder, boost_iterations=100):
        """
        Loads data and trains the detector.

        Parameters:
            image_folder (str): Path to training images.
            labels_folder (str): Path to corresponding YOLO labels.
            boost_iterations (int): Number of boosting iterations.

        Returns:
            list: Trained classifier.
        """
        self.load_data_from_paths(image_folder, labels_folder)
        
        if not self.data:
            raise ValueError("No training data loaded. Check image and label paths.")
            
        print(f"Starting training with {boost_iterations} boosting iterations")
        print(f"Using square window size of {self.window_size}x{self.window_size}")
        
        classifiers = self.train_ada_boost(steps=boost_iterations)
        
        # Apply classifier to training data
        print("Applying classifier to training data...")
        self.detect_faces(classifiers)
        
        return classifiers

    def detect(self, image, classifiers, multi_scale=True):
        """
        Detect faces in a single image.
        
        Parameters:
            image (np.ndarray): Input image
            classifiers (list): Trained classifier
            multi_scale (bool): Whether to scan with multiple window sizes
            
        Returns:
            np.ndarray: Image with detection rectangles drawn
        """
        result_image = image.copy()
        all_detections = []
        
        if multi_scale:
            # Use multiple window sizes for better detection
            scale_factors = [0.5, 0.75, 1.0, 1.5, 2.0]
            window_sizes = [int(self.window_size * factor) for factor in scale_factors]
        else:
            window_sizes = [self.window_size]
            
        print(f"Detecting with {len(window_sizes)} window sizes")
        
        for window_size in window_sizes:
            if window_size < 8:  # Too small windows aren't useful
                continue
                
            # Run detection at this scale
            detections = self.apply_classifier(image, classifiers, window_size)
            all_detections.extend(detections)
            
        # Apply non-maximum suppression to remove overlapping detections
        filtered_detections = self.non_max_suppression(all_detections)
        
        # Draw detection rectangles
        return self.draw_detections(result_image, filtered_detections)


# Example Usage
if __name__ == "__main__":
    # Initialize detector with square window size
    detector = ViolaJonesDetector(window_size=128)
    
    # Define paths
    image_folder_path = 'images/train'      # Replace with your images folder path
    labels_folder_path = 'labels/train'     # Replace with your labels folder path
    
    # Create output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')
    
    print("Starting Viola-Jones detector training")
    classifier = detector.train(
        image_folder=image_folder_path, 
        labels_folder=labels_folder_path, 
        boost_iterations=3
    )
    
        # Run detection on a test image
    test_image_path = 'images/train/0a099fc48a03d928.jpg'  # Replace with your test image path
    if os.path.exists(test_image_path):
        test_image = cv2.imread(test_image_path)
        if test_image is not None:
            print("Running detection on test image...")
            detected_image = detector.detect(test_image, classifier, multi_scale=True)
            
            # Save and display result
            cv2.imwrite('output/test_detection.jpg', detected_image)
            print("Test detection saved to output/test_detection.jpg")
            
            # Display the result (comment out if running without GUI)
            cv2.imshow("Detections", detected_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
