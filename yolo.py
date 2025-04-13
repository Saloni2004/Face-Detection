import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# YOLOv1 Model Implementation (no change required in architecture besides C=1)
class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=1):
        """
        YOLOv1 model implementation for face detection
        
        Args:
            S (int): Grid size (SxS)
            B (int): Number of bounding boxes per grid cell
            C (int): Number of classes (for face detection, use 1)
        """
        super(YOLOv1, self).__init__()
        self.S = S  # Grid size
        self.B = B  # Number of bounding boxes per grid cell
        self.C = C  # Number of classes (set to 1 for face detection)
        
        # Darknet-inspired backbone (simplified version)
        self.features = nn.Sequential(
            # Initial conv layers
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 2
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 3
            nn.Conv2d(192, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 4
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Final layers
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1)
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9216, 4096),  # Assuming input size is 448x448
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (C + B * 5))  # For each grid cell: class probability (face) + B bounding boxes with 5 values each
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, 448, 448)
            
        Returns:
            Output tensor of shape (batch_size, S*S*(C+B*5))
        """
        x = self.features(x)  # (batch_size, 1024, 7, 7)
        x = self.fc(x)  # (batch_size, S*S*(C+B*5))
        
        # Reshape to (batch_size, S, S, C+B*5)
        batch_size = x.size(0)
        x = x.view(batch_size, self.S, self.S, self.C + self.B * 5)
        
        return x

# Custom Dataset class for YOLO specialized for face detection
class YOLOFaceDataset(Dataset):
    def __init__(self, image_dir, labels_dir, S=7, B=2, C=1, transform=None):
        """
        Custom dataset for YOLO face detection.
        
        Assumes that for each image in 'image_dir' there exists a label text file
        in 'labels_dir' with the same basename and a .txt extension. Each label file
        contains one bounding box per line in the format:
            x_center y_center width height
        
        Args:
            image_dir (str): Directory containing the images.
            labels_dir (str): Directory containing the label txt files.
            S (int): Grid size.
            B (int): Number of bounding boxes per grid cell.
            C (int): Number of classes (use 1 for face detection).
            transform: Image transformations.
        """
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C  # For face detection, this will be 1.
        self.class_names = ["face"]
        
        # Build list of images and corresponding label files.
        self.images = sorted([
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        # For each image, the label file is assumed to have the same basename with .txt extension.
        self.labels = []
        for img_path in self.images:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_dir, base_name + ".txt")
            boxes = []
            # If the label file exists, load its annotations.
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    if line.strip() == "":
                        continue
                    # Each line is: x_center y_center width height
                    parts = line.strip().split()
                    if len(parts) != 4:
                        continue
                    x_center, y_center, width, height = map(float, parts)
                    # For face detection, we set the class index as 0.
                    boxes.append([0, x_center, y_center, width, height])
            # If there are no annotations, boxes remain empty.
            self.labels.append(boxes)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get item by index
        
        Args:
            idx (int): Index
            
        Returns:
            image: Tensor of shape (3, 448, 448)
            target: Tensor of shape (S, S, C+B*5)
        """
        img_path = self.images[idx]
        boxes = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Create target tensor for YOLO
        # For each cell in the grid, we have C class probabilities and B bounding box predictions.
        target = torch.zeros(self.S, self.S, self.C + self.B * 5)
        
        for box in boxes:
            class_idx, x, y, width, height = box
            
            # Determine which cell the center falls in.
            grid_x = int(self.S * x)
            grid_y = int(self.S * y)
            
            # Handle edge cases
            grid_x = min(self.S - 1, grid_x)
            grid_y = min(self.S - 1, grid_y)
            
            # Calculate relative coordinates within the cell.
            x_cell = self.S * x - grid_x
            y_cell = self.S * y - grid_y
            
            # Set target values.
            # We set the class probability to 1 for the 'face' class.
            target[grid_y, grid_x, class_idx] = 1
            
            # For each bounding box predictor.
            for b in range(self.B):
                base_idx = self.C + b * 5
                target[grid_y, grid_x, base_idx]     = x_cell
                target[grid_y, grid_x, base_idx + 1] = y_cell
                target[grid_y, grid_x, base_idx + 2] = width
                target[grid_y, grid_x, base_idx + 3] = height
                target[grid_y, grid_x, base_idx + 4] = 1  # Confidence score
        
        return image, target

# Loss function for YOLO (remains unchanged)
class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=1, coord_scale=5, noobj_scale=0.5):
        """
        YOLO Loss Function for face detection
        
        Args:
            S (int): Grid size
            B (int): Number of bounding boxes per grid cell
            C (int): Number of classes (1 for face)
            coord_scale (float): Scale for coordinate prediction loss
            noobj_scale (float): Scale for no object confidence loss
        """
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.coord_scale = coord_scale
        self.noobj_scale = noobj_scale
        self.mse = nn.MSELoss(reduction="sum")
    
    def forward(self, predictions, targets):
        """
        Calculate YOLO loss
        
        Args:
            predictions: Tensor of shape (batch_size, S, S, C+B*5)
            targets: Tensor of shape (batch_size, S, S, C+B*5)
            
        Returns:
            loss: Total loss
        """
        batch_size = predictions.size(0)
        
        # Reshape for easier calculations
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        
        # Calculate IoU for both bounding boxes
        iou_b1 = self.calculate_iou(
            predictions[..., self.C:self.C+4],
            targets[..., self.C:self.C+4]
        )
        
        iou_b2 = self.calculate_iou(
            predictions[..., self.C+5:self.C+9],
            targets[..., self.C:self.C+4]
        )
        
        # Get the box with higher IoU
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = targets[..., self.C+4].unsqueeze(3)  # Identity of object (1 if object exists)
        
        # BOX COORDINATE LOSS
        box_predictions = exists_box * (
            best_box * predictions[..., self.C+5:self.C+9]
            + (1 - best_box) * predictions[..., self.C:self.C+4]
        )
        
        box_targets = exists_box * targets[..., self.C:self.C+4]
        
        # Take sqrt of width, height
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )
        
        # OBJECT LOSS
        pred_box = (
            best_box * predictions[..., self.C+9:self.C+10]
            + (1 - best_box) * predictions[..., self.C+4:self.C+5]
        )
        
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * targets[..., self.C+4:self.C+5])
        )
        
        # NO OBJECT LOSS
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C+4:self.C+5], start_dim=1),
            torch.flatten((1 - exists_box) * targets[..., self.C+4:self.C+5], start_dim=1)
        )
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C+9:self.C+10], start_dim=1),
            torch.flatten((1 - exists_box) * targets[..., self.C+4:self.C+5], start_dim=1)
        )
        
        # CLASS LOSS
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
            torch.flatten(exists_box * targets[..., :self.C], end_dim=-2)
        )
        
        # TOTAL LOSS
        loss = (
            self.coord_scale * box_loss
            + object_loss
            + self.noobj_scale * no_object_loss
            + class_loss
        )
        
        return loss
    
    def calculate_iou(self, boxes1, boxes2):
        """
        Calculate IoU between two sets of boxes
        
        Args:
            boxes1: Tensor of shape (batch_size, S, S, 4)
            boxes2: Tensor of shape (batch_size, S, S, 4)
            
        Returns:
            IoU: Tensor of shape (batch_size, S, S)
        """
        # Transform from center coords to corner coords
        box1_x1 = boxes1[..., 0:1] - boxes1[..., 2:3] / 2
        box1_y1 = boxes1[..., 1:2] - boxes1[..., 3:4] / 2
        box1_x2 = boxes1[..., 0:1] + boxes1[..., 2:3] / 2
        box1_y2 = boxes1[..., 1:2] + boxes1[..., 3:4] / 2
        box2_x1 = boxes2[..., 0:1] - boxes2[..., 2:3] / 2
        box2_y1 = boxes2[..., 1:2] - boxes2[..., 3:4] / 2
        box2_x2 = boxes2[..., 0:1] + boxes2[..., 2:3] / 2
        box2_y2 = boxes2[..., 1:2] + boxes2[..., 3:4] / 2
        
        # Intersection
        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        box1_area = torch.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2_area = torch.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
        union = box1_area + box2_area - intersection
        
        iou = intersection / (union + 1e-6)
        return iou

# YOLO Trainer (no major changes required aside from using the updated model and dataset)
class YOLOTrainer:
    def __init__(self, model, train_loader, val_loader=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        YOLO Trainer for face detection
        
        Args:
            model: YOLO model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use for training
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = YOLOLoss().to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
    
    def train(self, epochs, save_path=None):
        """
        Train the model
        
        Args:
            epochs (int): Number of epochs
            save_path (str): Path to save the model
            
        Returns:
            History of training and validation losses
        """
        history = {
            'train_loss': [],
            'val_loss': [] if self.val_loader else None
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch_idx, (images, targets) in enumerate(self.train_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(images)
                loss = self.criterion(predictions, targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / len(self.train_loader)
            history['train_loss'].append(avg_train_loss)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
            
            if self.val_loader:
                val_loss = self.validate()
                history['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}")
                self.scheduler.step(val_loss)
                
                if val_loss < best_val_loss and save_path:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': val_loss,
                    }, save_path)
                    print(f"Model saved to {save_path}")
        
        return history
    
    def validate(self):
        """
        Validate the model
        
        Returns:
            val_loss: Validation loss
        """
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(images)
                loss = self.criterion(predictions, targets)
                val_loss += loss.item()
        
        return val_loss / len(self.val_loader)

    def inference(self, image_path, model_weights='face_yolov1', conf_threshold=0.5, iou_threshold=0.5):
        """
        Run inference on an image
        
        Args:
            image_path (str): Path to image
            conf_threshold (float): Confidence threshold
            iou_threshold (float): IoU threshold for NMS
            
        Returns:
            boxes (list): List of bounding boxes in format [x1, y1, x2, y2, class_id, confidence]
        """
        print("Loading model weights...")
        self.model.load_state_dict(torch.load(model_weights, map_location='cpu')['model_state_dict'])
        self.model.to(self.device)
        print("Model loaded successfully")
        print("Running inference...")
        self.model.eval()
        
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])
        
        image = Image.open(image_path).convert("RGB")
        orig_width, orig_height = image.size
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        print("Predictions obtained")
        boxes = self._process_predictions(predictions[0], conf_threshold, iou_threshold)
        
        for box in boxes:
            box[0] *= orig_width
            box[1] *= orig_height
            box[2] *= orig_width
            box[3] *= orig_height
        
        return boxes, image
    
    def _process_predictions(self, prediction, conf_threshold, iou_threshold):
        S, B, C = self.model.S, self.model.B, self.model.C
        
        boxes = []
        
        for row in range(S):
            for col in range(S):
                for b in range(B):
                    conf = prediction[row, col, C + b * 5 + 4]
                    
                    if conf > conf_threshold:
                        x = prediction[row, col, C + b * 5]
                        y = prediction[row, col, C + b * 5 + 1]
                        w = prediction[row, col, C + b * 5 + 2]
                        h = prediction[row, col, C + b * 5 + 3]
                        
                        x = (col + x) / S
                        y = (row + y) / S
                        
                        class_scores = prediction[row, col, :C]
                        class_id = torch.argmax(class_scores).item()
                        class_conf = class_scores[class_id]
                        
                        final_conf = conf * class_conf
                        
                        if final_conf > conf_threshold:
                            x1 = max(0, x - w / 2)
                            y1 = max(0, y - h / 2)
                            x2 = min(1, x + w / 2)
                            y2 = min(1, y + h / 2)
                            
                            boxes.append([x1, y1, x2, y2, class_id, final_conf])
        
        if not boxes:
            return []
        
        boxes = torch.tensor(boxes, device=self.device)
        boxes = self._non_max_suppression(boxes, iou_threshold)
        
        return boxes.cpu().numpy().tolist()
    
    def _non_max_suppression(self, boxes, iou_threshold):
        boxes = boxes[boxes[:, 5].argsort(descending=True)]
        keep_boxes = []
        
        while boxes.size(0) > 0:
            keep_boxes.append(boxes[0])
            if boxes.size(0) == 1:
                break
            ious = self._box_iou(boxes[0, :4], boxes[1:, :4])
            mask = ious < iou_threshold
            boxes = boxes[1:][mask]
        
        return torch.stack(keep_boxes) if keep_boxes else torch.tensor([])
    
    def _box_iou(self, box1, boxes):
        x1 = torch.max(box1[0], boxes[:, 0])
        y1 = torch.max(box1[1], boxes[:, 1])
        x2 = torch.min(box1[2], boxes[:, 2])
        y2 = torch.min(box1[3], boxes[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box1_area + boxes_area - intersection
        
        return intersection / (union + 1e-6)

    def visualize_prediction(self, image, boxes, class_names, save_path=None):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        ax = plt.gca()
        
        for box in boxes:
            x1, y1, x2, y2, class_id, conf = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, 
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            class_name = class_names[int(class_id)]
            plt.text(
                x1, y1 - 5, 
                f"{class_name}: {conf:.2f}",
                color='white', fontsize=12, 
                bbox=dict(facecolor='red', alpha=0.5)
            )
        
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            print(f"Visualization saved to {save_path}")
        
        plt.show()

# Example usage for face detection
def main():
    # For face detection, we only have one class.
    class_names = ["face"]
    
    # Initialize the model with C=1.
    model = YOLOv1(S=7, B=2, C=len(class_names))
    
    # Define transformations.
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets using image and label directories.
    train_dataset = YOLOFaceDataset(
        image_dir='/csehome/karki.1/yolov1/images/train',
        labels_dir='/csehome/karki.1/yolov1/labels2',  # Folder containing label txt files
        transform=transform
    )
    
    val_dataset = YOLOFaceDataset(
        image_dir='/csehome/karki.1/yolov1/images/val',
        labels_dir='/csehome/karki.1/yolov1/labels2',
        transform=transform
    )
    
    # Create data loaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create the trainer.
    trainer = YOLOTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Train the model (update epochs and save_path as needed)
    # history = trainer.train(epochs=50, save_path='face_yolov1.pth')

    # Example inference on a test image
    
    boxes, image = trainer.inference('/csehome/karki.1/yolov1/images/val/0a0bdec4b07ca3c1.jpg')
    trainer.visualize_prediction(image, boxes, class_names, save_path='inference_result.jpg')


if __name__ == '__main__':
    main()
