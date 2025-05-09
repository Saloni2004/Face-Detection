# -*- coding: utf-8 -*-
"""Untitled50.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HYaCBdKiC5CsGzt0lmjQWnyl9fD1plT6
"""

!ls /content/unzipped_wider_face/WIDER_train/WIDER_train

!pip install -q kaggle


!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json


!kaggle datasets download -d mksaad/wider-face-a-face-detection-benchmark


!mkdir -p data/WIDER_FACE
!unzip -q wider-face-a-face-detection-benchmark.zip -d data/WIDER_FACE



import zipfile
import os


zip_path = '/content/wider-face-a-face-detection-benchmark.zip'
extract_path = '/content/unzipped_wider_face'


os.makedirs(extract_path, exist_ok=True)


with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(f"Zip file extracted to: {extract_path}")


print("The contents are:")
for root, dirs, files in os.walk(extract_path):

    print("Directory:", root)

    if dirs:
        print(" Subdirectories:", dirs)
    if files:
        print(" Files:", files)
    #
    break

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt


def collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    images = torch.stack(images, 0)
    return images, targets

!ls /content/unzipped_wider_face/WIDER_val/WIDER_val/images

class WIDERFaceDataset(Dataset):
    def __init__(self, img_folder, annot_path, transform=None):
        """
        :param img_folder: Path to the folder containing images.
        :param annot_path: Path to the annotation text file.
        :param transform: Torchvision transformations to apply.
        """
        self.img_folder = img_folder
        self.transform = transform
        self.image_paths = []
        self.annotations = []

        with open(annot_path, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.endswith('.jpg'):

                img_full_path = os.path.join(self.img_folder, line)
                self.image_paths.append(img_full_path)
                i += 1
                num_faces = int(lines[i].strip())
                i += 1
                boxes = []
                for _ in range(num_faces):
                    box_data = list(map(float, lines[i].strip().split()))
                    if len(box_data) >= 4:
                        boxes.append(box_data[:4])
                    i += 1
                self.annotations.append(boxes)
            else:
                i += 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        boxes = self.annotations[index]
        target = torch.tensor(boxes) if boxes and len(boxes) > 0 else torch.empty((0, 4))
        return image, target


transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

train_img_folder = '/content/unzipped_wider_face/WIDER_train/WIDER_train/images'
train_annot_path = '/content/unzipped_wider_face/wider_face_split/wider_face_split/wider_face_train_bbx_gt.txt'
val_img_folder = '/content/unzipped_wider_face/WIDER_val/WIDER_val/images'
val_annot_path = '/content/unzipped_wider_face/wider_face_split/wider_face_split/wider_face_val_bbx_gt.txt'


train_dataset = WIDERFaceDataset(train_img_folder, train_annot_path, transform=transform)
val_dataset = WIDERFaceDataset(val_img_folder, val_annot_path, transform=transform)



subset_train_dataset = Subset(train_dataset, list(range(min(20, len(train_dataset)))))
subset_val_dataset = Subset(val_dataset, list(range(min(5, len(val_dataset)))))


train_loader = DataLoader(
    subset_train_dataset,
    batch_size=10,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True
)
val_loader = DataLoader(
    subset_val_dataset,
    batch_size=10,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True
)

class RetinaFaceSimple(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(RetinaFaceSimple, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.stage1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1,
        )
        self.stage2 = resnet.layer2
        self.stage3 = resnet.layer3
        self.stage4 = resnet.layer4


        self.fpn1 = nn.Conv2d(256, 256, kernel_size=1)
        self.fpn2 = nn.Conv2d(512, 256, kernel_size=1)
        self.fpn3 = nn.Conv2d(1024, 256, kernel_size=1)

        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.cls_head = nn.Linear(256, 2)
        self.box_head = nn.Linear(256, 4)

    def forward(self, x):
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)

        f1 = self.fpn1(stage1)
        f2 = self.fpn2(stage2)
        f3 = self.fpn3(stage3)

        f3_up = nn.functional.interpolate(f3, size=f2.shape[2:], mode='nearest')
        f2 = f2 + f3_up
        f2 = self.smooth1(f2)

        f2_up = nn.functional.interpolate(f2, size=f1.shape[2:], mode='nearest')
        f1 = f1 + f2_up
        f1 = self.smooth2(f1)

        pooled = self.avgpool(f1)
        pooled = pooled.view(pooled.size(0), -1)

        cls_out = self.cls_head(pooled)
        box_out = self.box_head(pooled)
        return cls_out, box_out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, targets in data_loader:
        images = images.to(device)

        gt_boxes = []
        for target in targets:
            if target.numel() > 0:
                gt_boxes.append(target[0].tolist())
            else:
                gt_boxes.append([0, 0, 0, 0])
        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32, device=device)
        cls_labels = torch.ones(len(images), dtype=torch.long, device=device)

        optimizer.zero_grad()
        pred_cls, pred_box = model(images)
        loss_cls = nn.CrossEntropyLoss()(pred_cls, cls_labels)
        loss_box = nn.SmoothL1Loss()(pred_box, gt_boxes)
        loss = loss_cls + loss_box
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def validate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            gt_boxes = []
            for target in targets:
                if target.numel() > 0:
                    gt_boxes.append(target[0].tolist())
                else:
                    gt_boxes.append([0, 0, 0, 0])
            gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32, device=device)
            cls_labels = torch.ones(len(images), dtype=torch.long, device=device)

            pred_cls, pred_box = model(images)
            loss_cls = nn.CrossEntropyLoss()(pred_cls, cls_labels)
            loss_box = nn.SmoothL1Loss()(pred_box, gt_boxes)
            loss = loss_cls + loss_box
            total_loss += loss.item()
    return total_loss / len(data_loader)

learning_rates = [0.001, 0.0005]
num_tuning_epochs = 2
best_lr = None
best_val_loss = float('inf')

print("Starting hyperparameter tuning:")
for lr in learning_rates:
    print(f"\nTesting learning rate: {lr}")
    model = RetinaFaceSimple().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_tuning_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        print(f"  Epoch {epoch+1}/{num_tuning_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_lr = lr

print(f"\nBest learning rate selected: {best_lr}")

num_epochs = 5
model = RetinaFaceSimple().to(device)
optimizer = optim.Adam(model.parameters(), lr=best_lr)

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    val_loss = validate(model, val_loader, device)
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")


torch.save(model.state_dict(), "retinaface_simple.pth")

def visualize_prediction(model, dataset, device, index):
    model.eval()

    orig_image = np.array(Image.open(dataset.image_paths[index]).convert('RGB'))

    image, _ = dataset[index]
    image_tensor = image.unsqueeze(0).to(device)
    with torch.no_grad():
        cls_pred, box_pred = model(image_tensor)

    box = box_pred[0].cpu().numpy()
    x, y, w, h = box

    x, y, w, h = int(x), int(y), int(w), int(h)

    plt.figure(figsize=(8,8))
    plt.imshow(orig_image)
    rect = plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2)
    plt.gca().add_patch(rect)
    plt.title("Predicted Face Bounding Box")
    plt.axis("off")
    plt.show()


visualize_prediction(model, val_dataset, device, index=1)