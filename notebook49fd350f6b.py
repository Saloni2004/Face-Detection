import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import traceback
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------- Convert YOLO to bounding box ---------
def yolo_to_bbox(yolo, w, h):
    x_center, y_center, width, height = map(float, yolo[1:])
    x1 = int((x_center - width / 2) * w)
    y1 = int((y_center - height / 2) * h)
    x2 = int((x_center + width / 2) * w)
    y2 = int((y_center + height / 2) * h)
    return [x1, y1, x2, y2]

# --------- Dataset ---------
class FaceDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.label_dir, name + ".txt")

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape

            bboxes = []
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    bbox = yolo_to_bbox(parts, w, h)
                    bboxes.append(bbox)

            if self.transform:
                image_tensor = self.transform(image)
            else:
                image_tensor = torch.tensor(image).permute(2, 0, 1)

            return {'image': image_tensor, 'bboxes': torch.tensor(bboxes, dtype=torch.float32), 'img_raw': image}
        except Exception as e:
            print(f"[ERROR in dataset @ index {idx}]: {e}")
            traceback.print_exc()
            return None

# --------- Training Function ---------
def train_mtcnn(model, dataloader, num_epochs=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        for i, batch in enumerate(dataloader):
            try:
                images = batch['image'].to(device)
                bboxes_list = batch['bboxes']
                raw_imgs = batch['img_raw']

                for j in range(images.size(0)):
                    img_np = raw_imgs[j]
                    gt_bboxes = bboxes_list[j].to(device)

                    if gt_bboxes.shape[0] == 0:
                        continue

                    det_boxes, _ = model.detect(img_np, landmarks=False)
                    if det_boxes is None or len(det_boxes) == 0:
                        continue

                    pred_boxes = torch.tensor(det_boxes, dtype=torch.float32).to(device)

                    min_len = min(len(pred_boxes), len(gt_bboxes))
                    loss = loss_fn(pred_boxes[:min_len], gt_bboxes[:min_len])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    print(f"Epoch {epoch+1} | Batch {i+1} | Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"[ERROR in training loop @ batch {i}]: {e}")
                traceback.print_exc()

# --------- Visualization ---------
def visualize(model, dataset, count=5):
    model.eval()
    shown = 0
    with torch.no_grad():
        for sample in dataset:
            try:
                img = sample['img_raw']
                gt_boxes = sample['bboxes']
                boxes, _ = model.detect(img)


def compute_iou(boxA, boxB):
    # box: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = max(0, (boxA[2] - boxA[0] + 1)) * max(0, (boxA[3] - boxA[1] + 1))
    boxBArea = max(0, (boxB[2] - boxB[0] + 1)) * max(0, (boxB[3] - boxB[1] + 1))

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def visualize(model, dataset, count=10):
    model.eval()
    total_iou = 0
    total_boxes = 0
    shown = 0

    indices = random.sample(range(len(dataset)), count)

    with torch.no_grad():
        for idx in indices:
            try:
                sample = dataset[idx]
                img = sample['img_raw']
                gt_boxes = sample['bboxes']
                boxes, _ = model.detect(img)

                vis_img = img.copy()

                print(f"\n--- Image {shown + 1} ---")
                print("Ground Truth Boxes:")
                for gt in gt_boxes:
                    print(f"  {gt.tolist()}")
                    x1, y1, x2, y2 = map(int, gt.tolist())
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if boxes is not None:
                    print("Predicted Boxes:")
                    for pred in boxes:
                        print(f"  {pred.tolist()}")
                        x1, y1, x2, y2 = map(int, pred.tolist())
                        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # Compute IoU between each GT and closest predicted box
                    for gt in gt_boxes:
                        gt_np = gt.cpu().numpy()
                        best_iou = 0
                        for pred in boxes:
                            iou = compute_iou(gt_np, pred)
                            best_iou = max(best_iou, iou)
                        total_iou += best_iou
                        total_boxes += 1

                plt.imshow(vis_img)
                plt.title(f"Blue: Predicted | Green: GT | Image {shown+1}")
                plt.axis("off")
                plt.show()

                shown += 1

            except Exception as e:
                print(f"[ERROR in visualization]: {e}")
                traceback.print_exc()

    mean_iou = total_iou / total_boxes if total_boxes > 0 else 0
    print(f"\nMean IoU over {total_boxes} GT boxes: {mean_iou:.4f}")

# --------- Run ---------
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = FaceDataset("/kaggle/input/face-detection-dataset/images/train", "/kaggle/input/face-detection-dataset/labels/train", transform=transform)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=lambda x: {
        'image': torch.stack([i['image'] for i in x if i]),
        'bboxes': [i['bboxes'] for i in x if i],
        'img_raw': [i['img_raw'] for i in x if i]
    })

    mtcnn_model = MTCNN(keep_all=True, device=device)
    train_mtcnn(mtcnn_model, train_loader, num_epochs=3)

    val_set = FaceDataset("/kaggle/input/face-detection-dataset/images/val", "/kaggle/input/face-detection-dataset/labels/val", transform=transform)
    visualize(mtcnn_model, val_set)
