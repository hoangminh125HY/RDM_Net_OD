import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from random import sample, shuffle

class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, epoch_length=None, mosaic=True, train=True):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.clearimage_lines = annotation_lines  # Giả sử ảnh clear cũng giống
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.mosaic:
            indices = sample(range(self.length), 3)
            indices.append(index)
            lines = [self.annotation_lines[i] for i in indices]

            # Ảnh clear lấy từ ảnh chính
            clear_line = self.clearimage_lines[index]
            clear_img_path = clear_line.split()[0]
            clear_img = cv2.imread(clear_img_path)
            clear_img = cv2.resize(clear_img, self.input_shape)
            clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB) / 255.0

            image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)
        else:
            line = self.annotation_lines[index]
            image, box = self.get_random_data(line, self.input_shape)
            clear_img = image.copy()

        # Đưa về định dạng Tensor
        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        clear_img = torch.from_numpy(np.transpose(clear_img, (2, 0, 1))).float()
        box = np.array(box, dtype=np.float32)

        return image, box, clear_img

    def get_random_data_with_Mosaic(self, lines, input_shape):
        h, w = input_shape
        new_image = np.zeros((h, w, 3), dtype=np.uint8)
        new_boxes = []

        for i, line in enumerate(lines):
            parts = line.strip().split()
            img_path = parts[0]
            boxes = [list(map(int, b.split(','))) for b in parts[1:]]

            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.resize(image, (w // 2, h // 2))
            ih, iw = image.shape[:2]

            if i == 0:
                new_image[:h // 2, :w // 2] = image
                dx, dy = 0, 0
            elif i == 1:
                new_image[:h // 2, w // 2:] = image
                dx, dy = w // 2, 0
            elif i == 2:
                new_image[h // 2:, :w // 2] = image
                dx, dy = 0, h // 2
            elif i == 3:
                new_image[h // 2:, w // 2:] = image
                dx, dy = w // 2, h // 2

            for box in boxes:
                xmin, ymin, xmax, ymax, cls_id = box
                xmin = int(xmin * (w // 2) / iw) + dx
                xmax = int(xmax * (w // 2) / iw) + dx
                ymin = int(ymin * (h // 2) / ih) + dy
                ymax = int(ymax * (h // 2) / ih) + dy
                new_boxes.append([xmin, ymin, xmax, ymax, cls_id])

        new_image = new_image.astype(np.float32) / 255.0
        return new_image, new_boxes

    def get_random_data(self, line, input_shape):
        parts = line.strip().split()
        img_path = parts[0]
        boxes = [list(map(int, b.split(','))) for b in parts[1:]]

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image {img_path} not found")
        image = cv2.resize(image, input_shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        return image, boxes

def yolo_dataset_collate(batch):
    images, bboxes, clearimgs = [], [], []
    for img, box, cimg in batch:
        images.append(img)
        bboxes.append(torch.tensor(box, dtype=torch.float32))
        clearimgs.append(cimg)
    return torch.stack(images), bboxes, torch.stack(clearimgs)
