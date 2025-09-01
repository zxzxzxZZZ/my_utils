from torchvision import transforms
import random
import torch
import os
import io
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from io import BytesIO
import json
class ImageTextDataset(Dataset):
    def __init__(self, data_dict, processor, label_map_path="label_map.json",
                 transform=None, augment_prob=0.5, pair_num=2):
        """
        data_dict: {label1: [image_bytes1, image_bytes2, ...], label2: [...], ...}
        processor: HuggingFace 的 AutoProcessor
        transform: 可选 torchvision transforms
        augment_prob: 数据增强概率
        pair_num: 是否生成裁剪对 (pair_num>1 表示生成原图+裁剪图)
        """
        self.processor = processor
        self.augment_prob = augment_prob
        self.pair_num = pair_num

        # 标签映射
        try:
            with open(label_map_path, 'r', encoding='utf-8') as f:
                self.label_map = json.load(f)
        except (TypeError, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error, label_map init failed! Reason: {e}")
            self.label_map = {}

        # 构建 (image_data, label_id) 列表
        self.data = []
        for label, img_list in data_dict.items():
            if label not in self.label_map:
                raise ValueError(f"发现未定义的标签：{label}")
            for img_bytes in img_list:
                self.data.append((img_bytes, self.label_map[label]))

        # transform逻辑：传入优先，否则使用默认
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(degrees=10),
            ])

    def __len__(self):
        return len(self.data)

    def safe_augment(self, image):
        """安全增强，异常返回原图"""
        try:
            return self.transform(image)
        except Exception as e:
            print(f"[Warning] Image augmentation failed: {e}")
            return image

    def __getitem__(self, idx):
        image_data, label_id = self.data[idx]

        # 加载图像
        try:
            if isinstance(image_data, str):
                if os.path.exists(image_data):
                    image = Image.open(image_data).convert("RGB")
                else:
                    image_data = eval(image_data)
                    image = Image.open(BytesIO(image_data)).convert("RGB")
            elif isinstance(image_data, (bytes, bytearray)):
                image = Image.open(BytesIO(image_data)).convert("RGB")
            else:
                raise ValueError(f"Unsupported image_data type: {type(image_data)}")
        except (UnidentifiedImageError, ValueError) as e:
            print(f"[Warning] Skipping corrupted sample: {e}")
            return None

        # 可选裁剪对
        if self.pair_num > 1 and "location" in dir(image_data):  # 如果有坐标信息
            width, height = image.size
            xmin, xmax, ymin, ymax = image_data["location"]
            x1, y1 = int(xmin * width), int(ymin * height)
            x2, y2 = int(xmax * width), int(ymax * height)
            cropped = image.crop((x1, y1, x2, y2))
            if random.random() < self.augment_prob:
                image = self.safe_augment(image)
            if random.random() < self.augment_prob:
                cropped = self.safe_augment(cropped)
            images = [image, cropped]
        else:
            # 单图增强
            if random.random() < self.augment_prob:
                image = self.safe_augment(image)
            images = image

        # 使用 processor
        if isinstance(images, list):
            encoding = self.processor(images=images, return_tensors="pt")
        else:
            encoding = self.processor(images=images, return_tensors="pt")

        pixel_values = encoding["pixel_values"].squeeze()
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label_id, dtype=torch.long)
        }

    def collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return {
                "labels": torch.tensor([]),
                "pixel_values": torch.empty(0, 3, *self.processor.size),
            }

        pixel_values = torch.cat([b["pixel_values"] for b in batch], dim=0)
        labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)

        return {
            "labels": labels,
            "pixel_values": pixel_values,
        }
    


