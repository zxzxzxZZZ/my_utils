from PIL import Image, UnidentifiedImageError
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn
from io import BytesIO
import os
import json
import random
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AdvImageTextDataset(Dataset):
    def __init__(self, data_dict, processor, model=None, loss_fn=None,
                 epsilon=0.01, mode="mix",
                 label_map_path=None,
                 transform=None, augment_prob=0.5, pair_num=2, device=None):
        """
        mode: "clean" | "adv" | "mix"
        model + loss_fn + epsilon 用于生成对抗样本
        """
        self.processor = processor
        self.model = model
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.mode = mode
        self.augment_prob = augment_prob
        self.pair_num = pair_num
        self.device = device if device else (next(model.parameters()).device if model else "cpu")
        # print(self.device)
        # 标签映射
        if label_map_path is not None:
            try:
                with open(label_map_path, 'r', encoding='utf-8') as f:
                    self.label_map = json.load(f)
            except (TypeError, FileNotFoundError, json.JSONDecodeError) as e:
                print(f"[Warning] label_map init failed! Reason: {e}")
                self.label_map = {}
        else:
            self.label_map = {}

        # 构建 (image_data, label_id) 列表
        self.data = []
        for label, img_list in data_dict.items():
            if label not in self.label_map:
                raise ValueError(f"发现未定义的标签：{label}")
            for img_bytes in img_list:
                self.data.append((img_bytes, self.label_map[label]))

        # transform
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
        """安全增强"""
        try:
            return self.transform(image)
        except Exception as e:
            print(f"[Warning] Image augmentation failed: {e}")
            return image

    def __getitem__(self, idx):
        image_data, label_id = self.data[idx]

        # 加载图片 (PIL)
        if isinstance(image_data, str) and os.path.exists(image_data):
            image = Image.open(image_data).convert("RGB")
        elif isinstance(image_data, (bytes, bytearray)):
            image = Image.open(BytesIO(image_data)).convert("RGB")
        else:
            raise ValueError("Unsupported image_data type")

        # processor 直接转 tensor
        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].to(self.device)  # (1, 3, H, W)
        labels = torch.tensor([label_id], dtype=torch.long, device=self.device)

        # FGSM 攻击
        if self.mode == "adv":
            pixel_values = self.fgsm_attack(pixel_values, labels)
        elif self.mode == "mix":
            adv_pixel_values = self.fgsm_attack(pixel_values, labels)
            # print("adv_pixel_values.shape:", adv_pixel_values.shape)
            pixel_values = torch.cat([pixel_values, adv_pixel_values], dim=0)
            # print("pixel_values.shape:", pixel_values.shape)
            labels = [label_id] * pixel_values.shape[0]
            # print("labels_len:", pixel_values)
            # print("labels_len:", labels)


        return {
            "pixel_values": pixel_values,   # (B, 3, H, W)
            "labels": labels
        }

    def fgsm_attack(self, pixel_values, labels):
        self.model.eval()
        pixel_values = pixel_values.clone().detach().to(self.device)
        pixel_values.requires_grad = True

        outputs = self.model(pixel_values=pixel_values)
        logits = outputs['logits']
        loss = self.loss_fn(logits, labels)

        self.model.zero_grad()

        # 获取梯度
        grad = torch.autograd.grad(
            loss, pixel_values,
            retain_graph=False,
            create_graph=False
        )[0]

        adv = pixel_values + self.epsilon * grad.sign()
        adv = torch.clamp(adv, 0, 1).detach()
        return adv
    
    def collate_fn(self, batch):
        # 处理 pixel_values
        # b["pixel_values"] 可能是 Tensor，也可能是 list of Tensor
        pixel_values_list = []
        labels_list = []

        for b in batch:
            # 如果 pixel_values 是 list，就展开（比如含有对抗样本）
            if isinstance(b["pixel_values"], list):
                pixel_values_list.extend(b["pixel_values"])
            else:
                pixel_values_list.append(b["pixel_values"])

            # labels 同样处理
            if isinstance(b["labels"], list):
                labels_list.extend(b["labels"])
            else:
                labels_list.append(b["labels"])

        # 拼接 pixel_values (假设每张图已经是 [C,H,W])
        pixel_values = torch.stack(pixel_values_list, dim=0)
        b, n = pixel_values.shape[:2]
        pixel_values = pixel_values.view(b * n, *pixel_values.shape[2:])

        # 拼接 labels
        labels = torch.tensor(labels_list, dtype=torch.long)

        return {"pixel_values": pixel_values, "labels": labels}