# AdvImageTextDataset 文档

## 1. 类概述

`AdvImageTextDataset` 是一个用于图像分类任务的数据集类，支持以下功能：

- **标准图像加载**：支持从文件路径或二进制数据（bytes/bytearray）加载图片。
- **数据增强**：提供可选的图像增强（如水平翻转、颜色抖动、旋转）。
- **对抗样本生成**：支持使用 **FGSM (Fast Gradient Sign Method)** 生成对抗样本。
- **Mix 模式**：可以在每个样本中同时使用原始图像和对抗图像进行训练。
- **自定义标签映射**：可通过 JSON 文件映射标签名称到整数 ID。
- **可与 PyTorch DataLoader 无缝结合**：提供自定义 `collate_fn` 支持批处理。

---

## 2. 初始化参数

```python
AdvImageTextDataset(
    data_dict,         # dict, {label_name: [image_data1, image_data2, ...]}
    processor,         # 图像处理器（如 HuggingFace AutoProcessor）
    model=None,        # 用于生成对抗样本的模型
    loss_fn=None,      # 对抗样本损失函数
    epsilon=0.01,      # 对抗扰动幅度
    mode="mix",        # 数据模式： "clean" | "adv" | "mix"
    label_map_path=None,# 标签映射 JSON 文件路径
    transform=None,    # torchvision transforms，图像增强
    augment_prob=0.5,  # 数据增强概率
    pair_num=2,        # Mix 模式下每个样本的复制数量
    device=None        # 设备，默认使用模型所在设备或 GPU
)
```

## 2. 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `data_dict` | dict | 原始数据字典，键为标签名，值为图片路径或字节列表 |
| `processor` | HF Processor | 将 PIL 图像转换为模型可接受的 tensor |
| `model` | nn.Module | 对抗样本生成模型 |
| `loss_fn` | torch loss | 对抗样本生成时使用的损失函数 |
| `epsilon` | float | FGSM 扰动幅度 |
| `mode` | str | `"clean"` 只使用原图，`"adv"` 只使用对抗样本，`"mix"` 使用原图 + 对抗样本 |
| `label_map_path` | str | JSON 文件路径，将标签名映射为整数 ID |
| `transform` | torchvision.transforms | 自定义数据增强操作 |
| `augment_prob` | float | 数据增强概率 |
| `pair_num` | int | Mix 模式下每个样本复制数量 |
| `device` | torch.device | 数据存放设备 |

---

## 3. 核心方法

### 3.1 `__getitem__(self, idx)`

- 加载图片（支持路径或 bytes）
- 将图片通过 `processor` 转为 tensor
- 根据 `mode` 决定是否生成对抗样本：
  - `"clean"`：只返回原图
  - `"adv"`：只返回对抗样本
  - `"mix"`：返回原图 + 对抗样本
- 返回字典：
```python
{
    "pixel_values": tensor,  # [B, 3, H, W]
    "labels": tensor         # [B]
}
```
### 3.2 `fgsm_attack(self, pixel_values, labels)`

使用 **FGSM** 生成对抗样本：

```python
adv = pixel_values + epsilon * grad.sign()
adv = torch.clamp(adv, 0, 1)
```
**参数**：

- `pixel_values`：输入图像 tensor  
- `labels`：对应标签  

**返回**：

- `adv_pixel_values`：对抗图像 tensor  

**注**：  
FGSM 对同一输入生成的扰动相同，如需多样化可使用随机噪声或 PGD 方法

### 3.3 `collate_fn(self, batch)`

自定义批处理函数，用于处理 Mix 模式下的多张图像。

**功能**：

- 支持 `pixel_values` 为 `Tensor` 或 `list`  
- 支持 `labels` 为 `Tensor` 或 `list`  
- 自动展开 Mix 模式中每个样本的多张图片，并保持标签对应  

**输出**：

```python
{
    "pixel_values": torch.Tensor,  # [batch_size, 3, H, W]
    "labels": torch.Tensor         # [batch_size]
}
```
- 自动展开 Mix 模式中每个样本的多张图片
### 4. 数据增强

默认使用的 `transforms`：

```python
transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(degrees=10),
])
```
- 可以传入自定义 transform 来替换默认的数据增强操作。

### 5. 使用示例

```python
from transformers import AutoProcessor
from torch.utils.data import DataLoader
import torch.nn as nn

# 初始化 processor
processor = AutoProcessor.from_pretrained("model_name")

# 构建数据集
dataset = AdvImageTextDataset(
    data_dict=my_data_dict,
    processor=processor,
    model=my_model,
    loss_fn=nn.CrossEntropyLoss(),
    epsilon=0.01,
    mode="mix",
    label_map_path="label_map.json"
)

# 构建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=dataset.collate_fn
)

# 遍历批次
for batch in dataloader:
    images = batch["pixel_values"]  # [B, 3, H, W]
    labels = batch["labels"]        # [B]

```
### 6. 注意事项

- **设备匹配**：确保 `processor` 和 `model` 输出的 tensor 与 `device` 一致。

- **FGSM 扰动**：
  - 单步 FGSM 对同一输入生成的扰动相同。
  - 如需生成多样化对抗样本，可使用随机初始化或 PGD 方法。

- **Mix 模式**：
  - 返回的每个样本可能包含原图 + 对抗图。
  - `collate_fn` 会正确展开并对应标签。
