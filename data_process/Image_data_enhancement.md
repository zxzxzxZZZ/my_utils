# ImageTextDataset 使用说明

## 1. 类功能简介

`ImageTextDataset` 是一个用于图像-文本数据处理的 PyTorch 数据集类，主要用途是：

- 支持从字节流或本地路径加载图像；
- 可选对图像进行数据增强（翻转、亮度/对比度/饱和度调整、旋转等）；
- 支持生成裁剪对（原图 + 裁剪图），方便用于对比学习或多视图训练；
- 将图像与对应标签映射为模型可接受的张量形式（通过 `processor`）；
- 提供 `collate_fn` 方法用于 DataLoader 批处理。

适用于视觉分类、多视图对比学习或图像-文本联合任务的数据预处理。

---

## 2. 标签映射 (label_map)

- `label_map.json` 用于将文本标签映射为整数 ID。
- 格式示例：
```json
{
    "dog": 0,
    "cat": 1,
    "bird": 2
}
```
- Dataset 初始化时会检查 data_dict 中的标签是否存在于 label_map 中，若不存在会报错。
## 3. 数据组织格式

### 输入数据

`data_dict` 是一个 Python 字典，键为标签名（必须在 `label_map` 中），值为图像列表（每个图像可以是字节流或本地路径）：

```python
data_dict = {
    "dog": ["dog_image1.jpg", "dog_image2.jpg", "dog_image3.jpg"],
    "cat": ["cat_image1.jpg", "cat_image2.jpg"],
    "bird": ["bird_image1.jpg", "bird_image2.jpg"]
}
```
- 其中图片应是对应的完整路径
## 4. 主要参数说明

| 参数 | 说明 |
|------|------|
| `data_dict` | 数据字典，{label: [image_bytes_or_path]} |
| `processor` | HuggingFace 的 AutoProcessor，用于图像编码 |
| `label_map_path` | 标签映射 JSON 文件路径 |
| `transform` | 可选的 torchvision transform，用于数据增强 |
| `augment_prob` | 数据增强概率 |
| `pair_num` | 是否生成裁剪对 (pair_num>1 表示生成原图+裁剪图) |

## 5. 使用示例
```python
from transformers import AutoProcessor
from torch.utils.data import DataLoader

# 初始化 processor
processor = AutoProcessor.from_pretrained("模型名")

# 创建数据集
dataset = ImageTextDataset(
    data_dict=data_dict,
    processor=processor,
    label_map_path="label_map.json",
    pair_num=2
)

# 传入 DataLoader
loader = DataLoader(dataset, batch_size=16, collate_fn=dataset.collate_fn)

```