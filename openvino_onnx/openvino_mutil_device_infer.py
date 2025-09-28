import numpy as np
from transformers import AutoProcessor
from PIL import Image
from openvino.runtime import Core
import torch
import torch.nn.functional as F
import argparse

# ---------------------------
# 工具函数：计算文本-图像距离
# ---------------------------
def pair_distance(text_embeds, image_embeds):
    if isinstance(text_embeds, np.ndarray):
        text_embeds = torch.from_numpy(text_embeds).float()
    if isinstance(image_embeds, np.ndarray):
        image_embeds = torch.from_numpy(image_embeds).float()
    text_norm = F.normalize(text_embeds, p=2, dim=-1)
    image_norm = F.normalize(image_embeds, p=2, dim=-1)
    sim = (text_norm @ image_norm.T).squeeze()
    return 2 * (1 - sim)

# ---------------------------
# 核心函数：执行 OpenVINO 推理
# ---------------------------
def openvino_inference(
    processor_path: str,
    vision_model_path: str,
    text_model_path: str,
    image_path: str,
    texts: list,
    devices: list,
    use_hetero: bool = False
):
    """
    使用 OpenVINO 推理文本和视觉模型，并返回文本-图像距离
    """
    # 1. 加载处理器
    processor = AutoProcessor.from_pretrained(processor_path)

    # 2. 加载图像
    image = Image.open(image_path)

    # 3. 处理输入
    inputs = processor(text=texts, images=image, padding="max_length", return_tensors="np")

    # 4. 初始化 OpenVINO Core
    core = Core()

    # 内部函数：编译模型
    def compile_model(model_path, devices, use_hetero=False):
        model = core.read_model(model_path)
        if use_hetero:
            compiled = core.compile_model(model, "HETERO:" + ",".join(devices))
        elif len(devices) > 1:
            compiled = core.compile_model(model, "MULTI:" + ",".join(devices))
        else:
            compiled = core.compile_model(model, devices[0])
        return compiled

    # 5. 编译模型
    compiled_vision = compile_model(vision_model_path, devices, use_hetero)
    compiled_text = compile_model(text_model_path, devices, use_hetero)

    # 6. 构造输入数据
    input_vision = {"pixel_values": inputs["pixel_values"]}
    input_text = {"input_ids": inputs["input_ids"]}

    # 7. 推理
    vision_out = compiled_vision(input_vision["pixel_values"])
    image_embeds = vision_out[1]  # 根据模型输出调整索引

    text_out = compiled_text(input_text["input_ids"])
    text_embeds = text_out[1]

    # 8. 计算文本-图像距离
    score = pair_distance(text_embeds, image_embeds)
    return score

# ---------------------------
# main 函数：解析命令行参数
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenVINO 多设备推理")
    parser.add_argument("--processor_path", type=str, required=True, help="PyTorch/transformers checkpoint 路径")
    parser.add_argument("--vision_model_path", type=str, required=True, help="OpenVINO 视觉模型 XML 路径")
    parser.add_argument("--text_model_path", type=str, required=True, help="OpenVINO 文本模型 XML 路径")
    parser.add_argument("--image_path", type=str, required=True, help="测试图片路径")
    parser.add_argument("--texts", nargs='+', required=True, help="文本列表，用空格分隔")
    parser.add_argument("--devices", nargs='+', default=["CPU"], help="设备列表，如 CPU GPU NPU")
    parser.add_argument("--use_hetero", action="store_true", help="是否使用 HETERO 算子级分配")

    args = parser.parse_args()

    score = openvino_inference(
        processor_path=args.processor_path,
        vision_model_path=args.vision_model_path,
        text_model_path=args.text_model_path,
        image_path=args.image_path,
        texts=args.texts,
        devices=args.devices,
        use_hetero=args.use_hetero
    )

    print("文本-图像距离:", score)
