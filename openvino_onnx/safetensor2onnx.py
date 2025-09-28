import torch
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
import argparse
from transformers import AutoModel, AutoTokenizer, AutoProcessor


def export_text_model(model, inputs, output_dir, opset_version=17, precision="fp32"):
    text_model = model.text_model if hasattr(model, "text_model") else model
    text_onnx_path = os.path.join(output_dir, "text_model.onnx")

    torch.onnx.export(
        text_model,
        (inputs["input_ids"],),
        text_onnx_path,
        input_names=["input_ids"],
        output_names=["text_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "text_embeds": {0: "batch"}
        },
        opset_version=opset_version
    )
    print(f"✅ 文本模型已导出 ({precision}):", text_onnx_path)

    # INT8 动态量化
    if precision == "int8":
        text_model_int8_path = text_onnx_path.replace(".onnx", "_int8.onnx")
        quantize_dynamic(
            text_onnx_path,
            text_model_int8_path,
            weight_type=QuantType.QInt8
        )
        print("✅ 文本模型已量化为 INT8:", text_model_int8_path)


def export_vision_model(model, inputs, output_dir, opset_version=17, precision="fp32"):
    vision_model = model.vision_model if hasattr(model, "vision_model") else model
    vision_onnx_path = os.path.join(output_dir, "vision_model.onnx")

    torch.onnx.export(
        vision_model,
        (inputs["pixel_values"],),
        vision_onnx_path,
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes={
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "image_embeds": {0: "batch"}
        },
        opset_version=opset_version
    )
    print(f"✅ 视觉模型已导出 ({precision}):", vision_onnx_path)

    # INT8 动态量化
    if precision == "int8":
        vision_model_int8_path = vision_onnx_path.replace(".onnx", "_int8.onnx")
        quantize_dynamic(
            vision_onnx_path,
            vision_model_int8_path,
            weight_type=QuantType.QUInt8
        )
        print("✅ 视觉模型已量化为 INT8:", vision_model_int8_path)


def export_to_onnx_and_quantize(model, processor_or_tokenizer, output_dir, inputs, opset_version=17, precision="fp32"):
    os.makedirs(output_dir, exist_ok=True)

    # precision 逻辑
    if precision == "fp16":
        model = model.half()
    else:
        model = model.float()

    model.eval()

    # 文本导出
    if "input_ids" in inputs:
        export_text_model(model, inputs, output_dir, opset_version, precision)

    # 视觉导出
    if "pixel_values" in inputs:
        export_vision_model(model, inputs, output_dir, opset_version, precision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出 Transformer 模型到 ONNX 并可选量化")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="模型路径或 HuggingFace 模型名")
    parser.add_argument("--output_dir", type=str, required=True, help="导出目录")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本 (默认: 17)")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "int8"], default="fp32", help="导出精度")
    parser.add_argument("--is_vision", action="store_true", help="是否导出视觉模型")
    parser.add_argument("--is_text", action="store_true", help="是否导出文本模型")

    args = parser.parse_args()

    # 加载模型 & processor/tokenizer
    model = AutoModel.from_pretrained(args.model_name_or_path)

    if args.is_vision:
        processor = AutoProcessor.from_pretrained(args.model_name_or_path)
        inputs = processor(images=torch.randn(1, 3, 224, 224), return_tensors="pt")
    elif args.is_text:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        inputs = tokenizer("Hello world", return_tensors="pt")
    else:
        raise ValueError("必须指定 --is_text 或 --is_vision 至少一个")

    export_to_onnx_and_quantize(model, processor if args.is_vision else tokenizer, args.output_dir, inputs, args.opset, args.precision)
