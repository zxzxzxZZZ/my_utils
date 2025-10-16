
import argparse
import onnx
import onnxruntime as ort
import numpy as np

def test_onnx_model(onnx_path, batch_size=1, default_hw=224):
    """
    DINOv3 ONNX 模型测试工具
    自动根据模型输入 shape 构造随机输入，并验证推理是否可用。
    
    Args:
        onnx_path (str): ONNX 模型路径
        batch_size (int): 测试 batch size（替换输入的第 0 维）
        default_hw (int): 高/宽默认值，如果 shape 不确定则使用
    """
    # -----------------------------
    # 1. 检查 ONNX 模型结构
    # -----------------------------
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"✅ ONNX 模型结构检查通过: {onnx_path}")
    except onnx.checker.ValidationError as e:
        print(f"❌ ONNX 模型检查失败: {e}")
        return

    # -----------------------------
    # 2. ONNX Runtime 推理
    # -----------------------------
    ort_session = ort.InferenceSession(onnx_path)

    # 获取输入信息
    inputs_info = ort_session.get_inputs()
    outputs_info = ort_session.get_outputs()

    feed_dict = {}

    for inp in inputs_info:
        name = inp.name
        shape = []
        for i, dim in enumerate(inp.shape):
            if isinstance(dim, int):
                shape.append(dim)
            else:
                # batch 维
                if i == 0:
                    shape.append(batch_size)
                # 图像通道
                elif i == 1:
                    shape.append(3)
                # 高/宽
                else:
                    shape.append(default_hw)

        # ONNX dtype 转 numpy dtype
        dtype = np.float32 if "float" in inp.type else np.int64

        # 构造随机输入
        if dtype == np.float32:
            dummy_input = np.random.randn(*shape).astype(dtype)
        else:
            dummy_input = np.random.randint(0, 100, size=shape, dtype=dtype)

        feed_dict[name] = dummy_input
        print(f"🔹 输入 [{name}] shape={shape}, dtype={dtype}")

    # 获取输出名称
    output_names = [out.name for out in outputs_info]

    # 推理
    results = ort_session.run(output_names, feed_dict)

    # 输出结果信息
    for name, result in zip(output_names, results):
        print(f"✅ 输出 [{name}] shape={result.shape}, dtype={result.dtype}")
        print("示例输出:\n", result[:2])  # 打印前 2 条结果


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINOv3 ONNX 模型测试工具")
    parser.add_argument("--onnx_path", type=str, required=True, help="ONNX 模型路径")
    parser.add_argument("--batch_size", type=int, default=1, help="测试 batch size (默认=1)")
    parser.add_argument("--default_hw", type=int, default=256, help="默认图像高/宽 (默认=224)")
    args = parser.parse_args()

    test_onnx_model(args.onnx_path, args.batch_size, args.default_hw)
