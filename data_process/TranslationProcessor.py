import os
import json
import ast
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

class TranslationProcessor:
    def __init__(
        self,
        source_file,
        output_file,
        error_file,
        client_factory=None,
        client_params=None,
        model="qwen3",
        prompt_template=None,
        output_template=None,
        max_workers=32,
        languages=["English"],
        read_json=True
    ):
        """
        通用翻译/数据处理模板
        :param source_file: 原始 JSON 文件路径
        :param output_file: 处理后输出文件路径（jsonl 格式）
        :param error_file: 错误输出文件路径（jsonl 格式）
        :param client_factory: client 创建函数，接收 client_params 返回 client
        :param client_params: client 初始化参数字典
        :param model: 使用的模型
        :param prompt_template: 翻译 prompt 模板
        :param output_template: 输出示例模板
        :param max_workers: 多线程并发数
        :param languages: 翻译语言列表
        :param read_json: 是否读取 JSON 格式（True），否则按 JSONL
        """
        self.source_file = source_file
        self.output_file = output_file
        self.error_file = error_file
        self.model = model
        self.prompt_template = prompt_template or "请将下面的中文文本列表翻译成 {languages}：\n中文文本列表："
        self.output_template = output_template
        self.max_workers = max_workers
        self.languages = languages
        self.write_lock = threading.Lock()

        # 初始化 client
        if callable(client_factory):
            self.client = client_factory(**(client_params or {}))
        else:
            self.client = client_factory

        # 读取数据
        if read_json:
            with open(source_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = []
            with open(source_file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line.strip()))

    def _translate_item(self, idx, data_item, api_kwargs=None):
        api_kwargs = api_kwargs or {}
        try:
            source_text = data_item.get("label", [])[:-1]
            length = len(source_text)

            # 构造 prompt
            content = self.prompt_template.format(languages=", ".join(self.languages)) + str(source_text)
            if self.output_template:
                content += "\n" + self.output_template

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                **api_kwargs
            )

            translation_str = response.choices[0].message.content.strip()
            translation_dict = ast.literal_eval(translation_str)

            # 更新数据
            for lang in self.languages:
                data_item["label"].extend(translation_dict.get(lang, []))

            # 可选长度校验
            if len(data_item["label"]) != (length * (len(self.languages)+1) + 1):
                raise ValueError(f"length mismatch, got {len(data_item['label'])}, expected {(length * (len(self.languages)+1) + 1)}")

            # 写入输出文件
            with self.write_lock:
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(data_item, ensure_ascii=False) + "\n")

            return idx

        except Exception as e:
            error_entry = {"index": idx, "data": data_item, "error": str(e)}
            with self.write_lock:
                with open(self.error_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
            return None

    def run(self, api_kwargs=None):
        api_kwargs = api_kwargs or {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._translate_item, idx, item, api_kwargs): idx for idx, item in enumerate(self.data)}
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                pass
        print("处理完成！输出和错误文件已保存。")


# ---------------- 使用示例 ----------------
def create_client(**kwargs):
    return OpenAI(**kwargs)


if __name__ == "__main__":
    source_file = "data/input.json"
    output_file = "data/output.jsonl"
    error_file = "data/errors.jsonl"

    client_params = {
        "api_key": "test",
        "base_url": "http://10.10.20.147:46389/v1"
    }

    prompt_template = '''
你是一个专业翻译助手。
请将下面的中文文本列表翻译成 {languages}：
'''

    output_template = '''
输出示例格式：
{
    "English": ["...", "...", "..."]
}
'''

    processor = TranslationProcessor(
        source_file=source_file,
        output_file=output_file,
        error_file=error_file,
        client_factory=create_client,
        client_params=client_params,
        model="qwen3",
        prompt_template=prompt_template,
        output_template=output_template,
        max_workers=32,
        languages=["English"]
    )

    # 可传入额外 API 参数，如 temperature, max_tokens 等
    api_kwargs = {
        "temperature": 0.7,
        "max_tokens": 2048
    }

    processor.run(api_kwargs=api_kwargs)
    
