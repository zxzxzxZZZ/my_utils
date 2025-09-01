# 通用数据翻译处理模板（Python）

这段代码实现了一个通用数据翻译处理模板，主要用于批量翻译 JSON 或 JSONL 数据文件中的文本内容，支持多语言翻译、多线程加速以及错误管理。该模板可作为数据处理、文本翻译或 NLP 预处理的通用框架，具有高度可定制性。

## 核心功能

### 多语言翻译支持
- 通过 `languages` 参数可指定目标语言列表，如 `["English", "German", "French"]`。
- 输出严格按照输入列表顺序生成，确保原始文本与翻译结果一一对应。
- 结果动态保存，每处理一条追加写一条，防止程序崩溃导致处理后的数据丢失。

### 可自定义 Prompt 模板
- `prompt_template`：定义翻译指令模板，可根据任务灵活修改提示内容。
- `output_template`：定义输出示例模板，确保翻译结果符合预期的字典结构。

### 客户端自定义与扩展
- `client_factory` 可自定义 OpenAI 客户端创建方法，方便接入不同 API 或私有部署服务。
- `client_params` 可传入任意客户端初始化参数，如 `api_key`、`base_url`、`timeout` 等。
- `api_kwargs` 可在 `run` 方法中传入额外的请求参数，如 `temperature`、`max_tokens`、`top_p` 等。

### 多线程处理
- 使用 `ThreadPoolExecutor` 并发执行翻译任务，加快大规模数据处理速度。
- `max_workers` 参数可根据机器性能调整线程数。

### 错误管理与日志
- 遇到翻译失败或格式异常时，会将错误信息保存到指定的 `error_file` 文件中。
- 正确的翻译结果写入 `output_file` 文件，以 JSONL 格式保存，每条记录对应原始数据及其翻译结果。



## 代码结构说明

### `TranslationProcessor` 类：核心处理类
- **初始化方法 `__init__`**：设置文件路径、客户端、模型、模板、线程数、语言列表等。
- **`_translate_item` 方法**：处理单条数据的翻译及异常捕获。
- **`run` 方法**：启动多线程翻译任务，并显示进度条。

### `create_client` 函数
演示自定义客户端的创建方法，返回 OpenAI client 对象。

### `if __name__ == "__main__"`：示例使用代码
- 指定源文件 `source_file`、输出文件 `output_file`、错误文件 `error_file`。
- 配置客户端参数、Prompt 模板和翻译语言。
- 可选传入 `api_kwargs` 控制模型参数。

## 使用示例

```python
processor = TranslationProcessor(
    source_file="data/input.json",
    output_file="data/output.jsonl",
    error_file="data/errors.jsonl",
    client_factory=create_client,
    client_params={
        "api_key": "test",
        "base_url": "http://10.10.20.147:46389/v1"
    },
    model="qwen3",
    prompt_template=prompt_template,
    output_template=output_template,
    max_workers=32,
    languages=["English", "German"]
)
processor.run(api_kwargs={"temperature": 0.7, "max_tokens": 2048})

```
## 输入输出示例
```python
# 假设目标语言仅为English，(原数据输入格式固定)
 # TO DO: 添加更多适配的数据格式
原数据(json)：
    [
        {path:"pic.jpg", label:["一朵花","一朵小红花"]},
        ....
    ]
输出数据(jsonl)：
    {path:"pic.jpg", label:["一朵花", "一朵红花", "a flower", "a red flower"]}
    ....
    