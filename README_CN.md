# ComfyUI-DeepseekOCR

[English](README.md) | [中文](README_CN.md)

自用节点，将 **DeepSeek-OCR** 封装为 ComfyUI 插件，提供强大的 OCR 识别和文档解析功能。


## 功能特性



## 快速开始


```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/Geo1230/ComfyUI-DeepseekOCR.git
```

**安装依赖**

如果你使用的是 ComfyUI 自带的便携版 / venv：
运行：
path/to/ComfUI/python_embeded/python.exe -s -m pip install -r requirements.txt

如果你使用的是系统自带的 Python：
运行：
pip install -r requirements.txt

**下载模型**

创建目录并进入：
```bash
# 1. 进入 ComfyUI 的 models 目录
cd ComfyUI\models

# 2. 创建 deepseek-ocr 目录（如果不存在）
mkdir deepseek-ocr
cd deepseek-ocr

# 3. 创建模型目录
mkdir deepseek-ai_DeepSeek-OCR
cd deepseek-ai_DeepSeek-OCR
```

下载模型到当前目录：
```bash
huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir . --repo-type model
```

**说明**：模型会下载到 `ComfyUI\models\deepseek-ocr\deepseek-ai_DeepSeek-OCR\` 目录

**或使用自动下载**（不推荐，稳定性一般）：

首次运行 Load 节点时会自动下载，下载进度在控制台显示。

如果希望**禁用**自动下载，可设置环境变量：
```bash
# Windows PowerShell
$env:DPSK_AUTODOWNLOAD = "0"
```


## 使用方法

### 节点 1：DeepSeek OCR: Load Model

加载模型并缓存，输出模型句柄供 Run 节点使用。

**参数：**
- `dtype`：数据精度
  - `bf16`（推荐，默认值）- 精度与性能平衡
  - `fp16` - 显存不足时使用
  - `fp32` - 兼容性最好但显存占用大
- `device`：运行设备（默认：`cuda`）


### 节点 2：DeepSeek OCR: Run

执行 OCR 推理，输出识别文本。

**参数：**
- `model`：模型句柄（来自 Load 节点）
- `image`：输入图像（ComfyUI IMAGE 类型）
- `task`：任务模式
  - `Free OCR`：通用 OCR 识别
  - `Convert to Markdown`：文档转 Markdown
  - `Parse Figure`：解析图表
  - `Locate by Reference`：定位指定对象（需配合 `reference_text`）
- `resolution`：分辨率预设
  - `Gundam`（推荐，长文档精度高）：1024/640/crop/compress
  - `Tiny`：512x512
  - `Small`：640x640
  - `Base`：1024x1024
  - `Large`：1280x1280
- `output_type`：**输出类型**（决定返回什么内容）
  - `all`（默认）：同时输出文本和可视化图像
  - `text`：仅输出文本，图像输出为原图
  - `image`：仅输出可视化图像（适用于 Locate 任务）
- `reference_text`：（可选）**仅当** task=`Locate by Reference` 时填写，要定位的对象描述
- `box_color`：（可选）检测框颜色，默认 `red`
  - 预设颜色：`red`、`green`、`blue`、`yellow`、`cyan`、`magenta`、`white`、`black`
  - 自定义 RGB：如 `"255,0,0"`（红色）、`"0,255,0"`（绿色）
- `box_width`：（可选）检测框宽度，默认 `2` px，范围 1-10

**输出：**
- `text`：识别的文本内容（STRING）
  - 包含原始标记（如 `<|ref|>...<|/ref|><|det|>[[坐标]]<|/det|>`）
- `visualization`：可视化图像（IMAGE）
  - **Locate by Reference** 任务：带自定义样式标注框的图像
  - 其他任务：返回原始输入图像

## 效果展示


## 使用指南

### 💡 输出类型选择

- `all`（默认）：同时输出文本和可视化图像
- `text`：仅输出文本（OCR/Markdown 转换）
- `image`：仅输出可视化图像（Locate 定位任务）

### 🎯 Locate by Reference 定位任务

**参数配置**：
- `task`：选择 `Locate by Reference`
- `reference_text`：填写要定位的对象
  - 中文示例：`"价格"`、`"标题"`、`"二维码"`
  - 英文示例：`"the teacher"`、`"price"`、`"table"`、`"logo"`

### 🎨 自定义边框样式

**支持的预设颜色（16种）**：

| 颜色名 | RGB | 效果 | 颜色名 | RGB | 效果 |
|--------|-----|------|--------|-----|------|
| `red` | 255,0,0 | 🔴 红色（默认） | `orange` | 255,165,0 | 🟠 橙色 |
| `green` | 0,255,0 | 🟢 绿色 | `purple` | 128,0,128 | 🟣 紫色 |
| `blue` | 0,0,255 | 🔵 蓝色 | `pink` | 255,192,203 | 🩷 粉色 |
| `yellow` | 255,255,0 | 🟡 黄色 | `lime` | 0,255,0 | 🟢 柠檬绿 |
| `cyan` | 0,255,255 | 🔵 青色 | `navy` | 0,0,128 | 🔵 海军蓝 |
| `magenta` | 255,0,255 | 🟣 洋红 | `teal` | 0,128,128 | 🔵 蓝绿 |
| `white` | 255,255,255 | ⚪ 白色 | `gold` | 255,215,0 | 🟡 金色 |
| `black` | 0,0,0 | ⚫ 黑色 | `silver` | 192,192,192 | ⚪ 银色 |

**自定义 RGB 格式**：
- 输入格式：`"R,G,B"`（如 `"255,128,0"` 深橙色）
- 范围：0-255

**边框宽度**：
- `box_width`：1-10 像素（默认 2px）

**示例配置**：
```
box_color = "red"          → 红色 2px 边框（默认）
box_color = "orange"       → 橙色边框
box_color = "255,105,180"  → 亮粉色边框
box_width = 5              → 5px 粗边框
```

### 📌 基本工作流

```
LoadImage
   ↓
DeepSeek OCR: Load Model  
   ↓
DeepSeek OCR: Run
   ├─→ text → Display Text / Save Text
   └─→ visualization → Preview Image / Save Image
```

### 📚 典型应用场景

**1. 文档转 Markdown**
```
task = "Convert to Markdown"
resolution = "Gundam"
→ 输出格式化的 Markdown 文本
```

**2. 图表解析**
```
task = "Parse Figure"
resolution = "Base"
→ 提取表格、图表中的结构化数据
```

**3. 对象定位**
```
task = "Locate by Reference"
reference_text = "哆啦A梦"
box_color = "red"
box_width = 2
→ 文本包含坐标，图像显示红框标注
```


```
ComfyUI/
├─ models/
│  └─ deepseek-ocr/                    # ← 固定权重目录
│     ├─ deepseek-ai_DeepSeek-OCR/     # 模型权重
│     └─ hf_cache/                     # HuggingFace 缓存
├─ output/
│  └─ DeepseekOCR/                     # 输出目录（可视化结果）
│     └─ 2025-11-05_20-31-00/          # 时间戳目录
├─ log/
│  └─ deepseek_ocr.log                 # 插件日志
└─ custom_nodes/
   └─ ComfyUI-DeepseekOCR/
      ├─ __init__.py
      ├─ config.py
      ├─ model_manager.py
      ├─ nodes.py
      ├─ resolver.py
      ├─ io_utils.py
      ├─ tool/
      │  └─ download_weights.py
      ├─ requirements.txt
      └─ README.md
```

## 日志

插件日志位于：`ComfyUI/log/deepseek_ocr.log`

关键日志内容：
- 权重下载进度
- 模型加载状态（device/dtype/attn_impl）
- 缓存命中信息
- 降级策略触发记录
- 错误详情与建议


本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。


## 致谢

- [DeepSeek AI](https://www.deepseek.com/) - 提供强大的 DeepSeek-OCR 模型
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 优秀的节点式 UI 框架
- 所有贡献者和用户
