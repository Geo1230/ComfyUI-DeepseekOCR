"""
集中放置常量、路径发现与环境变量读取。
约束：模型必须下载到 ComfyUI/models/DeepSeekOCR/ 目录下。
"""
from pathlib import Path
from dataclasses import dataclass
import os

# 插件根：.../ComfyUI/custom_nodes/ComfyUI-DeepseekOCR
PLUGIN_ROOT = Path(__file__).resolve().parent

# Comfy 根：.../ComfyUI
COMFY_ROOT = PLUGIN_ROOT.parent.parent

# 模型根：.../ComfyUI/models/deepseek-ocr   ← 你要求的固定位置
MODELS_HOME = Path(os.getenv("DEEPSEEK_OCR_HOME", COMFY_ROOT / "models" / "deepseek-ocr"))

# 本插件专属的 HF 缓存（避免污染全局 ~/.cache）
HF_CACHE = MODELS_HOME / "hf_cache"

# DeepSeek-OCR 默认仓库（可通过环境变量覆盖）
REPO_ID = os.getenv("DPSK_REPO_ID", "deepseek-ai/DeepSeek-OCR")
REVISION = os.getenv("DPSK_REVISION", None)  # 可选：固定某个 commit
# 默认启用自动下载（设置 DPSK_AUTODOWNLOAD=0 可禁用）
AUTODOWNLOAD = os.getenv("DPSK_AUTODOWNLOAD", "1") == "1"

def repo2dirname(repo_id: str) -> str:
    # deepseek-ai/DeepSeek-OCR → deepseek-ai_DeepSeek-OCR
    return repo_id.replace("/", "_")

def local_snapshot_dir(repo_id: str) -> Path:
    return MODELS_HOME / repo2dirname(repo_id)

# 输出与日志位置（可用于 Run 节点的默认落盘）
OUTPUT_ROOT = COMFY_ROOT / "output" / "DeepseekOCR"
LOG_DIR     = COMFY_ROOT / "log"


@dataclass(frozen=True)
class LoadConfig:
    local_dir: str   # 由 model_manager.ensure_weights_local 产出
    device: str      # "cuda" / "cpu"
    dtype: str       # "bf16" / "fp16" / "fp32"
    attn_impl: str   # "sdpa" / "eager" / "flash_attention_2"

