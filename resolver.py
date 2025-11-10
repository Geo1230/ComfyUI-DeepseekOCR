"""
将 UI 选择（分辨率预设/任务模式）映射为推理所需的参数与 Prompt。
这里全部是“纯函数”，不依赖 GPU/Transformers，易于单测。
"""

from typing import Dict

# ---- 分辨率映射（DeepSeek-OCR README 的推荐预设） ----
# Gundam 为动态分辨率（大文档更准）
_PRESET_TABLE = {
    "tiny":   dict(base_size=512,  image_size=512,  crop_mode=False, test_compress=False),
    "small":  dict(base_size=640,  image_size=640,  crop_mode=False, test_compress=False),
    "base":   dict(base_size=1024, image_size=1024, crop_mode=False, test_compress=False),
    "large":  dict(base_size=1280, image_size=1280, crop_mode=False, test_compress=False),
    "gundam": dict(base_size=1024, image_size=640,  crop_mode=True,  test_compress=True),
}

def map_resolution(preset: str) -> Dict:
    """
    输入：'Gundam'/'Tiny'/'Small'/'Base'/'Large'（大小写均可）
    输出：infer(...) 所需四个参数。
    """
    key = (preset or "gundam").strip().lower()
    return _PRESET_TABLE.get(key, _PRESET_TABLE["gundam"]).copy()

def build_prompt(task: str, ref_text: str = "") -> str:
    """
    将任务模式转成官方示例的 Prompt 模板。
    - Free OCR
    - Convert to Markdown
    - Parse Figure
    - Locate by Reference (需要 ref_text)
    """
    t = (task or "").strip().lower()
    if t in ("free ocr", "free_ocr", "free"):
        return "<image>\nFree OCR."
    if t in ("convert to markdown", "markdown", "md"):
        return "<image>\n<|grounding|>Convert the document to markdown."
    if t in ("parse figure", "figure", "chart", "table"):
        return "<image>\nParse the figure."
    if t in ("locate by reference", "locate", "grounding"):
        ref = ref_text or "object"
        return f"<image>\nLocate <|ref|>{ref}<|/ref|> in the image."
    # 默认兜底
    return "<image>\nOCR this image."
