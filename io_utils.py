"""
ComfyUI 图像张量与 PIL 互转、输出目录生成等小工具。
尽量保持“纯工具”，方便单测。
"""

from pathlib import Path
from datetime import datetime
from typing import Tuple
import numpy as np
from PIL import Image
import torch

def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    输入：ComfyUI IMAGE（float32，0~1，可能是 [B,H,W,C] 或 [H,W,C]）
    输出：PIL.Image（RGB）
    """
    # 处理批次维度：取第一张
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    
    # 转换为 numpy: [H,W,C]
    img_np = image_tensor.cpu().numpy()
    
    # 从 [0,1] float32 转换为 [0,255] uint8
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    
    # 转换为 PIL
    return Image.fromarray(img_np, mode='RGB')

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    可选：将 PIL 转回 Comfy 规范张量（float32, 0~1, [H,W,C]）
    """
    # 转换为 RGB（防止 RGBA 等）
    img = img.convert('RGB')
    
    # 转换为 numpy: [H,W,C]
    img_np = np.array(img, dtype=np.float32)
    
    # 从 [0,255] 转换为 [0,1]
    img_np = img_np / 255.0
    
    # 转换为 tensor
    return torch.from_numpy(img_np)

def make_output_dir(root: Path) -> Path:
    """
    在 root 下创建时间戳目录，返回路径。
    示例：ComfyUI/output/DeepSeekOCR/2025-11-05_20-31-00/
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def redraw_boxes_on_image(image: Image.Image, text_output: str, color=(255, 0, 0), width=2) -> Image.Image:
    """
    在图像上重新绘制检测框（自定义颜色和宽度）
    
    Args:
        image: PIL Image
        text_output: DeepSeek-OCR 的文本输出，包含 <|ref|>...<|det|>[[x1,y1,x2,y2]]<|/det|> 标记
        color: 边框颜色 RGB，默认红色 (255, 0, 0)
        width: 边框宽度，默认 2px
    
    Returns:
        带标注框的 PIL Image
    """
    import re
    from PIL import ImageDraw, ImageFont
    
    # 复制图像
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    # 解析文本中的检测框坐标
    # 格式：<|ref|>标签<|/ref|><|det|>[[x1,y1,x2,y2], ...]<|/det|>
    pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>'
    matches = re.findall(pattern, text_output, re.DOTALL)
    
    image_width, image_height = image.size
    
    for label, coords_str in matches:
        try:
            # 解析坐标
            coords_list = eval(coords_str)  # [[x1,y1,x2,y2], ...]
            
            for coords in coords_list:
                x1, y1, x2, y2 = coords
                
                # 坐标归一化：DeepSeek-OCR 使用 0-999 范围
                x1 = int(x1 / 999 * image_width)
                y1 = int(y1 / 999 * image_height)
                x2 = int(x2 / 999 * image_width)
                y2 = int(y2 / 999 * image_height)
                
                # 绘制边框
                draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                
                # 绘制标签（可选）
                if label.strip():
                    try:
                        font = ImageFont.load_default()
                        text_bbox = draw.textbbox((0, 0), label, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        
                        text_x = x1
                        text_y = max(0, y1 - text_height - 2)
                        
                        # 白色背景
                        draw.rectangle(
                            [text_x, text_y, text_x + text_width + 4, text_y + text_height + 2],
                            fill=(255, 255, 255)
                        )
                        # 文字
                        draw.text((text_x + 2, text_y), label, font=font, fill=color)
                    except:
                        pass
        except Exception as e:
            print(f"[DeepseekOCR] 绘制边框失败: {e}")
            continue
    
    return img_with_boxes
