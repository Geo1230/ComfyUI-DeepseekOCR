"""
两类节点的“薄壳”：
- DeepSeek OCR: Load Model
- DeepSeek OCR: Run
这里只定义接口、参数与调用链路；核心逻辑托管在 model_manager / resolver / io_utils。
"""

from typing import Tuple
import torch
from .config import LoadConfig, REPO_ID
from . import model_manager, resolver, io_utils

# ---------- 节点 1：加载器 ----------
class DeepSeekOCR_Load:
    CATEGORY = "DeepSeek/OCR"
    RETURN_TYPES = ("DPSK_OCR",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "dtype": (["bf16","fp16","fp32"], {"default":"bf16"}),
            "device": ("STRING", {"default":"cuda"})
        }}

    def load(self, dtype: str, device: str):
        """
        1) 确保权重下载到 models/DeepSeekOCR 下（model_manager.ensure_weights_local）
        2) 构造 LoadConfig
        3) 调用 model_manager.load_model(cfg)
        """
        # 1. 确保权重已下载（使用固定的官方仓库）
        local_dir = model_manager.ensure_weights_local(REPO_ID)
        
        # 2. 构造 LoadConfig
        cfg = LoadConfig(
            local_dir=str(local_dir),
            device=device,
            dtype=dtype,
            attn_impl="auto"  # 固定值，模型内部自动选择
        )
        
        # 3. 加载模型
        handle = model_manager.load_model(cfg)
        
        return (handle,)

# ---------- 节点 2：执行器 ----------
class DeepSeekOCR_Run:
    CATEGORY = "DeepSeek/OCR"
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", "visualization")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("DPSK_OCR",),
                "image": ("IMAGE",),
                "task": (["Free OCR","Convert to Markdown","Parse Figure","Locate by Reference"], {"default":"Convert to Markdown"}),
                "resolution": (["Gundam","Tiny","Small","Base","Large"], {"default":"Gundam"}),
                "output_type": (["text", "image", "all"], {"default":"all"}),  # 输出类型
            },
            "optional": {
                "reference_text": ("STRING", {"default": ""}),
                "box_color": ("STRING", {"default": "red"}),
                "box_width": ("INT", {"default": 2, "min": 1, "max": 10}),
            }
        }

    def run(self, model, image, task, resolution, output_type="all", reference_text="", box_color="red", box_width=2):
        """
        1) 张量→PIL（io_utils.tensor_to_pil）
        2) 若 output_dir 为空：使用 io_utils.make_output_dir(...)
        3) 根据 resolution / task 生成 infer 参数与 prompt（resolver.map_resolution / build_prompt）
        4) 调用 model.model.infer(tokenizer=..., image_file=..., prompt=..., base_size=..., ...)
        5) 返回文本 + 可视化图像
        """
        import tempfile
        from pathlib import Path
        from .config import OUTPUT_ROOT
        from PIL import Image
        
        # 解析边框颜色（支持预设颜色名或自定义 RGB）
        color_map = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "orange": (255, 165, 0),
            "purple": (128, 0, 128),
            "pink": (255, 192, 203),
            "lime": (0, 255, 0),
            "navy": (0, 0, 128),
            "teal": (0, 128, 128),
            "gold": (255, 215, 0),
            "silver": (192, 192, 192),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
        }
        
        if box_color.lower() in color_map:
            box_rgb = color_map[box_color.lower()]
        else:
            # 尝试解析 RGB 格式 "255,0,0"
            try:
                parts = box_color.split(',')
                box_rgb = (int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip()))
            except:
                box_rgb = (255, 0, 0)  # 默认红色
                print(f"[DeepseekOCR] 无法解析颜色 '{box_color}'，使用默认红色")
        
        print(f"[DeepseekOCR] 边框样式: 颜色={box_rgb}, 宽度={box_width}px")
        
        # 1. 张量→PIL
        pil_image = io_utils.tensor_to_pil(image)
        
        # 2. 准备临时输出目录（推理完成后会自动清理）
        out_path = io_utils.make_output_dir(OUTPUT_ROOT)
        
        # 3. 保存临时 PNG（DeepSeek-OCR infer 接口需要文件路径）
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=out_path) as tmp:
            tmp_path = tmp.name
            pil_image.save(tmp_path)
        
        # 4. 获取分辨率预设与 Prompt
        preset = resolver.map_resolution(resolution)
        prompt = resolver.build_prompt(task, reference_text)
        
        # 5. 调用推理
        # 注意：我们总是使用 save_results=True 来生成文本文件和可视化
        print(f"[DeepseekOCR] 开始推理...")
        print(f"[DeepseekOCR] 输出类型: {output_type}")
        print(f"[DeepseekOCR] prompt: {prompt[:100]}...")
        print(f"[DeepseekOCR] preset: {preset}")
        
        # 总是启用内部 save_results，以生成 result.mmd 和可视化
        returned_text = model.infer_one(
            image_path=tmp_path,
            prompt=prompt,
            preset=preset,
            save_results=True,  # 内部总是启用，用于生成文本文件
            output_path=str(out_path)
        )
        
        print(f"[DeepseekOCR] 推理完成！")
        
        # 获取文本结果（eval_mode=True 时直接返回）
        if returned_text is not None and str(returned_text).strip():
            text_result = str(returned_text).strip()
            print(f"[DeepseekOCR] ✓ 获取文本结果: {len(text_result)} 字符")
        else:
            text_result = "❌ 模型未返回文本\n\n请检查控制台日志"
            print(f"[DeepseekOCR] ❌ 返回值为空")
        
        # 确保是字符串类型
        if text_result is None:
            text_result = ""
        
        print(f"[DeepseekOCR] 最终文本输出长度: {len(text_result)} 字符")
        if text_result and len(text_result) < 500:
            print(f"[DeepseekOCR] 文本内容:\n{text_result}")
        elif text_result:
            print(f"[DeepseekOCR] 文本预览（前200字符）:\n{text_result[:200]}...")
        
        # 6. 根据 output_type 生成可视化图像
        vis_image = None
        
        # 只有 output_type 包含 image 或 all 时才生成可视化
        if output_type in ["image", "all"]:
            # 如果是 Locate by Reference 任务且有检测框，绘制标注
            if text_result and '<|det|>' in text_result:
                print(f"[DeepseekOCR] 绘制检测框（颜色={box_rgb}, 宽度={box_width}px）")
                try:
                    vis_pil = io_utils.redraw_boxes_on_image(
                        pil_image, 
                        text_result, 
                        color=box_rgb,
                        width=box_width
                    )
                    vis_tensor = io_utils.pil_to_tensor(vis_pil)
                    if vis_tensor.ndim == 3:
                        vis_tensor = vis_tensor.unsqueeze(0)
                    vis_image = vis_tensor
                    print(f"[DeepseekOCR] ✓ 检测框绘制成功")
                except Exception as e:
                    print(f"[DeepseekOCR] 绘制失败: {e}")
                    vis_image = image  # 失败时返回原图
            else:
                # 其他任务或无检测框，直接返回原图
                vis_image = image
        else:
            # output_type="text"，不需要图像输出，返回空白图像
            print(f"[DeepseekOCR] 输出类型为 text，返回原图")
            vis_image = image
        
        # 7. 保留临时文件（用户可通过输出目录查看完整结果）
        print(f"[DeepseekOCR] 输出文件保留在: {out_path}")
        
        # 最终输出检查
        print(f"[DeepseekOCR] === 输出摘要 ===")
        print(f"[DeepseekOCR] 文本输出: {len(text_result)} 字符")
        print(f"[DeepseekOCR] 图像输出形状: {vis_image.shape if hasattr(vis_image, 'shape') else 'unknown'}")
        print(f"[DeepseekOCR] 图像输出类型: {type(vis_image)}")
        
        return (text_result, vis_image)

# ---------- 节点注册 ----------
NODE_CLASS_MAPPINGS = {
    "DeepSeekOCR_Load": DeepSeekOCR_Load,
    "DeepSeekOCR_Run":  DeepSeekOCR_Run,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepSeekOCR_Load": "DeepSeek OCR: Load Model",
    "DeepSeekOCR_Run":  "DeepSeek OCR: Run",
}
