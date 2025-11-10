# model_manager.py
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass
import os, time, json, logging, warnings, subprocess, shutil

from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModel
import torch

from .config import (
    REPO_ID, MODELS_HOME, HF_CACHE, REVISION, AUTODOWNLOAD, local_snapshot_dir, LoadConfig, LOG_DIR
)

# 抑制 huggingface_hub 的 hf_xet 警告
warnings.filterwarnings("ignore", message=".*hf_xet.*")
warnings.filterwarnings("ignore", message=".*Xet Storage.*")

# 配置日志
LOG_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "deepseek_ocr.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

_LOCK_NAME = ".download.lock"


def _format_bytes(size: int) -> str:
    if size <= 0:
        return "0B"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    idx = 0
    value = float(size)
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    if idx == 0:
        return f"{int(value)}{units[idx]}"
    return f"{value:.2f}{units[idx]}"


def _preferred_workers() -> int:
    env_value = os.getenv("DPSK_DOWNLOAD_WORKERS")
    if env_value:
        try:
            workers = int(env_value)
            if workers > 0:
                return workers
        except ValueError:
            logger.warning(f"DPSK_DOWNLOAD_WORKERS 非法：{env_value}")
    cpu = os.cpu_count() or 4
    return max(4, min(32, cpu * 2))


def _repo_snapshot_meta(repo_id: str, revision: Optional[str]):
    try:
        api = HfApi()
        info = api.repo_info(
            repo_id=repo_id,
            revision=revision,
            repo_type="model",
            files_metadata=True,
        )
        total_bytes = sum((s.size or 0) for s in info.siblings if getattr(s, "size", None))
        files = len([s for s in info.siblings if getattr(s, "size", None)])
        return info, total_bytes, files
    except Exception as exc:
        logger.warning(f"无法获取仓库元数据：{exc}")
        return None, None, None


def _download_with_cli(repo_id: str, local_dir: Path, revision: Optional[str], max_workers: int) -> str:
    """使用 huggingface-cli 下载模型，实时显示原生进度信息"""
    cli_path = shutil.which("huggingface-cli")
    if not cli_path:
        raise RuntimeError(
            "未找到 huggingface-cli 命令！\n"
            "请安装 huggingface_hub：\n"
            "  pip install huggingface_hub[cli]\n"
            "或：\n"
            "  pip install -U huggingface_hub"
        )

    cmd = [
        cli_path,
        "download",
        repo_id,
        "--repo-type",
        "model",
        "--local-dir",
        str(local_dir),
        "--cache-dir",
        str(HF_CACHE),
        "--resume-download",
        "--max-workers",
        str(max_workers),
    ]
    if revision:
        cmd.extend(["--revision", revision])

    env = os.environ.copy()
    env.setdefault("HF_HOME", str(HF_CACHE))
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")

    logger.info(f"使用 huggingface-cli 下载 ({max_workers} workers)...")
    logger.info("=" * 70)
    
    # 直接继承当前终端，实时显示完整输出
    return_code = subprocess.call(cmd, env=env)

    if return_code != 0:
        raise RuntimeError(f"huggingface-cli 下载失败 (exit code: {return_code})")

    # 获取 commit SHA
    try:
        info = HfApi().repo_info(repo_id=repo_id, revision=revision, repo_type="model")
        return info.sha
    except Exception:
        return revision or "main"

def _is_ready(local_dir: Path) -> bool:
    """
    判断快照目录是否已完整：至少要有 config/tokenizer 和任意 .safetensors
    """
    if not local_dir.exists():
        return False
    need = ["config.json", "tokenizer.json"]
    has_weights = any(p.suffix == ".safetensors" for p in local_dir.glob("*.safetensors"))
    return all((local_dir / n).exists() for n in need) and has_weights

class _FileLock:
    """
    简易文件锁：创建独占文件，存在则轮询等待（适合首次下载并发场景）
    """
    def __init__(self, lock_path: Path, interval: float = 0.5):
        self.lock_path = lock_path
        self.interval = interval
        self._fd = None

    def __enter__(self):
        while True:
            try:
                self._fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self._fd, b"1")
                return self
            except FileExistsError:
                time.sleep(self.interval)

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._fd is not None:
                os.close(self._fd)
            if self.lock_path.exists():
                os.remove(self.lock_path)
        except Exception:
            pass

def _write_manifest(local_dir: Path, commit: Optional[str]):
    data = {"repo_id": REPO_ID, "revision": commit, "ts": int(time.time())}
    (local_dir / "manifest.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def ensure_weights_local(repo_id: str = REPO_ID) -> Path:
    """
    确保 DeepSeek-OCR 权重被下载到：
        ComfyUI/models/deepseek-ocr/<org_repo_下划线替换>/
    - 未就绪且 AUTODOWNLOAD=0：抛出指引性错误
    - 未就绪且 AUTODOWNLOAD=1：自动下载（带文件锁）
    返回本地快照目录 Path
    """
    local = local_snapshot_dir(repo_id)  # e.g. .../models/deepseek-ocr/deepseek-ai_DeepSeek-OCR
    local.mkdir(parents=True, exist_ok=True)

    if _is_ready(local):
        return local

    if not AUTODOWNLOAD:
        raise RuntimeError(
            f"[DeepseekOCR] 权重缺失：{local}\n"
            f"自动下载已禁用（DPSK_AUTODOWNLOAD=0）。\n"
            f"解决方案：\n"
            f"1. 删除环境变量 DPSK_AUTODOWNLOAD（启用自动下载）\n"
            f"2. 或手动运行 tools\\download_weights.py 预下载"
        )

    lock = MODELS_HOME / _LOCK_NAME
    logger.info(f"准备下载权重到: {local}")
    
    with _FileLock(lock):
        # 可能别的进程已经下好了，再检查一次
        if _is_ready(local):
            logger.info("其他进程已完成下载")
            return local

        logger.info(f"开始从 HuggingFace 下载: {repo_id}")
        logger.info(f"目标目录: {local}")
        logger.info("下载模型权重（约 8-10GB，首次下载需要时间，请耐心等待）")
        
        try:
            # 获取仓库元数据（可选）
            info, total_bytes, total_files = _repo_snapshot_meta(repo_id, REVISION)
            if total_files:
                logger.info(f"仓库共有 {total_files} 个文件待下载/校验")
            if total_bytes:
                logger.info(f"预估总下载量：{_format_bytes(total_bytes)}")

            max_workers = _preferred_workers()
            resolved_revision = REVISION or (info.sha if info else None)

            # 使用 huggingface-cli 下载
            commit = _download_with_cli(repo_id, local, resolved_revision, max_workers)

            # 写入清单文件
            _write_manifest(local, commit)
            logger.info("=" * 70)
            logger.info(f"✓ 下载完成！commit: {commit}")
            return local
            
        except Exception as e:
            logger.error(f"✗ 下载失败: {e}")
            raise RuntimeError(
                f"权重下载失败！\n"
                f"建议：\n"
                f"1. 检查网络连接或设置 HF_ENDPOINT 镜像\n"
                f"2. 设置 HF_TOKEN（如果仓库需要认证）\n"
                f"3. 确保已安装 huggingface_hub[cli]\n"
                f"4. 手动下载到 {local}\n"
                f"原始错误: {e}"
            ) from e


# ========== OCRHandle & 模型加载 ==========

@dataclass
class OCRHandle:
    """模型句柄：包含 tokenizer、model、config"""
    tokenizer: any
    model: any
    cfg: LoadConfig
    
    def infer_one(self, image_path: str, prompt: str, preset: dict, save_results: bool, output_path: str = None):
        """
        便捷推理接口，内部转发到 model.infer(...)
        preset: resolver.map_resolution() 返回的字典
        
        策略：使用 eval_mode=True 直接获取文本返回值
        """
        # 使用 eval_mode=True 获取文本返回值
        result = self.model.infer(
            tokenizer=self.tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=output_path or ".",
            base_size=preset.get("base_size", 1024),
            image_size=preset.get("image_size", 640),
            crop_mode=preset.get("crop_mode", True),
            save_results=False,  # 不生成文件，直接返回
            test_compress=False,  # 不打印压缩率
            eval_mode=True,  # 启用 eval_mode，返回文本
        )
        
        return result  # 返回文本
    
    @property
    def device(self) -> str:
        return self.cfg.device
    
    @property
    def dtype(self) -> str:
        return self.cfg.dtype
    
    @property
    def attn_impl(self) -> str:
        return self.cfg.attn_impl


# 全局缓存：key=LoadConfig, value=OCRHandle
_GLOBAL_HANDLES: Dict[LoadConfig, OCRHandle] = {}


def load_model(cfg: LoadConfig) -> OCRHandle:
    """
    加载模型，支持缓存与降级策略：
    - attn_impl: 优先 sdpa，flash_attention_2 失败时回退
    - dtype: bf16 → fp16 → fp32 逐级降级
    返回 OCRHandle
    """
    # 检查缓存
    if cfg in _GLOBAL_HANDLES:
        logger.info(f"[DeepseekOCR] 命中缓存：{cfg.local_dir} | {cfg.device} | {cfg.dtype} | {cfg.attn_impl}")
        return _GLOBAL_HANDLES[cfg]
    
    logger.info(f"[DeepseekOCR] 开始加载模型：{cfg.local_dir}")
    logger.info(f"  device={cfg.device}, dtype={cfg.dtype}")
    
    # 注意：DeepSeek-OCR 模型内部会自动选择 attention 实现
    # 如果安装了 flash-attn，会自动使用；否则使用 eager attention
    
    # 1. 加载 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.local_dir,
            trust_remote_code=True
        )
        logger.info("[DeepseekOCR] tokenizer 加载成功")
    except Exception as e:
        logger.error(f"[DeepseekOCR] tokenizer 加载失败: {e}")
        raise RuntimeError(f"Tokenizer 加载失败: {e}") from e
    
    # 3. 加载 model，带 dtype 降级
    dtype_list = []
    if cfg.dtype == "bf16":
        dtype_list = [torch.bfloat16, torch.float16, torch.float32]
    elif cfg.dtype == "fp16":
        dtype_list = [torch.float16, torch.float32]
    else:
        dtype_list = [torch.float32]
    
    # 环境检查
    cuda_available = torch.cuda.is_available()
    if cfg.device.startswith("cuda") and not cuda_available:
        raise RuntimeError(
            "[DeepseekOCR] CUDA 不可用！\n"
            f"请求设备: {cfg.device}\n"
            f"CUDA 可用: {cuda_available}\n"
            "请检查：\n"
            "1. 是否安装了正确的 PyTorch CUDA 版本\n"
            "2. 显卡驱动是否正常\n"
            "3. 或改用 device='cpu'（速度较慢）"
        )
    
    if cuda_available and cfg.device.startswith("cuda"):
        device_id = 0 if cfg.device == "cuda" else int(cfg.device.split(":")[-1])
        try:
            props = torch.cuda.get_device_properties(device_id)
            total_mem = props.total_memory / (1024**3)  # GB
            logger.info(f"[DeepseekOCR] GPU: {props.name}, 显存: {total_mem:.2f} GB")
        except Exception:
            pass
    
    model = None
    final_dtype = None
    last_error = None
    
    for dtype in dtype_list:
        try:
            logger.info(f"[DeepseekOCR] 尝试加载模型 dtype={dtype}")
            
            # DeepSeek-OCR 不支持通过 _attn_implementation 参数指定 attention
            # 模型会使用默认的 eager attention 实现
            # 如果需要 flash_attention_2，需要安装 flash-attn 后模型会自动检测使用
            model = AutoModel.from_pretrained(
                cfg.local_dir,
                trust_remote_code=True,
                torch_dtype=dtype,
                use_safetensors=True,
            )
            
            logger.info(f"[DeepseekOCR] 模型使用默认 attention 实现（eager）")
            model.eval()
            logger.info(f"[DeepseekOCR] 模型已加载，正在移动到 {cfg.device}...")
            model.to(cfg.device)
            final_dtype = dtype
            logger.info(f"[DeepseekOCR] ✓ 模型加载成功：dtype={dtype}, device={cfg.device}")
            break
        except Exception as e:
            last_error = e
            error_str = str(e)
            logger.error(f"[DeepseekOCR] ✗ dtype={dtype} 加载失败")
            logger.error(f"  错误类型: {type(e).__name__}")
            logger.error(f"  错误信息: {error_str[:200]}")
            
            # 检查是否是显存不足
            if "out of memory" in error_str.lower() or "oom" in error_str.lower():
                logger.error("  >> 诊断: 显存不足（OOM）")
            elif "cuda" in error_str.lower() and "driver" in error_str.lower():
                logger.error("  >> 诊断: CUDA 驱动问题")
            elif "trust_remote_code" in error_str.lower():
                logger.error("  >> 诊断: trust_remote_code 相关问题，检查模型文件完整性")
            
            # 如果是最后一个 dtype，不 continue
            if dtype == dtype_list[-1]:
                break
            else:
                logger.warning(f"  尝试降级到下一个 dtype...")
                continue
    
    if model is None:
        # 构建详细错误信息
        error_details = f"\n最后一次错误: {type(last_error).__name__}: {str(last_error)[:300]}" if last_error else ""
        
        raise RuntimeError(
            "[DeepseekOCR] 模型加载失败！所有 dtype 尝试均失败。\n"
            f"尝试的 dtype: {[str(d) for d in dtype_list]}\n"
            f"{error_details}\n\n"
            "建议排查：\n"
            "1. 显存不足：\n"
            "   - 关闭其他占用显存的程序\n"
            "   - 使用更小的 dtype (fp16)\n"
            "   - 或在 Run 节点选择 Tiny/Small 分辨率\n"
            "2. 模型文件损坏：\n"
            f"   - 删除 {cfg.local_dir}\n"
            "   - 重新下载模型\n"
            "3. PyTorch/CUDA 问题：\n"
            "   - 检查 torch.cuda.is_available()\n"
            "   - 更新显卡驱动\n"
            f"4. 查看完整日志: {LOG_DIR / 'deepseek_ocr.log'}"
        )
    
    # 4. 创建 Handle 并缓存
    handle = OCRHandle(tokenizer=tokenizer, model=model, cfg=cfg)
    _GLOBAL_HANDLES[cfg] = handle
    logger.info(f"[DeepseekOCR] 模型已缓存，最终配置：dtype={final_dtype}")
    
    return handle
