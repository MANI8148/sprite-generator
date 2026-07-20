from PIL import Image
import numpy as np
from typing import Optional, Tuple

_REALESRGAN_AVAILABLE = False
_UPSCALER_MODEL = None


def is_available() -> bool:
    return _REALESRGAN_AVAILABLE


def load_model(
    model_name: str = "RealESRGAN_x4plus",
    scale: int = 4,
    device: Optional[str] = None,
) -> bool:
    global _REALESRGAN_AVAILABLE, _UPSCALER_MODEL
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        upscaler = RealESRGANer(
            scale=scale,
            model_path=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=device == "cuda",
            device=device,
        )
        _UPSCALER_MODEL = upscaler
        _REALESRGAN_AVAILABLE = True
        return True
    except Exception:
        _REALESRGAN_AVAILABLE = False
        _UPSCALER_MODEL = None
        return False


def upscale_with_realesrgan(
    image: Image.Image,
    scale: int = 4,
    model_name: str = "RealESRGAN_x4plus",
) -> Image.Image:
    if not _REALESRGAN_AVAILABLE:
        load_model(model_name=model_name, scale=scale)

    if not _REALESRGAN_AVAILABLE or _UPSCALER_MODEL is None:
        w, h = image.size
        return image.resize((w * scale, h * scale), Image.LANCZOS)

    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3].copy()
    rgb = Image.fromarray(arr[:, :, :3], "RGB")

    try:
        output, _ = _UPSCALER_MODEL.enhance(np.array(rgb), outscale=scale)
        result_arr = np.array(Image.fromarray(output).convert("RGBA"))
        result_arr[:, :, 3] = np.array(
            Image.fromarray(alpha).resize(
                (result_arr.shape[1], result_arr.shape[0]), Image.NEAREST
            )
        )
        return Image.fromarray(result_arr)
    except Exception:
        w, h = image.size
        return image.resize((w * scale, h * scale), Image.LANCZOS)
