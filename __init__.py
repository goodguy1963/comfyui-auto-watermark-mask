"""
{
	"name": "comfyui-auto-watermark-mask",
	"description": "ComfyUI custom nodes for automatic watermark masking and direct removal with OpenCV, EasyOCR, and Big-LaMa.",
	"author": "goodguy1963",
	"version": "0.2.3",
	"url": "https://github.com/goodguy1963/comfyui-auto-watermark-mask",
	"category": "image"
}
"""

from .auto_watermark_mask import AutoWatermarkMaskOCR, AutoWatermarkRemover, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["AutoWatermarkMaskOCR", "AutoWatermarkRemover", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
