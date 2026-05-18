# ComfyUI Auto Watermark Mask

ComfyUI helper nodes for finding text-like watermarks and removing them directly with OpenCV or Big-LaMa.

## What is included

- `Auto Watermark Mask (OCR)` builds a mask from EasyOCR detections with a CV2 fallback.
- `Auto Watermark Remover` can clean the detected area with OpenCV or Big-LaMa.

## Performance-focused behavior

This version keeps the same workflow shape but reduces the expensive parts of the pipeline:

- Detection work is automatically downscaled when the input image is very large, then mapped back to the original resolution mask.
- EasyOCR readers are cached per language and GPU setting.
- Big-LaMa inference runs once per ComfyUI image batch instead of one forward pass per image.
- Big-LaMa is skipped entirely when the batch has no detected mask pixels.

## Installation

1. Clone this repository into `ComfyUI/custom_nodes/comfyui-auto-watermark-mask`.
2. Install the Python dependencies from `requirements.txt` if they are not already present.
3. Restart ComfyUI.

## Model setup

For `big_lama`, place `big-lama.pt` in `ComfyUI/models/inpaint/big-lama.pt`.

Notes:

- The first OCR run may download EasyOCR language weights.
- `cv2_only` avoids EasyOCR startup cost and is the fastest mode.
- `opencv_telea` is usually the best first pass for small text watermarks.
- `big_lama` is better when the watermark is thicker or overlaps detailed texture.

## Recommended starting settings

- `detection_mode`: `cv2_only`
- `region`: `edges` for corner marks, `full_image` otherwise
- `padding`: `8` to `16`
- `dilate`: `4` to `10`
- `blur`: `3` to `7`
- `cv2_sensitivity`: `0.7` to `0.9`
- `inpaint_method`: `opencv_telea` for speed, `big_lama` for quality

## Research basis

The implementation direction here matches the common public pattern in other watermark removers:

- mask detection first, then inpaint
- optional OCR when text is explicit
- LaMa for higher-quality fills
- auto-download or local caching expectations for OCR and model weights

The biggest gap in the original local node was not missing features, but avoidable overhead in large-image detection and per-image LaMa execution.
