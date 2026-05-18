# ComfyUI Auto Watermark Mask

ComfyUI helper nodes for finding text-like watermarks and removing them directly with OpenCV or Big-LaMa.

This repository is intended to be shareable as a standalone custom node package.

The canonical install location is directly under `ComfyUI/custom_nodes/comfyui-auto-watermark-mask`.

## What is included

- `Auto Watermark Mask (OCR)` builds a mask from EasyOCR detections with a CV2 fallback.
- `Auto Watermark Remover` can clean the detected area with OpenCV or Big-LaMa.

## Performance-focused behavior

This version keeps the same workflow shape but reduces the expensive parts of the pipeline:

- Detection work is automatically downscaled when the input image is very large, then mapped back to the original resolution mask.
- EasyOCR readers are cached per language and GPU setting.
- Big-LaMa inference runs once per ComfyUI image batch instead of one forward pass per image.
- Big-LaMa is skipped entirely when the batch has no detected mask pixels.
- Missing `big-lama.pt` can be fetched automatically from Hugging Face on first use.

## Installation

1. Clone this repository into `ComfyUI/custom_nodes/comfyui-auto-watermark-mask`.
2. Install the Python dependencies from `requirements.txt` if they are not already present.
3. Restart ComfyUI.

For your own portable setup, keep only one copy of this repository and place that git checkout directly in `custom_nodes`. Avoid maintaining a second copy elsewhere in the workspace.

## Model setup

`big_lama` now tries to resolve `big-lama.pt` in your configured ComfyUI inpaint model paths first.

If the file is missing, the node will automatically download `fashn-ai/LaMa/big-lama.pt` into the first available inpaint model directory.

Notes:

- The first OCR run may download EasyOCR language weights.
- Automatic Big-LaMa download is enabled by default and can be disabled with `COMFYUI_AUTO_WATERMARK_MASK_AUTO_DOWNLOAD=0`.
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

## Benchmark

Run the included benchmark harness to compare the optimized paths against small legacy reference implementations:

```powershell
python benchmark\benchmark_node.py --repeat 30
```

The benchmark covers:

- full-resolution legacy CV2 detection vs current downscaled detection
- legacy per-image LaMa loop vs current batched LaMa path

It uses synthetic inputs and a lightweight dummy LaMa model, so it is intended for regression and relative speed checks rather than absolute quality scoring.

## Manager And Registry Notes

- `requirements.txt` is included so ComfyUI-Manager can install Python dependencies.
- `node_list.json` is included so the node pack can be indexed even if static scanning changes.
- `examples/auto_watermark_remover_basic.json` provides a minimal example workflow for the node.
- The repo includes `tool.comfy` metadata in `pyproject.toml`.
- The current `PublisherId` is set to `goodguy1963` as the intended registry id. If you create a different Comfy Registry publisher, update that field before publishing.

## License

MIT
