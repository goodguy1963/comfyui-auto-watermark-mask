# ComfyUI Auto Watermark Mask

ComfyUI helper nodes for detecting visible watermarks, building a mask, and cleaning the detected region with OpenCV, Big-LaMa, template-aware reconstruction, or optional Diffusers inpainting.

This repository is intended to be shareable as a standalone custom node package.

The canonical install location is directly under `ComfyUI/custom_nodes/comfyui-auto-watermark-mask`.

## What this node is good at

This node works best when the watermark is still visible enough to detect and mask.

- readable text watermarks
- corner logos and stock-photo marks
- faint semi-transparent overlays that still have recoverable edges
- known repeated watermark families when you can provide a matching template image

It is not a perfect general remover for every possible watermark. The best results come from choosing the right detection mode and, for harder symbol-style marks, connecting a `watermark_template` image.

## ComfyUI user experience

The important controls now include hover tooltips inside ComfyUI. Hover over the input name or widget to see what it does and when to use it.

The most important idea for users is:

- start simple with automatic detection
- only switch to the heavier modes when the simple mask is wrong
- connect `watermark_template` when the same symbol or logo keeps appearing

`symbol_reference` is still accepted for older workflows, but `watermark_template` is the clearer user-facing name going forward.

## Included nodes

- `Auto Watermark Mask (OCR)` builds a mask from EasyOCR detections, CV2 analysis, or both.
- `Auto Watermark Mask (OCR)` also supports corner fallback masking for small corner logos that OCR and generic CV2 can miss.
- `Auto Watermark Mask (OCR)` includes `detail_detector=decomposition` for harder faint overlays.
- `Auto Watermark Mask (OCR)` includes `detail_detector=symbol_template` for known symbol or logo watermarks when a `watermark_template` image is connected.
- `Auto Watermark Remover` cleans the detected area with OpenCV or Big-LaMa.
- `Auto Watermark Remover` includes `symbol_reverse_blend`, a narrower method for dark semi-transparent symbol overlays when you have a matching `watermark_template`.
- `Auto Watermark Remover` includes `gemini_reverse_alpha`, an experimental special case for Gemini-style sparkle overlays.
- `Auto Watermark Remover` includes optional `diffusers_sd_inpaint` for harder semantic cleanup on faces, bodies, costumes, and character art.

## Setup guide

### Basic install

1. Clone this repository into `ComfyUI/custom_nodes/comfyui-auto-watermark-mask`.
2. Install the Python dependencies from `requirements.txt` into the same Python environment used by ComfyUI.
3. Restart ComfyUI.

For portable ComfyUI installs, keep a single copy of this repository directly under `custom_nodes` instead of maintaining a second duplicate checkout elsewhere.

### Optional setup for Diffusers inpainting

Install these packages in the same environment as ComfyUI:

- `diffusers`
- `transformers`
- `pillow`

Then set `diffusers_model_id` to either:

- a local Diffusers inpainting model folder
- or a Hugging Face model id such as `stable-diffusion-v1-5/stable-diffusion-inpainting`

## Model setup

`big_lama` tries to resolve `big-lama.pt` in your configured ComfyUI inpaint model paths first.

If the file is missing, the node automatically downloads `fashn-ai/LaMa/big-lama.pt` into the first available inpaint model directory.

Notes:

- The first OCR run may download EasyOCR language weights.
- Automatic Big-LaMa download is enabled by default and can be disabled with `COMFYUI_AUTO_WATERMARK_MASK_AUTO_DOWNLOAD=0`.
- `cv2_only` avoids EasyOCR startup cost and is the fastest mode.
- `detail_detector=decomposition` is slower than plain `cv2_only`, but can recover faint blended overlays that the simpler contour heuristic misses.
- `detail_detector=symbol_template` is slower than the default path and only makes sense when you already know the watermark shape you want to remove.
- `symbol_reverse_blend` is narrower than generic inpainting: it tries to reconstruct a dark semi-transparent symbol from the matched template region before using a very small cleanup mask.
- `diffusers_sd_inpaint` is much slower than OpenCV or Big-LaMa, but it is the better fallback when the mask sits on anatomy, costume detail, or other semantic character art that the simpler inpainting methods tend to smear.
- `opencv_telea` is usually the best first pass for small text watermarks.
- `big_lama` is better when the watermark is thicker or overlaps detailed texture.
- `gemini_reverse_alpha` is intentionally narrow and not a generic watermark cleanup mode.

## User guide

### Quick start for most images

Use `Auto Watermark Remover` and start with:

- `detection_mode`: `cv2_only`
- `region`: `edges` for border watermarks, `full_image` otherwise
- `inpaint_method`: `opencv_telea`

If the mask preview looks good but cleanup quality is weak:

- try `big_lama`
- or use `diffusers_sd_inpaint` when the watermark crosses a face, body, costume, or detailed illustration

### When to use OCR

Switch to `ocr_then_cv2` when the watermark is readable text or mixed text plus shape.

Use `ocr_only` only when you specifically want text detection without any CV2 fallback.

### When to use a watermark template

Connect `watermark_template` when:

- the watermark is a symbol or logo instead of text
- the same overlay appears repeatedly across images
- auto-detection keeps locking onto the wrong region

Good template sources:

- a clean external reference image of the watermark
- a tight crop of the logo from another source image with similar scale

Avoid loose or noisy templates when possible. A crop from the same image can work, but it often includes hidden background detail that makes matching and reconstruction worse.

### Recommended template workflow

For detection only:

1. Use `Auto Watermark Mask (OCR)`.
2. Set `detail_detector` to `symbol_template`.
3. Connect `watermark_template`.
4. Start `symbol_match_threshold` around `0.45` to `0.55`.
5. Check the `mask_preview` output before doing removal.

For removal:

1. Use `Auto Watermark Remover`.
2. Connect the same `watermark_template`.
3. Start with `inpaint_method=opencv_telea` or `big_lama` if the mask is correct.
4. Switch to `symbol_reverse_blend` when the watermark is a dark semi-transparent symbol and a generic fill looks smeared.

### Corner-logo workflow

For stock-photo style corner marks:

- set `region` to `corners` or `edges`
- enable `corner_fallback=region_hint` if the automatic mask keeps missing the corner
- start with `padding` around `8` to `16`
- raise `dilate` slightly if the mask clips the logo edges

### Hard semantic workflow

Use `diffusers_sd_inpaint` when the watermark overlaps:

- faces
- skin
- anatomy
- costume details
- illustrated character features

Start with:

- `diffusers_strength`: `0.85` to `0.95`
- `diffusers_guidance_scale`: `3.5` to `5.5`
- `diffusers_steps`: `25` to `40`
- `diffusers_crop_padding`: `32` to `64`

## Recommended starting settings

- `detection_mode`: `cv2_only`
- `region`: `edges` for corner marks, `full_image` otherwise
- `padding`: `8` to `16`
- `dilate`: `4` to `10`
- `blur`: `3` to `7`
- `cv2_sensitivity`: `0.7` to `0.9`
- `corner_fallback`: `region_hint` when the watermark is likely pinned to a corner but auto-detection misses it
- `corner_width_ratio` / `corner_height_ratio`: start around `0.12` / `0.08` for stock-photo or avatar-style marks
- `detail_detector`: keep `none` by default, switch to `decomposition` when the watermark looks like a semi-transparent blended layer instead of crisp text
- `detail_detector`: switch to `symbol_template` when the hard case is a repeated symbol-like overlay rather than readable text
- `watermark_template`: provide a tight crop of the target symbol or logo on a similar background when using `detail_detector=symbol_template`
- `symbol_match_threshold`: start around `0.45` to `0.55`, raise it if the template starts latching onto lookalikes
- `inpaint_method`: `opencv_telea` for speed, `big_lama` for quality
- `inpaint_method`: switch to `symbol_reverse_blend` when the watermark is a dark semi-transparent symbol-like overlay and you have a good `watermark_template`
- `inpaint_method`: `gemini_reverse_alpha` only when the image likely contains the Gemini sparkle overlay near the bottom-right corner
- `inpaint_method`: switch to `diffusers_sd_inpaint` when the watermark crosses a face, body, costume, or illustrated character detail and the faster methods break structure
- `diffusers_strength`: start around `0.85` to `0.95` for watermark removal
- `diffusers_guidance_scale`: start around `3.5` to `5.5` to stay closer to the source image
- `diffusers_steps`: start around `25` to `40`
- `diffusers_crop_padding`: start around `32` to `64` so the model sees enough local context without redrawing the whole image

## Performance-focused behavior

This version keeps the same workflow shape but reduces the expensive parts of the pipeline:

- Detection work is automatically downscaled when the input image is very large, then mapped back to the original resolution mask.
- EasyOCR readers are cached per language and GPU setting.
- Big-LaMa inference runs once per ComfyUI image batch instead of one forward pass per image.
- Big-LaMa is skipped entirely when the batch has no detected mask pixels.
- Missing `big-lama.pt` can be fetched automatically from Hugging Face on first use.

## Research basis

The implementation direction here matches the common public pattern in other watermark removers:

- mask detection first, then inpaint
- optional OCR when text is explicit
- LaMa for higher-quality fills
- auto-download or local caching expectations for OCR and model weights

That is still the practical open-source frontier for generic visible watermark removal in 2025 and 2026. Stronger benchmark papers exist, but the broadly reusable public implementations are still mostly detector-plus-inpainting systems, or special-case removers for one known overlay family.

The biggest gap in the original local node was not missing features, but avoidable overhead in large-image detection and per-image LaMa execution.

Implementation notes from external watermark removers:

- `ComfyUI-SimpleWatermarkRemover` keeps the pipeline very small: manual mask input, pad for LaMa, auto-download the model, then run a direct inpaint pass.
- `WaterMarkRemover_ComfyUI` follows a similarly short path and also auto-downloads the model when missing.
- `santifer/watermark-remover` adds a practical upgrade for stock-photo style cases: an explicit corner fallback before LaMa or OpenCV cleanup.
- `IOPaint` shows the more scalable pattern: treat model lookup and switching as first-class concerns, keep model discovery centralized, and avoid repeated heavy initialization.
- `wiltodelta/remove-ai-watermarks` is stronger than most current tools, but only because it special-cases the Gemini sparkle overlay with template-style detection and inverse alpha removal. That is not a generic replacement for this node.

This node intentionally combines the useful parts of those approaches without copying their code:

- automatic model bootstrap for fresh installs
- cached OCR readers and cached inpaint models
- automatic mask generation instead of manual-mask-only flows
- batched Big-LaMa inference and downscaled detection for better runtime efficiency
- optional corner fallback masking for the common small-logo corner case
- an opt-in decomposition-style detector for tougher blended overlays
- an opt-in symbol-template detector for known logo or symbol shapes that OCR is not suited to
- a symbol-specific reverse-blend remover for semi-transparent matched symbol overlays
- a separate experimental Gemini-style sparkle remover path for the one overlay family that benefits from reverse-alpha reconstruction
- an optional crop-aware Diffusers inpaint path for the harder semantic cases where texture-only fill models are not enough

## Benchmark

Run the included benchmark harness to compare the optimized paths against small legacy reference implementations:

```powershell
python benchmark\benchmark_node.py --repeat 30
```

The benchmark covers:

- full-resolution legacy CV2 detection vs current downscaled detection
- decomposition-style detection vs the legacy full-resolution CV2 path
- legacy per-image LaMa loop vs current batched LaMa path
- a synthetic Gemini-style sparkle case that measures detector runtime, remover runtime, and reconstruction error

You can also point the harness at a folder of real images:

```powershell
python benchmark\benchmark_node.py --repeat 5 --input-dir .\benchmark_samples --input-limit 20
```

That real-sample mode reports mean runtime, hit rate, and average mask coverage for:

- `cv2_only`
- `detail_detector=decomposition`
- `detail_detector=symbol_template` when you provide a matching reference symbol
- `gemini_reverse_alpha`

The synthetic runs use generated inputs and a lightweight dummy LaMa model, so they are intended for regression and relative speed checks rather than absolute quality scoring. The real-sample mode is the path to use when tuning detector defaults on your own image set.

## Manager And Registry Notes

- `requirements.txt` is included so ComfyUI-Manager can install Python dependencies.
- `node_list.json` is included so the node pack can be indexed even if static scanning changes.
- `examples/auto_watermark_remover_basic.json` provides a minimal example workflow for the node.
- The repo includes `tool.comfy` metadata in `pyproject.toml`.
- The current `PublisherId` is set to `goodguy1963` as the intended registry id. If you create a different Comfy Registry publisher, update that field before publishing.

## Registry Publishing

The repo now includes [.github/workflows/publish.yml](.github/workflows/publish.yml), which follows the same publish pattern used for the earlier ThinkingLLM registry setup:

- it runs manually with `workflow_dispatch`
- it also runs automatically when `pyproject.toml` changes on `main`
- it uses `Comfy-Org/publish-node-action@v1`
- it reads the registry token from the GitHub Actions secret `REGISTRY_ACCESS_TOKEN`

To make the action actually publish to ComfyUI-Manager and the Comfy Registry, you still need these prerequisites:

1. Create a publisher at `https://registry.comfy.org/`.
2. Confirm the publisher id matches `tool.comfy.PublisherId` in `pyproject.toml`.
3. Add a GitHub repository secret named `REGISTRY_ACCESS_TOKEN` with a Comfy Registry publishing API key for that publisher.
4. Bump the semantic version in `pyproject.toml` when you want a new registry release, then push to `main` or run the workflow manually.

Without the publisher and secret, the workflow file is correct but publishing will fail at runtime.

## License

MIT
