import os
import threading

import cv2
import numpy as np
import torch
import torch.nn.functional as F


_READER_CACHE = {}
_READER_LOCK = threading.Lock()
_INPAINT_MODEL_CACHE = {}
_INPAINT_MODEL_LOCK = threading.Lock()
_DIFFUSERS_PIPELINE_CACHE = {}
_DIFFUSERS_PIPELINE_LOCK = threading.Lock()
_GEMINI_TEMPLATE_CACHE = {}
_GEMINI_TEMPLATE_LOCK = threading.Lock()
_DETECTION_MAX_SIDE = 1280
_LAMA_TARGET_SIZE = 256
_BIG_LAMA_MODEL_NAME = "big-lama.pt"
_BIG_LAMA_REPO_ID = "fashn-ai/LaMa"
_AUTO_DOWNLOAD_ENV = "COMFYUI_AUTO_WATERMARK_MASK_AUTO_DOWNLOAD"
_GEMINI_SMALL_SIZE = 48
_GEMINI_LARGE_SIZE = 96
_GEMINI_SMALL_MARGIN = 32
_GEMINI_LARGE_MARGIN = 64
_DIFFUSERS_DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
_DIFFUSERS_DEFAULT_PROMPT = "clean image, preserve the original subject and style, remove watermark and text"
_DIFFUSERS_DEFAULT_NEGATIVE_PROMPT = "watermark, text, logo, signature, artifact, distortion, deformed anatomy, blurry face"
_DIFFUSERS_MAX_SIDE = 768
_DIFFUSERS_MIN_SIDE = 128


def _tensor_to_uint8(image):
    array = image.detach().cpu().numpy()
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def _diffusers_available():
    try:
        import diffusers  # noqa: F401
        import transformers  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception:
        return False
    return True


def _get_inpaint_methods():
    methods = ["opencv_telea", "opencv_navier_stokes", "big_lama", "symbol_reverse_blend", "gemini_reverse_alpha"]
    if _diffusers_available():
        methods.append("diffusers_sd_inpaint")
    return methods


def _get_reader(language_codes, gpu):
    langs = tuple(code.strip() for code in language_codes.split(",") if code.strip()) or ("en",)
    key = (langs, bool(gpu))
    with _READER_LOCK:
        reader = _READER_CACHE.get(key)
        if reader is not None:
            return reader
        import easyocr

        reader = easyocr.Reader(list(langs), gpu=bool(gpu), verbose=False)
        _READER_CACHE[key] = reader
        return reader


def _prepare_detection_rgb(rgb):
    height, width = rgb.shape[:2]
    max_side = max(height, width)
    if max_side <= _DETECTION_MAX_SIDE:
        return rgb, 1.0

    scale = _DETECTION_MAX_SIDE / float(max_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def _region_filter(width, height, region):
    if region == "full_image":
        return lambda _cx, _cy: True
    if region == "corners":
        x_margin = width * 0.35
        y_margin = height * 0.35
        return lambda cx, cy: (cx <= x_margin or cx >= width - x_margin) and (cy <= y_margin or cy >= height - y_margin)
    if region == "edges":
        x_margin = width * 0.25
        y_margin = height * 0.25
        return lambda cx, cy: cx <= x_margin or cx >= width - x_margin or cy <= y_margin or cy >= height - y_margin
    return lambda _cx, _cy: True


def _mask_ocr_results(mask, results, keep_region, min_confidence, padding, width, height, scale):
    detected_text = []
    for box, text, confidence in results:
        if float(confidence) < float(min_confidence):
            continue
        points = np.array(box, dtype=np.float32)
        if scale != 1.0:
            points /= float(scale)
        cx = float(points[:, 0].mean())
        cy = float(points[:, 1].mean())
        if not keep_region(cx, cy):
            continue

        x_min = max(int(np.floor(points[:, 0].min())) - int(padding), 0)
        y_min = max(int(np.floor(points[:, 1].min())) - int(padding), 0)
        x_max = min(int(np.ceil(points[:, 0].max())) + int(padding), width - 1)
        y_max = min(int(np.ceil(points[:, 1].max())) + int(padding), height - 1)
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, thickness=-1)
        if text:
            detected_text.append(str(text))
    return detected_text


def _mask_cv2_text_like_regions(mask, detection_rgb, keep_region, padding, sensitivity, width, height, scale):
    detection_height, detection_width = detection_rgb.shape[:2]
    gray = cv2.cvtColor(detection_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    lower = max(10, int(90 - float(sensitivity) * 70))
    upper = max(lower + 20, int(190 - float(sensitivity) * 90))
    edges = cv2.Canny(blur, lower, upper)

    kernel_w = max(3, int(round(detection_width * 0.012)))
    kernel_h = max(3, int(round(detection_height * 0.008)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    joined = cv2.dilate(edges, kernel, iterations=2)
    joined = cv2.morphologyEx(joined, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _hierarchy = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(detection_width * detection_height)
    min_area = max(12.0, image_area * 0.00003)
    max_area = image_area * 0.12
    region_count = 0

    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area = float(box_width * box_height)
        if area < min_area or area > max_area:
            continue
        if box_width < 6 or box_height < 5:
            continue
        aspect = box_width / float(max(box_height, 1))
        if aspect < 0.15 or aspect > 35.0:
            continue
        scale_factor = 1.0 / float(scale)
        cx = (x + box_width * 0.5) * scale_factor
        cy = (y + box_height * 0.5) * scale_factor
        if not keep_region(cx, cy):
            continue

        x_min = max(int(np.floor(x * scale_factor)) - int(padding), 0)
        y_min = max(int(np.floor(y * scale_factor)) - int(padding), 0)
        x_max = min(int(np.ceil((x + box_width) * scale_factor)) + int(padding), width - 1)
        y_max = min(int(np.ceil((y + box_height) * scale_factor)) + int(padding), height - 1)
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, thickness=-1)
        region_count += 1

    return region_count


def _odd_kernel_size(value, minimum=3):
    size = max(int(round(value)), int(minimum))
    if size % 2 == 0:
        size += 1
    return size


def _project_detection_mask(mask, detection_mask, padding, width, height, scale):
    projected = detection_mask
    if scale != 1.0:
        projected = cv2.resize(projected, (width, height), interpolation=cv2.INTER_LINEAR)
    projected = ((projected > 0).astype(np.uint8) * 255)

    if int(padding) > 0 and projected.max() > 0:
        kernel_size = int(padding) * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        projected = cv2.dilate(projected, kernel, iterations=1)

    np.maximum(mask, projected, out=mask)


def _mask_decomposition_regions(mask, detection_rgb, keep_region, padding, sensitivity, width, height, scale):
    detection_height, detection_width = detection_rgb.shape[:2]
    gray = cv2.cvtColor(detection_rgb, cv2.COLOR_RGB2GRAY)

    large_kernel_w = _odd_kernel_size(detection_width * (0.02 + 0.015 * float(sensitivity)), minimum=5)
    large_kernel_h = _odd_kernel_size(detection_height * (0.015 + 0.01 * float(sensitivity)), minimum=5)
    small_kernel_w = _odd_kernel_size(large_kernel_w // 2, minimum=3)
    small_kernel_h = _odd_kernel_size(large_kernel_h // 2, minimum=3)

    large_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (large_kernel_w, large_kernel_h))
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (small_kernel_w, small_kernel_h))

    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, large_kernel)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, large_kernel)
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, small_kernel)
    score = np.maximum(np.maximum(blackhat, tophat), gradient)
    score = cv2.GaussianBlur(score, (3, 3), 0)

    threshold = max(18, int(round(34 - float(sensitivity) * 12)))
    edge_map = cv2.Canny(gray, max(10, threshold // 2), max(30, threshold * 3))
    binary = (score >= threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, small_kernel, iterations=2)
    binary = cv2.dilate(binary, small_kernel, iterations=1)

    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(detection_width * detection_height)
    min_area = max(20.0, image_area * 0.00002)
    max_area = image_area * 0.18
    region_count = 0
    detection_mask = np.zeros((detection_height, detection_width), dtype=np.uint8)

    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area = float(box_width * box_height)
        if area < min_area or area > max_area:
            continue
        if box_width < 6 or box_height < 6:
            continue
        if box_height > detection_height * 0.075:
            continue
        extent = area / float(max(cv2.contourArea(contour), 1.0))
        if extent > 4.5:
            continue
        aspect = box_width / float(max(box_height, 1))
        if aspect < 0.15 or aspect > 20.0:
            continue

        scale_factor = 1.0 / float(scale)
        cx = (x + box_width * 0.5) * scale_factor
        cy = (y + box_height * 0.5) * scale_factor
        if not keep_region(cx, cy):
            continue

        component_mask = np.zeros((detection_height, detection_width), dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], -1, 255, thickness=-1)
        component_pixels = max(1, int(np.count_nonzero(component_mask)))
        edge_density = float(np.count_nonzero((edge_map > 0) & (component_mask > 0))) / float(component_pixels)
        mean_score = float(np.asarray(cv2.mean(score, mask=component_mask)).reshape(-1)[0])
        if edge_density < 0.055:
            continue
        if mean_score < threshold * 1.1:
            continue
        np.maximum(detection_mask, component_mask, out=detection_mask)
        region_count += 1

    if region_count > 0:
        _project_detection_mask(mask, detection_mask, padding, width, height, scale)

    return region_count


def _mask_low_contrast_overlay_regions(mask, detection_rgb, keep_region, padding, sensitivity, width, height, scale):
    detection_height, detection_width = detection_rgb.shape[:2]
    gray = cv2.cvtColor(detection_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lab = cv2.cvtColor(detection_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = cv2.magnitude(lab[..., 1] - 128.0, lab[..., 2] - 128.0)

    sigma = max(10.0, max(detection_width, detection_height) * (0.03 + 0.025 * float(sensitivity)))
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    residual = gray - background

    neutral_threshold = 19.0 + (1.0 - float(sensitivity)) * 6.0
    dark_background_threshold = 102.0 + (1.0 - float(sensitivity)) * 12.0
    residual_threshold = max(5.5, 9.5 - float(sensitivity) * 3.0)
    residual_upper_threshold = 22.0 + float(sensitivity) * 8.0

    binary = (
        (residual >= residual_threshold)
        & (residual <= residual_upper_threshold)
        & (chroma <= neutral_threshold)
        & (background <= dark_background_threshold)
    ).astype(np.uint8) * 255

    cleanup_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            _odd_kernel_size(min(detection_width, detection_height) * (0.015 + 0.008 * float(sensitivity)), minimum=5),
            _odd_kernel_size(min(detection_width, detection_height) * (0.015 + 0.008 * float(sensitivity)), minimum=5),
        ),
    )
    merge_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            _odd_kernel_size(min(detection_width, detection_height) * (0.022 + 0.012 * float(sensitivity)), minimum=9),
            _odd_kernel_size(min(detection_width, detection_height) * (0.022 + 0.012 * float(sensitivity)), minimum=9),
        ),
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cleanup_kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, merge_kernel, iterations=1)

    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(detection_width * detection_height)
    min_area = max(200.0, image_area * 0.0012)
    max_area = image_area * 0.16
    min_width = max(32.0, detection_width * 0.18)
    min_height = max(40.0, detection_height * 0.12)
    region_count = 0
    detection_mask = np.zeros((detection_height, detection_width), dtype=np.uint8)

    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        box_area = float(box_width * box_height)
        if box_area < min_area or box_area > max_area:
            continue
        if box_width < min_width or box_height < min_height:
            continue
        if x <= 0 or y <= 0 or x + box_width >= detection_width or y + box_height >= detection_height:
            continue

        aspect = box_width / float(max(box_height, 1))
        if aspect < 0.12 or aspect > 4.0:
            continue

        contour_area = float(max(cv2.contourArea(contour), 1.0))
        coverage = contour_area / float(max(box_area, 1.0))
        if coverage < 0.08 or coverage > 0.72:
            continue

        scale_factor = 1.0 / float(scale)
        cx = (x + box_width * 0.5) * scale_factor
        cy = (y + box_height * 0.5) * scale_factor
        if not keep_region(cx, cy):
            continue

        component_mask = np.zeros((detection_height, detection_width), dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], -1, 255, thickness=-1)

        mean_residual = float(np.asarray(cv2.mean(residual, mask=component_mask)).reshape(-1)[0])
        mean_background = float(np.asarray(cv2.mean(background, mask=component_mask)).reshape(-1)[0])
        mean_chroma = float(np.asarray(cv2.mean(chroma, mask=component_mask)).reshape(-1)[0])
        if mean_residual < residual_threshold * 1.2:
            continue
        if mean_residual > residual_upper_threshold * 0.9:
            continue
        if mean_background > dark_background_threshold:
            continue
        if mean_chroma > neutral_threshold + 2.5:
            continue

        np.maximum(detection_mask, component_mask, out=detection_mask)
        region_count += 1

    if region_count > 0:
        _project_detection_mask(mask, detection_mask, padding, width, height, scale)

    return region_count


def _get_batch_reference_image(symbol_reference, batch_index):
    if symbol_reference is None:
        return None
    if not torch.is_tensor(symbol_reference):
        return None
    if len(symbol_reference.shape) == 4:
        reference_index = min(int(batch_index), int(symbol_reference.shape[0]) - 1)
        return _tensor_to_uint8(symbol_reference[reference_index])
    return _tensor_to_uint8(symbol_reference)


def _resolve_template_input(watermark_template=None, symbol_reference=None):
    if watermark_template is not None:
        return watermark_template
    return symbol_reference


def _build_symbol_template(symbol_reference_rgb):
    if symbol_reference_rgb is None:
        return None

    rgb = symbol_reference_rgb
    height, width = rgb.shape[:2]
    if height < 8 or width < 8:
        return None

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    gradient = cv2.magnitude(
        cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3),
    )
    if float(gradient.max()) <= 0.0:
        return None
    gradient /= float(gradient.max())

    border_rows = min(4, max(1, height // 8))
    border_cols = min(4, max(1, width // 8))
    border_pixels = np.concatenate(
        [
            rgb[:border_rows, :, :].reshape(-1, 3),
            rgb[-border_rows:, :, :].reshape(-1, 3),
            rgb[:, :border_cols, :].reshape(-1, 3),
            rgb[:, -border_cols:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    border_mean = border_pixels.astype(np.float32).mean(axis=0)
    border_gray = float(cv2.cvtColor(border_pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32).mean())
    color_distance = np.linalg.norm(rgb.astype(np.float32) - border_mean[None, None, :], axis=2)
    foreground_hint = np.maximum(border_gray - gray.astype(np.float32), 0.0)

    positive_gradients = gradient[gradient > 0.0]
    positive_foreground = foreground_hint[foreground_hint > 0.0]
    gradient_threshold = 0.16 if positive_gradients.size == 0 else max(0.16, float(np.percentile(positive_gradients, 62)))
    color_threshold = max(8.0, float(np.percentile(color_distance, 68)))
    foreground_threshold = 6.0 if positive_foreground.size == 0 else max(6.0, float(np.percentile(positive_foreground, 58)))
    template_mask = (
        (gradient >= gradient_threshold)
        | (color_distance >= color_threshold)
        | (foreground_hint >= foreground_threshold)
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    template_mask = cv2.morphologyEx(template_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    template_mask = cv2.morphologyEx(template_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    template_mask = cv2.dilate(template_mask, kernel, iterations=1)

    points = cv2.findNonZero(template_mask)
    if points is None:
        return None

    x, y, crop_width, crop_height = cv2.boundingRect(points)
    pad = 2
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(width, x + crop_width + pad)
    y1 = min(height, y + crop_height + pad)

    cropped_gradient = gradient[y0:y1, x0:x1].astype(np.float32)
    cropped_mask = template_mask[y0:y1, x0:x1]
    cropped_gray = gray[y0:y1, x0:x1].astype(np.float32)
    if cropped_gradient.shape[0] < 8 or cropped_gradient.shape[1] < 8:
        return None

    masked_values = cropped_gradient[cropped_mask > 0]
    if masked_values.size == 0:
        return None

    foreground = np.maximum(border_gray - cropped_gray, 0.0)
    foreground = np.where(cropped_mask > 0, foreground, 0.0)
    if float(foreground.max()) > 0.0:
        foreground /= float(foreground.max())

    cropped_gradient = np.where(cropped_mask > 0, cropped_gradient, 0.0)
    max_value = float(masked_values.max())
    if max_value > 0.0:
        cropped_gradient /= max_value

    return {
        "gradient": cropped_gradient.astype(np.float32),
        "foreground": foreground.astype(np.float32),
        "mask": cropped_mask.astype(np.uint8),
        "width": int(cropped_gradient.shape[1]),
        "height": int(cropped_gradient.shape[0]),
    }


def _mask_symbol_template_regions(
    mask,
    detection_rgb,
    symbol_reference_rgb,
    keep_region,
    padding,
    width,
    height,
    scale,
    match_threshold,
):
    template = _build_symbol_template(symbol_reference_rgb)
    if template is None:
        return None

    detection_height, detection_width = detection_rgb.shape[:2]
    detection_gray = cv2.cvtColor(detection_rgb, cv2.COLOR_RGB2GRAY)
    detection_blur = cv2.GaussianBlur(detection_gray, (3, 3), 0)
    detection_gradient = cv2.magnitude(
        cv2.Sobel(detection_blur, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(detection_blur, cv2.CV_32F, 0, 1, ksize=3),
    )
    if float(detection_gradient.max()) <= 0.0:
        return None
    detection_gradient /= float(detection_gradient.max())
    local_background = cv2.GaussianBlur(detection_gray.astype(np.float32), (0, 0), sigmaX=15.0, sigmaY=15.0)
    detection_dark = np.maximum(local_background - detection_gray.astype(np.float32), 0.0)
    if float(detection_dark.max()) > 0.0:
        detection_dark /= float(detection_dark.max())

    template_width = int(template["width"])
    template_height = int(template["height"])
    min_scale = max(0.8, 16.0 / float(max(template_width, template_height)))
    max_scale = min(
        2.4,
        float(min(detection_width, detection_height)) / float(max(template_width, template_height)),
    )
    if max_scale < min_scale:
        return None

    best_match = None
    search_scales = np.linspace(min_scale, max_scale, 11)
    for scale_value in search_scales:
        scaled_width = max(8, int(round(template_width * float(scale_value))))
        scaled_height = max(8, int(round(template_height * float(scale_value))))
        if scaled_width >= detection_width or scaled_height >= detection_height:
            continue

        interpolation = cv2.INTER_AREA if scale_value < 1.0 else cv2.INTER_LINEAR
        scaled_template = cv2.resize(template["gradient"], (scaled_width, scaled_height), interpolation=interpolation).astype(np.float32)
        scaled_foreground = cv2.resize(template["foreground"], (scaled_width, scaled_height), interpolation=interpolation).astype(np.float32)
        scaled_mask = cv2.resize(template["mask"], (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)
        if np.count_nonzero(scaled_mask) < 16:
            continue

        template_mask = ((scaled_mask > 0).astype(np.uint8) * 255)
        try:
            response_gradient = cv2.matchTemplate(
                detection_gradient,
                scaled_template,
                cv2.TM_CCORR_NORMED,
                mask=template_mask,
            )
            response_dark = cv2.matchTemplate(
                detection_dark,
                scaled_foreground,
                cv2.TM_CCORR_NORMED,
                mask=template_mask,
            )
        except Exception:
            response_gradient = cv2.matchTemplate(detection_gradient, scaled_template, cv2.TM_CCORR_NORMED)
            response_dark = cv2.matchTemplate(detection_dark, scaled_foreground, cv2.TM_CCORR_NORMED)

        response_gradient = np.where(np.isfinite(response_gradient), response_gradient, 0.0)
        response_dark = np.where(np.isfinite(response_dark), response_dark, 0.0)
        response = 0.75 * response_dark + 0.25 * response_gradient
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(response)

        x = int(max_loc[0])
        y = int(max_loc[1])
        cx = (x + scaled_width * 0.5) / float(scale)
        cy = (y + scaled_height * 0.5) / float(scale)
        if not keep_region(cx, cy):
            continue

        gradient_patch = detection_gradient[y : y + scaled_height, x : x + scaled_width]
        dark_patch = detection_dark[y : y + scaled_height, x : x + scaled_width]
        if gradient_patch.shape[0] != scaled_height or gradient_patch.shape[1] != scaled_width:
            continue
        if dark_patch.shape[0] != scaled_height or dark_patch.shape[1] != scaled_width:
            continue

        mask_selector = scaled_mask > 0
        masked_gradient_values = gradient_patch[mask_selector]
        masked_template_gradient_values = scaled_template[mask_selector]
        masked_dark_values = dark_patch[mask_selector]
        masked_template_dark_values = scaled_foreground[mask_selector]
        if masked_gradient_values.size == 0 or masked_template_gradient_values.size == 0:
            continue

        gradient_ratio = float(
            np.clip(
                masked_gradient_values.mean() / max(float(masked_template_gradient_values.mean()), 1e-3),
                0.0,
                1.5,
            )
        )
        dark_ratio = 0.0
        dark_coverage = 0.0
        if masked_dark_values.size > 0 and masked_template_dark_values.size > 0:
            template_dark_mean = max(float(masked_template_dark_values.mean()), 1e-3)
            dark_ratio = float(np.clip(masked_dark_values.mean() / template_dark_mean, 0.0, 1.5))
            dark_coverage = float(np.mean(masked_dark_values > (template_dark_mean * 0.45)))

        confidence = (
            0.55 * float(max_val)
            + 0.15 * float(response_gradient[max_loc[1], max_loc[0]])
            + 0.15 * dark_ratio
            + 0.10 * gradient_ratio
            + 0.05 * dark_coverage
        )

        candidate = {
            "x": x,
            "y": y,
            "width": scaled_width,
            "height": scaled_height,
            "mask": scaled_mask,
            "foreground": scaled_foreground,
            "confidence": confidence,
            "scale": float(scale_value),
        }
        if best_match is None or candidate["confidence"] > best_match["confidence"]:
            best_match = candidate

    if best_match is None or best_match["confidence"] < float(match_threshold):
        return None

    detection_mask = np.zeros((detection_height, detection_width), dtype=np.uint8)
    y0 = int(best_match["y"])
    x0 = int(best_match["x"])
    y1 = y0 + int(best_match["height"])
    x1 = x0 + int(best_match["width"])
    symbol_mask = best_match["mask"].astype(np.uint8)
    expansion_kernel = max(3, _odd_kernel_size(max(best_match["width"], best_match["height"]) * 0.05))
    if expansion_kernel > 1:
        symbol_mask = cv2.dilate(symbol_mask, np.ones((expansion_kernel, expansion_kernel), dtype=np.uint8), iterations=1)
        symbol_mask = cv2.GaussianBlur(
            symbol_mask.astype(np.float32),
            (0, 0),
            sigmaX=max(0.8, expansion_kernel * 0.4),
            sigmaY=max(0.8, expansion_kernel * 0.4),
        )
        symbol_mask = np.where(symbol_mask > 24.0, 255, 0).astype(np.uint8)

    detection_mask[y0:y1, x0:x1] = symbol_mask
    _project_detection_mask(mask, detection_mask, padding, width, height, scale)
    return best_match


def _build_gemini_sparkle_alpha(size):
    canvas = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    main_length = max(6, int(round(size * 0.22)))
    secondary_length = max(4, int(round(size * 0.14)))
    thickness = max(1, int(round(size * 0.04)))
    secondary_thickness = max(1, thickness // 2)

    main = np.zeros_like(canvas, dtype=np.uint8)
    cv2.line(main, (center, center - main_length), (center, center + main_length), 255, thickness, cv2.LINE_AA)
    cv2.line(main, (center - main_length, center), (center + main_length, center), 255, thickness, cv2.LINE_AA)
    cv2.line(main, (center - secondary_length, center - secondary_length), (center + secondary_length, center + secondary_length), 180, secondary_thickness, cv2.LINE_AA)
    cv2.line(main, (center - secondary_length, center + secondary_length), (center + secondary_length, center - secondary_length), 180, secondary_thickness, cv2.LINE_AA)
    canvas = np.maximum(canvas, main.astype(np.float32) / 255.0)

    accent_specs = [
        (int(round(size * 0.24)), int(round(size * 0.24)), max(2, int(round(size * 0.06)))),
        (int(round(size * 0.76)), int(round(size * 0.2)), max(2, int(round(size * 0.045)))),
        (int(round(size * 0.7)), int(round(size * 0.78)), max(2, int(round(size * 0.035)))),
    ]
    for cx, cy, radius in accent_specs:
        cv2.circle(main, (cx, cy), radius, 140, thickness=-1, lineType=cv2.LINE_AA)

    canvas = np.maximum(canvas, cv2.GaussianBlur(main.astype(np.float32) / 255.0, (0, 0), sigmaX=max(0.6, size / 28.0)))
    canvas = np.clip(canvas * 0.38, 0.0, 0.75)
    return canvas.astype(np.float32)


def _get_gemini_template(size):
    with _GEMINI_TEMPLATE_LOCK:
        cached = _GEMINI_TEMPLATE_CACHE.get(size)
        if cached is not None:
            return cached

        alpha = _build_gemini_sparkle_alpha(size)
        logo = np.ones((size, size, 3), dtype=np.float32)
        gradient_x = np.linspace(0.96, 1.0, size, dtype=np.float32)
        gradient_y = np.linspace(0.96, 1.0, size, dtype=np.float32)[:, None]
        logo[..., 0] *= gradient_y
        logo[..., 1] *= np.minimum(1.0, gradient_x[None, :] + 0.01)
        logo[..., 2] *= np.minimum(1.0, gradient_y * 1.01)
        spatial_template = alpha
        gradient_template = cv2.magnitude(
            cv2.Sobel(spatial_template, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(spatial_template, cv2.CV_32F, 0, 1, ksize=3),
        )
        if float(gradient_template.max()) > 0.0:
            gradient_template /= float(gradient_template.max())
        cached = {
            "alpha": alpha,
            "logo": logo,
            "spatial": spatial_template.astype(np.float32),
            "gradient": gradient_template.astype(np.float32),
        }
        _GEMINI_TEMPLATE_CACHE[size] = cached
        return cached


def _get_gemini_candidate_sizes(width, height):
    base = _GEMINI_SMALL_SIZE if min(width, height) <= 1024 else _GEMINI_LARGE_SIZE
    scales = (0.8, 1.0, 1.2)
    candidates = sorted({max(16, int(round(base * scale))) for scale in scales})
    return candidates


def _extract_search_window(image, template_size):
    height, width = image.shape[:2]
    margin = _GEMINI_SMALL_MARGIN if template_size <= _GEMINI_SMALL_SIZE else _GEMINI_LARGE_MARGIN
    region_width = max(template_size * 3, int(round(width * 0.38)))
    region_height = max(template_size * 3, int(round(height * 0.38)))
    x0 = max(0, width - region_width - margin)
    y0 = max(0, height - region_height - margin)
    x1 = width
    y1 = height
    return x0, y0, x1, y1


def _normalized_variance_score(gray_roi):
    variance = float(np.var(gray_roi.astype(np.float32) / 255.0))
    return float(np.clip(1.0 - variance / 0.08, 0.0, 1.0))


def _detect_gemini_sparkle(rgb, threshold):
    height, width = rgb.shape[:2]
    if width < 64 or height < 64:
        return None

    best_match = None
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    if float(grad.max()) > 0.0:
        grad /= float(grad.max())

    for template_size in _get_gemini_candidate_sizes(width, height):
        template = _get_gemini_template(template_size)
        x0, y0, x1, y1 = _extract_search_window(rgb, template_size)
        search_gray = gray[y0:y1, x0:x1]
        search_grad = grad[y0:y1, x0:x1]
        if search_gray.shape[0] < template_size or search_gray.shape[1] < template_size:
            continue

        spatial_map = cv2.matchTemplate(search_gray, template["spatial"], cv2.TM_CCOEFF_NORMED)
        gradient_map = cv2.matchTemplate(search_grad, template["gradient"], cv2.TM_CCOEFF_NORMED)
        combined_map = 0.5 * spatial_map + 0.3 * gradient_map
        _, _max_val, _, max_loc = cv2.minMaxLoc(combined_map)

        match_x = x0 + int(max_loc[0])
        match_y = y0 + int(max_loc[1])
        roi_gray = gray[match_y : match_y + template_size, match_x : match_x + template_size]
        if roi_gray.shape[0] != template_size or roi_gray.shape[1] != template_size:
            continue

        spatial_score = float(spatial_map[max_loc[1], max_loc[0]])
        gradient_score = float(gradient_map[max_loc[1], max_loc[0]])
        variance_score = _normalized_variance_score(roi_gray)
        confidence = 0.5 * spatial_score + 0.3 * gradient_score + 0.2 * variance_score

        candidate = {
            "x": match_x,
            "y": match_y,
            "size": template_size,
            "confidence": confidence,
            "alpha": template["alpha"],
            "logo": template["logo"],
        }
        if best_match is None or candidate["confidence"] > best_match["confidence"]:
            best_match = candidate

    if best_match is None or best_match["confidence"] < float(threshold):
        return None
    return best_match


def _apply_gemini_reverse_alpha(rgb, match, cleanup_radius):
    x = int(match["x"])
    y = int(match["y"])
    size = int(match["size"])
    alpha = np.clip(match["alpha"], 0.0, 0.92).astype(np.float32)
    logo = match["logo"].astype(np.float32)

    roi = rgb[y : y + size, x : x + size].astype(np.float32) / 255.0
    safe = np.maximum(1.0 - alpha[..., None], 1e-3)
    restored = np.clip((roi - alpha[..., None] * logo) / safe, 0.0, 1.0)

    edge_mask = cv2.GaussianBlur((alpha > 0.03).astype(np.uint8) * 255, (0, 0), sigmaX=max(0.8, size / 24.0))
    grad_mask = cv2.magnitude(cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3))
    if float(grad_mask.max()) > 0.0:
        grad_mask /= float(grad_mask.max())
    grad_mask = np.clip(grad_mask * 1.4, 0.0, 1.0)

    if edge_mask.max() > 0:
        inpaint_input = np.clip(restored * 255.0, 0, 255).astype(np.uint8)
        inpainted = cv2.inpaint(inpaint_input, edge_mask, float(max(1, cleanup_radius)), cv2.INPAINT_TELEA).astype(np.float32) / 255.0
        restored = restored * (1.0 - grad_mask[..., None]) + inpainted * grad_mask[..., None]

    cleaned = rgb.copy().astype(np.float32) / 255.0
    cleaned[y : y + size, x : x + size] = restored

    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    mask[y : y + size, x : x + size] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return np.clip(cleaned * 255.0, 0, 255).astype(np.uint8), mask


def _remove_gemini_sparkle_for_batch(image, threshold, cleanup_radius):
    cleaned_images = []
    masks = []
    previews = []
    detected_batches = []

    for batch_image in image:
        rgb = _tensor_to_uint8(batch_image)
        match = _detect_gemini_sparkle(rgb, threshold)
        if match is None:
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            cleaned = rgb
            detected_text = "gemini sparkle not detected"
        else:
            cleaned, mask = _apply_gemini_reverse_alpha(rgb, match, cleanup_radius)
            detected_text = f"gemini sparkle confidence: {match['confidence']:.3f}, size: {match['size']}"

        preview = _make_mask_preview(rgb, mask)
        cleaned_images.append(torch.from_numpy(cleaned.astype(np.float32) / 255.0).unsqueeze(0))
        masks.append(torch.from_numpy(mask.astype(np.float32) / 255.0))
        previews.append(torch.from_numpy(preview.astype(np.float32) / 255.0).unsqueeze(0))
        detected_batches.append(detected_text)

    return torch.cat(cleaned_images, dim=0), torch.stack(masks, dim=0), torch.cat(previews, dim=0), "\n".join(detected_batches)


def _get_corner_boxes(width, height, corner_fallback, region, width_ratio, height_ratio):
    if corner_fallback == "off":
        return []

    if corner_fallback == "region_hint":
        if region not in {"edges", "corners"}:
            return []
        selected_corners = ["top_left", "top_right", "bottom_left", "bottom_right"]
    elif corner_fallback == "all_corners":
        selected_corners = ["top_left", "top_right", "bottom_left", "bottom_right"]
    else:
        selected_corners = [corner_fallback]

    box_width = max(1, int(round(width * float(width_ratio))))
    box_height = max(1, int(round(height * float(height_ratio))))

    corners = {
        "top_left": (0, 0, box_width, box_height),
        "top_right": (width - box_width, 0, width, box_height),
        "bottom_left": (0, height - box_height, box_width, height),
        "bottom_right": (width - box_width, height - box_height, width, height),
    }
    return [(name, corners[name]) for name in selected_corners if name in corners]


def _apply_corner_fallback(mask, keep_region, padding, width, height, corner_fallback, region, corner_width_ratio, corner_height_ratio):
    fallback_hits = []
    for name, (x_min, y_min, x_max, y_max) in _get_corner_boxes(
        width,
        height,
        corner_fallback,
        region,
        corner_width_ratio,
        corner_height_ratio,
    ):
        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        if not keep_region(cx, cy):
            continue

        padded_x_min = max(int(x_min) - int(padding), 0)
        padded_y_min = max(int(y_min) - int(padding), 0)
        padded_x_max = min(int(x_max) + int(padding), width - 1)
        padded_y_max = min(int(y_max) + int(padding), height - 1)
        cv2.rectangle(mask, (padded_x_min, padded_y_min), (padded_x_max, padded_y_max), 255, thickness=-1)
        fallback_hits.append(name)

    return fallback_hits


def _apply_mask_postprocess(mask, dilate, blur):
    if int(dilate) > 0 and mask.max() > 0:
        kernel_size = int(dilate) * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    if int(blur) > 0 and mask.max() > 0:
        blur_size = int(blur)
        if blur_size % 2 == 0:
            blur_size += 1
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    return mask


def _make_mask_preview(rgb, mask):
    preview = rgb.copy()
    overlay = np.zeros_like(preview)
    overlay[..., 0] = 255
    alpha = (mask.astype(np.float32) / 255.0 * 0.45)[..., None]
    return (preview.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha).astype(np.uint8)


def _get_inpaint_model_names():
    try:
        import folder_paths

        model_names = folder_paths.get_filename_list("inpaint")
        return model_names or ["big-lama.pt"]
    except Exception:
        return [_BIG_LAMA_MODEL_NAME]


def _get_inpaint_roots():
    roots = []
    try:
        import folder_paths

        folder_roots = folder_paths.get_folder_paths("inpaint") or []
        for folder_root in folder_roots:
            if folder_root and folder_root not in roots:
                roots.append(folder_root)
    except Exception:
        pass

    comfy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    repo_root = os.path.abspath(os.path.dirname(__file__))
    fallback_roots = [
        os.path.join(comfy_root, "models", "inpaint"),
        os.path.join(repo_root, "models", "inpaint"),
    ]
    for fallback_root in fallback_roots:
        if fallback_root not in roots:
            roots.append(fallback_root)
    return roots


def _find_inpaint_model_path(model_name):
    for model_root in _get_inpaint_roots():
        model_path = os.path.join(model_root, model_name)
        if os.path.exists(model_path):
            return model_path
    return None


def _auto_download_enabled():
    value = os.environ.get(_AUTO_DOWNLOAD_ENV, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _download_big_lama():
    if not _auto_download_enabled():
        return None

    from huggingface_hub import hf_hub_download

    target_root = _get_inpaint_roots()[0]
    os.makedirs(target_root, exist_ok=True)
    downloaded_path = hf_hub_download(
        repo_id=_BIG_LAMA_REPO_ID,
        filename=_BIG_LAMA_MODEL_NAME,
        local_dir=target_root,
    )
    if os.path.exists(downloaded_path):
        return downloaded_path

    fallback_path = os.path.join(target_root, _BIG_LAMA_MODEL_NAME)
    if os.path.exists(fallback_path):
        return fallback_path
    return None


def _get_inpaint_model_path(model_name):
    try:
        import folder_paths

        model_path = folder_paths.get_full_path("inpaint", model_name)
        if model_path is not None:
            return model_path
    except Exception:
        pass

    model_path = _find_inpaint_model_path(model_name)
    if model_path is not None:
        return model_path

    if model_name == _BIG_LAMA_MODEL_NAME:
        try:
            downloaded_path = _download_big_lama()
        except Exception as exception:
            raise RuntimeError(
                "Inpaint model file not found and automatic Big-LaMa download failed: "
                f"{exception}"
            ) from exception
        if downloaded_path is not None:
            return downloaded_path

    raise RuntimeError(
        f"Inpaint model file not found: {model_name}. "
        f"Set {_AUTO_DOWNLOAD_ENV}=1 to allow automatic download for {_BIG_LAMA_MODEL_NAME}."
    )


def _get_torch_device():
    try:
        from comfy.model_management import get_torch_device

        return get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_inpaint_model(model_name):
    with _INPAINT_MODEL_LOCK:
        cached_model = _INPAINT_MODEL_CACHE.get(model_name)
        if cached_model is not None:
            return cached_model

        from spandrel import ModelLoader

        model_path = _get_inpaint_model_path(model_name)
        if model_path.endswith(".pt"):
            state_dict = torch.jit.load(model_path, map_location="cpu").state_dict()
        else:
            try:
                import comfy.utils

                state_dict = comfy.utils.load_torch_file(model_path, safe_load=True)
            except Exception:
                state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

        model = ModelLoader().load_from_state_dict(state_dict).eval().cpu()
        if getattr(getattr(model, "architecture", None), "id", None) != "LaMa":
            raise RuntimeError(f"Unsupported inpaint model architecture: {getattr(getattr(model, 'architecture', None), 'id', type(model))}")
        _INPAINT_MODEL_CACHE[model_name] = model
        return model


def _load_diffusers_inpaint_pipeline(model_id):
    resolved_model_id = (model_id or _DIFFUSERS_DEFAULT_MODEL_ID).strip() or _DIFFUSERS_DEFAULT_MODEL_ID
    with _DIFFUSERS_PIPELINE_LOCK:
        cached_pipeline = _DIFFUSERS_PIPELINE_CACHE.get(resolved_model_id)
        if cached_pipeline is not None:
            return cached_pipeline

        try:
            from diffusers import StableDiffusionInpaintPipeline
        except Exception as exception:
            raise RuntimeError(
                "Diffusers inpaint requires diffusers, transformers, and pillow to be installed."
            ) from exception

        torch_device = _get_torch_device()
        torch_dtype = torch.float16 if getattr(torch_device, "type", "cpu") == "cuda" else torch.float32
        load_kwargs = {"torch_dtype": torch_dtype}
        try:
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                resolved_model_id,
                safety_checker=None,
                requires_safety_checker=False,
                **load_kwargs,
            )
        except TypeError:
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(resolved_model_id, **load_kwargs)
            if hasattr(pipeline, "safety_checker"):
                pipeline.safety_checker = None
        except Exception as exception:
            raise RuntimeError(f"Could not load diffusers inpaint model '{resolved_model_id}': {exception}") from exception

        if hasattr(pipeline, "set_progress_bar_config"):
            pipeline.set_progress_bar_config(disable=True)
        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()

        _DIFFUSERS_PIPELINE_CACHE[resolved_model_id] = pipeline
        return pipeline


def _mask_to_torch(mask_tensor):
    if len(mask_tensor.shape) == 2:
        return mask_tensor.unsqueeze(0).unsqueeze(0)
    if len(mask_tensor.shape) == 3:
        return mask_tensor.unsqueeze(1)
    return mask_tensor


def _pad_reflect_once(tensor, padding):
    _batch, _channels, height, width = tensor.shape
    requested = np.array(padding)
    limits = np.array([width, width, height, height]) - 1
    initial_padding = np.minimum(requested, limits)
    extra_padding = requested - initial_padding
    tensor = F.pad(tensor, tuple(initial_padding), mode="reflect")
    if np.any(extra_padding > 0):
        tensor = F.pad(tensor, tuple(extra_padding), mode="constant")
    return tensor


def _resize_square(image, mask, size):
    _batch, _channels, height, width = image.shape
    pad_width = 0
    pad_height = 0
    previous_size = width
    if width == size and height == size:
        return image, mask, (pad_width, pad_height, previous_size)

    if width < height:
        pad_width = height - width
        previous_size = height
    elif height < width:
        pad_height = width - height
        previous_size = width

    image = _pad_reflect_once(image, (0, pad_width, 0, pad_height))
    mask = _pad_reflect_once(mask, (0, pad_width, 0, pad_height))
    if image.shape[-1] != size:
        image = F.interpolate(image, size=size, mode="nearest-exact")
        mask = F.interpolate(mask, size=size, mode="nearest-exact")
    return image, mask, (pad_width, pad_height, previous_size)


def _undo_resize_square(image, original_size):
    pad_width, pad_height, previous_size = original_size
    if image.shape[-1] != previous_size or image.shape[-2] != previous_size:
        image = F.interpolate(image, size=previous_size, mode="bilinear", align_corners=False)
    return image[:, :, 0 : previous_size - pad_height, 0 : previous_size - pad_width]


def _round_up_to_multiple(value, multiple):
    if multiple <= 1:
        return int(value)
    return int(((int(value) + multiple - 1) // multiple) * multiple)


def _mask_bounding_box(mask, padding):
    points = cv2.findNonZero((mask > 0).astype(np.uint8))
    if points is None:
        return None

    mask_height, mask_width = mask.shape[:2]
    x, y, box_width, box_height = cv2.boundingRect(points)
    crop_padding = max(0, int(padding))
    x0 = max(0, x - crop_padding)
    y0 = max(0, y - crop_padding)
    x1 = min(mask_width, x + box_width + crop_padding)
    y1 = min(mask_height, y + box_height + crop_padding)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _prepare_diffusers_crop_inputs(rgb_crop, mask_crop):
    original_height, original_width = rgb_crop.shape[:2]
    resized_rgb = rgb_crop
    resized_mask = mask_crop

    max_side = max(original_height, original_width)
    min_side = min(original_height, original_width)
    scale = 1.0
    if max_side > _DIFFUSERS_MAX_SIDE:
        scale = _DIFFUSERS_MAX_SIDE / float(max_side)
    elif min_side < _DIFFUSERS_MIN_SIDE:
        scale = min(_DIFFUSERS_MIN_SIDE / float(max(min_side, 1)), _DIFFUSERS_MAX_SIDE / float(max_side))

    if abs(scale - 1.0) > 1e-3:
        resized_width = max(8, int(round(original_width * scale)))
        resized_height = max(8, int(round(original_height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized_rgb = cv2.resize(rgb_crop, (resized_width, resized_height), interpolation=interpolation)
        resized_mask = cv2.resize(mask_crop, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)

    resized_height, resized_width = resized_rgb.shape[:2]
    target_width = max(8, _round_up_to_multiple(resized_width, 8))
    target_height = max(8, _round_up_to_multiple(resized_height, 8))
    pad_right = max(0, target_width - resized_width)
    pad_bottom = max(0, target_height - resized_height)

    if pad_right > 0 or pad_bottom > 0:
        work_rgb = cv2.copyMakeBorder(resized_rgb, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
        work_mask = cv2.copyMakeBorder(resized_mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0)
    else:
        work_rgb = resized_rgb
        work_mask = resized_mask

    meta = {
        "original_width": int(original_width),
        "original_height": int(original_height),
        "resized_width": int(resized_width),
        "resized_height": int(resized_height),
    }
    return work_rgb, work_mask, meta


def _restore_diffusers_crop_output(restored_rgb, meta):
    resized_width = int(meta["resized_width"])
    resized_height = int(meta["resized_height"])
    output = restored_rgb[:resized_height, :resized_width]
    original_width = int(meta["original_width"])
    original_height = int(meta["original_height"])
    if output.shape[1] != original_width or output.shape[0] != original_height:
        output = cv2.resize(output, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    return output


def _inpaint_with_lama(image, masks, model_name):
    if not masks or not any(mask.max().item() > 0 for mask in masks):
        return image

    model = _load_inpaint_model(model_name)
    input_device = image.device
    device = _get_torch_device()
    image_bchw = image.detach().permute(0, 3, 1, 2).cpu()
    mask_bchw = _mask_to_torch(torch.stack(masks, dim=0)).cpu()

    model.to(device)
    try:
        with torch.no_grad():
            work_image, work_mask, original_size = _resize_square(image_bchw, mask_bchw, _LAMA_TARGET_SIZE)
            work_mask = (work_mask >= 0.99).to(work_image.dtype)
            result = model(work_image.to(device), work_mask.to(device))
            result = _undo_resize_square(result.to("cpu"), original_size)
            original_mask = (mask_bchw >= 0.99).to(image_bchw.dtype)
            result = image_bchw + (result - image_bchw) * original_mask
    finally:
        model.cpu()

    return result.permute(0, 2, 3, 1).to(input_device)


def _inpaint_with_diffusers_crop(
    image,
    masks,
    model_id,
    prompt,
    negative_prompt,
    strength,
    guidance_scale,
    steps,
    crop_padding,
):
    if not masks or not any(mask.max().item() > 0 for mask in masks):
        return image

    pipeline = _load_diffusers_inpaint_pipeline(model_id)
    torch_device = _get_torch_device()
    effective_prompt = (prompt or "").strip() or _DIFFUSERS_DEFAULT_PROMPT
    effective_negative_prompt = (negative_prompt or "").strip() or _DIFFUSERS_DEFAULT_NEGATIVE_PROMPT
    strength = float(np.clip(strength, 0.0, 1.0))
    guidance_scale = float(max(guidance_scale, 1.0))
    steps = max(1, int(steps))
    crop_padding = max(0, int(crop_padding))

    try:
        from PIL import Image
    except Exception as exception:
        raise RuntimeError("Diffusers inpaint requires pillow to be installed.") from exception

    cleaned_images = []
    pipeline.to(torch_device)
    try:
        for batch_image, mask_tensor in zip(image, masks):
            rgb = _tensor_to_uint8(batch_image)
            mask = np.clip(mask_tensor.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            if mask.max() == 0:
                cleaned_images.append(torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0))
                continue

            bounding_box = _mask_bounding_box(mask, crop_padding)
            if bounding_box is None:
                cleaned_images.append(torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0))
                continue

            x0, y0, x1, y1 = bounding_box
            rgb_crop = rgb[y0:y1, x0:x1].copy()
            mask_crop = mask[y0:y1, x0:x1].copy()

            work_rgb, work_mask, crop_meta = _prepare_diffusers_crop_inputs(rgb_crop, mask_crop)
            result = pipeline(
                prompt=effective_prompt,
                negative_prompt=effective_negative_prompt,
                image=Image.fromarray(work_rgb),
                mask_image=Image.fromarray(work_mask),
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
            )

            restored_crop = np.array(result.images[0].convert("RGB"), dtype=np.uint8)
            restored_crop = _restore_diffusers_crop_output(restored_crop, crop_meta)

            cleaned_crop = rgb_crop.copy()
            cleaned_crop[mask_crop > 0] = restored_crop[mask_crop > 0]
            cleaned_rgb = rgb.copy()
            cleaned_rgb[y0:y1, x0:x1] = cleaned_crop
            cleaned_images.append(torch.from_numpy(cleaned_rgb.astype(np.float32) / 255.0).unsqueeze(0))
    finally:
        try:
            pipeline.to("cpu")
        except Exception:
            pass
        if getattr(torch_device, "type", "cpu") == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    return torch.cat(cleaned_images, dim=0)


def _project_symbol_match_to_original(match, detection_scale, width, height):
    scale_value = max(float(detection_scale), 1e-6)
    x0 = max(0, int(round(float(match["x"]) / scale_value)))
    y0 = max(0, int(round(float(match["y"]) / scale_value)))
    x1 = min(width, int(round((float(match["x"]) + float(match["width"])) / scale_value)))
    y1 = min(height, int(round((float(match["y"]) + float(match["height"])) / scale_value)))
    if x1 <= x0 or y1 <= y0:
        return None

    region_width = x1 - x0
    region_height = y1 - y0
    original_mask = cv2.resize(match["mask"].astype(np.uint8), (region_width, region_height), interpolation=cv2.INTER_NEAREST)
    original_foreground = cv2.resize(match["foreground"].astype(np.float32), (region_width, region_height), interpolation=cv2.INTER_LINEAR)
    if float(original_foreground.max()) > 0.0:
        original_foreground /= float(original_foreground.max())
    return {
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "mask": original_mask,
        "foreground": original_foreground.astype(np.float32),
        "confidence": float(match["confidence"]),
        "scale": float(match["scale"]),
    }


def _apply_symbol_reverse_blend(rgb, match, detection_scale, cleanup_radius):
    height, width = rgb.shape[:2]
    projected = _project_symbol_match_to_original(match, detection_scale, width, height)
    if projected is None:
        empty_mask = np.zeros((height, width), dtype=np.uint8)
        return rgb, empty_mask

    x0 = projected["x0"]
    y0 = projected["y0"]
    x1 = projected["x1"]
    y1 = projected["y1"]
    roi = rgb[y0:y1, x0:x1].copy()
    if roi.size == 0:
        empty_mask = np.zeros((height, width), dtype=np.uint8)
        return rgb, empty_mask

    template_mask = np.where(projected["mask"] > 0, 255, 0).astype(np.uint8)
    if template_mask.max() == 0:
        empty_mask = np.zeros((height, width), dtype=np.uint8)
        return rgb, empty_mask

    foreground = np.where(template_mask > 0, projected["foreground"], 0.0).astype(np.float32)
    if float(foreground.max()) > 0.0:
        foreground /= float(foreground.max())

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    local_background = cv2.GaussianBlur(roi_gray, (0, 0), sigmaX=14.0, sigmaY=14.0)
    local_dark_residual = np.maximum(local_background - roi_gray, 0.0)

    mask_selector = template_mask > 0
    foreground_values = foreground[mask_selector]
    residual_values = local_dark_residual[mask_selector]
    foreground_threshold = 0.4 if foreground_values.size == 0 else max(0.32, float(np.percentile(foreground_values, 78)))
    residual_threshold = 0.03 if residual_values.size == 0 else max(0.02, float(np.percentile(residual_values, 60)))

    stroke_mask = np.where(
        mask_selector & (foreground >= foreground_threshold) & (local_dark_residual >= residual_threshold),
        255,
        0,
    ).astype(np.uint8)
    thin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    stroke_mask = cv2.morphologyEx(stroke_mask, cv2.MORPH_OPEN, thin_kernel, iterations=1)
    stroke_mask = cv2.dilate(stroke_mask, thin_kernel, iterations=1)
    if int((stroke_mask > 0).sum()) < 64:
        stroke_mask = cv2.erode(template_mask, thin_kernel, iterations=1)

    repair_mask = cv2.dilate(stroke_mask, thin_kernel, iterations=1)
    background_rgb = cv2.inpaint(roi, repair_mask, float(max(1, min(3, int(cleanup_radius) + 1))), cv2.INPAINT_TELEA)

    roi_float = roi.astype(np.float32) / 255.0
    background_float = background_rgb.astype(np.float32) / 255.0
    background_gray = cv2.cvtColor(background_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    dark_residual = np.maximum(background_gray - roi_gray, 0.0)
    relative_dark = dark_residual / np.maximum(background_gray, 0.08)
    stroke_weight = cv2.GaussianBlur(
        stroke_mask.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=max(0.8, max(roi.shape[0], roi.shape[1]) / 58.0),
        sigmaY=max(0.8, max(roi.shape[0], roi.shape[1]) / 58.0),
    )
    alpha_map = np.clip(relative_dark * 1.1, 0.0, 0.72)
    alpha_map *= stroke_weight
    alpha_map *= (0.55 + 0.45 * foreground)
    alpha_map = np.where(alpha_map > 0.02, alpha_map, 0.0).astype(np.float32)

    safe = np.maximum(1.0 - alpha_map[..., None], 0.35)
    reverse_candidate = np.clip(roi_float / safe, 0.0, 1.0)
    reverse_candidate = np.minimum(reverse_candidate, np.clip(background_float * 1.15 + 0.05, 0.0, 1.0))
    blend_weight = np.clip(alpha_map * 1.25, 0.0, 1.0)
    restored = roi_float * (1.0 - blend_weight[..., None]) + (
        0.55 * reverse_candidate + 0.45 * background_float
    ) * blend_weight[..., None]

    residual_mask = repair_mask
    if residual_mask.max() > 0:
        refined_input = np.clip(restored * 255.0, 0, 255).astype(np.uint8)
        refined_rgb = cv2.inpaint(refined_input, residual_mask, float(max(1, int(cleanup_radius))), cv2.INPAINT_TELEA).astype(np.float32) / 255.0
        residual_weight = cv2.GaussianBlur(residual_mask.astype(np.float32) / 255.0, (0, 0), sigmaX=0.9, sigmaY=0.9)
        residual_weight *= 0.35
        restored = restored * (1.0 - residual_weight[..., None]) + refined_rgb * residual_weight[..., None]

    cleaned = rgb.copy().astype(np.float32) / 255.0
    cleaned[y0:y1, x0:x1] = np.clip(restored, 0.0, 1.0)

    applied_mask = np.zeros((height, width), dtype=np.uint8)
    alpha_u8 = np.clip(alpha_map * 255.0, 0, 255).astype(np.uint8)
    applied_mask[y0:y1, x0:x1] = np.maximum(alpha_u8, stroke_mask)
    return np.clip(cleaned * 255.0, 0, 255).astype(np.uint8), applied_mask


def _remove_symbol_reverse_blend_for_batch(image, region, symbol_reference, match_threshold, cleanup_radius):
    cleaned_images = []
    masks = []
    previews = []
    detected_batches = []

    for batch_index, batch_image in enumerate(image):
        rgb = _tensor_to_uint8(batch_image)
        height, width = rgb.shape[:2]
        symbol_reference_rgb = _get_batch_reference_image(symbol_reference, batch_index)
        if symbol_reference_rgb is None:
            mask = np.zeros((height, width), dtype=np.uint8)
            cleaned = rgb
            detected_text = "symbol reverse blend reference missing"
        else:
            detection_rgb, detection_scale = _prepare_detection_rgb(rgb)
            keep_region = _region_filter(width, height, region)
            scratch_mask = np.zeros((height, width), dtype=np.uint8)
            match = _mask_symbol_template_regions(
                scratch_mask,
                detection_rgb,
                symbol_reference_rgb,
                keep_region,
                0,
                width,
                height,
                detection_scale,
                match_threshold,
            )
            if match is None:
                mask = np.zeros((height, width), dtype=np.uint8)
                cleaned = rgb
                detected_text = "symbol reverse blend not detected"
            else:
                cleaned, mask = _apply_symbol_reverse_blend(rgb, match, detection_scale, cleanup_radius)
                detected_text = f"symbol reverse blend confidence: {match['confidence']:.3f}, scale: {match['scale']:.2f}"

        preview = _make_mask_preview(rgb, mask)
        cleaned_images.append(torch.from_numpy(cleaned.astype(np.float32) / 255.0).unsqueeze(0))
        masks.append(torch.from_numpy(mask.astype(np.float32) / 255.0))
        previews.append(torch.from_numpy(preview.astype(np.float32) / 255.0).unsqueeze(0))
        detected_batches.append(detected_text)

    return torch.cat(cleaned_images, dim=0), torch.stack(masks, dim=0), torch.cat(previews, dim=0), "\n".join(detected_batches)


def _detect_watermark_mask_for_batch(
    image,
    languages,
    detection_mode,
    detail_detector,
    region,
    min_confidence,
    padding,
    dilate,
    blur,
    cv2_sensitivity,
    gpu,
    corner_fallback,
    corner_width_ratio,
    corner_height_ratio,
    symbol_reference=None,
    symbol_match_threshold=0.45,
):
    reader = None
    ocr_error = None
    if detection_mode != "cv2_only":
        try:
            reader = _get_reader(languages, gpu and torch.cuda.is_available())
        except Exception as exception:
            if detection_mode == "ocr_only":
                raise RuntimeError(f"EasyOCR could not be loaded: {exception}") from exception
            ocr_error = str(exception)

    masks = []
    previews = []
    detected_batches = []
    rgb_batches = []

    for batch_index, batch_image in enumerate(image):
        rgb = _tensor_to_uint8(batch_image)
        height, width = rgb.shape[:2]
        detection_rgb, detection_scale = _prepare_detection_rgb(rgb)
        mask = np.zeros((height, width), dtype=np.uint8)
        keep_region = _region_filter(width, height, region)
        detected_text = []

        if reader is not None:
            results = reader.readtext(detection_rgb)
            detected_text = _mask_ocr_results(mask, results, keep_region, min_confidence, padding, width, height, detection_scale)

        if detail_detector == "decomposition" and detection_mode != "ocr_only":
            region_count = _mask_decomposition_regions(
                mask,
                detection_rgb,
                keep_region,
                padding,
                cv2_sensitivity,
                width,
                height,
                detection_scale,
            )
            if region_count > 0:
                detected_text.append(f"decomposition regions: {region_count}")

            overlay_region_count = _mask_low_contrast_overlay_regions(
                mask,
                detection_rgb,
                keep_region,
                padding,
                cv2_sensitivity,
                width,
                height,
                detection_scale,
            )
            if overlay_region_count > 0:
                detected_text.append(f"overlay regions: {overlay_region_count}")

        if detail_detector == "symbol_template" and detection_mode != "ocr_only":
            symbol_reference_rgb = _get_batch_reference_image(symbol_reference, batch_index)
            best_symbol_match = _mask_symbol_template_regions(
                mask,
                detection_rgb,
                symbol_reference_rgb,
                keep_region,
                padding,
                width,
                height,
                detection_scale,
                symbol_match_threshold,
            )
            if best_symbol_match is None and symbol_reference_rgb is None:
                detected_text.append("symbol template reference missing")
            elif best_symbol_match is not None:
                detected_text.append(
                    f"symbol template confidence: {best_symbol_match['confidence']:.3f}, scale: {best_symbol_match['scale']:.2f}"
                )

        if detection_mode != "ocr_only" and mask.max() == 0:
            region_count = _mask_cv2_text_like_regions(
                mask,
                detection_rgb,
                keep_region,
                padding,
                cv2_sensitivity,
                width,
                height,
                detection_scale,
            )
            if region_count > 0:
                detected_text.append(f"cv2 fallback regions: {region_count}")
            elif ocr_error:
                detected_text.append(f"EasyOCR unavailable: {ocr_error}")

        if mask.max() == 0:
            fallback_hits = _apply_corner_fallback(
                mask,
                keep_region,
                padding,
                width,
                height,
                corner_fallback,
                region,
                corner_width_ratio,
                corner_height_ratio,
            )
            if fallback_hits:
                detected_text.append(f"corner fallback: {', '.join(fallback_hits)}")

        mask = _apply_mask_postprocess(mask, dilate, blur)
        preview = _make_mask_preview(rgb, mask)

        rgb_batches.append(rgb)
        masks.append(torch.from_numpy(mask.astype(np.float32) / 255.0))
        previews.append(torch.from_numpy(preview.astype(np.float32) / 255.0).unsqueeze(0))
        detected_batches.append(", ".join(detected_text))

    return rgb_batches, masks, previews, detected_batches


class AutoWatermarkMaskOCR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "languages": ("STRING", {"default": "en", "tooltip": "EasyOCR language codes separated by commas, e.g. en,de."}),
                "detection_mode": (["ocr_then_cv2", "ocr_only", "cv2_only"], {"default": "ocr_then_cv2", "tooltip": "How to build the mask. Start with cv2_only for logos and faint symbols, or ocr_then_cv2 for readable text watermarks."}),
                "region": (["full_image", "edges", "corners"], {"default": "full_image", "tooltip": "Limits the search area. Use corners or edges when the watermark is near the border for fewer false positives and faster matching."}),
                "min_confidence": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Minimum OCR confidence before a text hit is added to the mask. Higher values are stricter."}),
                "padding": ("INT", {"default": 12, "min": 0, "max": 256, "step": 1, "tooltip": "Extra pixels added around each detected region. Increase if the mask clips the watermark edges."}),
                "dilate": ("INT", {"default": 10, "min": 0, "max": 256, "step": 1, "tooltip": "Expands the mask after detection. Useful when thin strokes or glow edges are missed."}),
                "blur": ("INT", {"default": 7, "min": 0, "max": 255, "step": 1, "tooltip": "Softens mask edges before preview and inpainting. Lower values keep the mask tighter."}),
                "cv2_sensitivity": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Controls how aggressively the CV2 fallback grabs low-contrast regions. Raise it for faint overlays, lower it to reduce false positives."}),
                "corner_fallback": (["off", "region_hint", "top_left", "top_right", "bottom_left", "bottom_right", "all_corners"], {"default": "off", "tooltip": "Adds a simple corner mask when the watermark is pinned to a corner and automatic detection misses it."}),
                "corner_width_ratio": ("FLOAT", {"default": 0.12, "min": 0.02, "max": 0.5, "step": 0.01, "tooltip": "Corner fallback width as a fraction of image width."}),
                "corner_height_ratio": ("FLOAT", {"default": 0.08, "min": 0.02, "max": 0.5, "step": 0.01, "tooltip": "Corner fallback height as a fraction of image height."}),
                "gpu": ("BOOLEAN", {"default": True, "tooltip": "Use GPU for EasyOCR when available. Turn off if OCR startup is unstable in your environment."}),
                "detail_detector": (["none", "decomposition", "symbol_template"], {"default": "none", "tooltip": "Extra detector for hard cases. Use decomposition for faint blended overlays, or symbol_template when you can connect a watermark_template image."}),
                "symbol_match_threshold": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Template match confidence threshold for symbol_template. Raise it to avoid lookalikes; lower it if the right watermark is being missed."}),
            },
            "optional": {
                "watermark_template": ("IMAGE", {"tooltip": "Optional reference image of the watermark you want to find. Best results come from a tight clean crop of the watermark on a similar background."}),
                "symbol_reference": ("IMAGE", {"tooltip": "Legacy alias for watermark_template. You only need one of these inputs connected."}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("mask", "mask_preview", "detected_text")
    FUNCTION = "detect"
    CATEGORY = "image/mask"

    def detect(
        self,
        image,
        languages,
        detection_mode,
        region,
        min_confidence,
        padding,
        dilate,
        blur,
        cv2_sensitivity,
        corner_fallback,
        corner_width_ratio,
        corner_height_ratio,
        gpu,
        detail_detector,
        symbol_match_threshold=0.45,
        watermark_template=None,
        symbol_reference=None,
    ):
        template_input = _resolve_template_input(watermark_template, symbol_reference)
        _rgb_batches, masks, previews, detected_batches = _detect_watermark_mask_for_batch(
            image,
            languages,
            detection_mode,
            detail_detector,
            region,
            min_confidence,
            padding,
            dilate,
            blur,
            cv2_sensitivity,
            gpu,
            corner_fallback,
            corner_width_ratio,
            corner_height_ratio,
            template_input,
            symbol_match_threshold,
        )

        return (torch.stack(masks, dim=0), torch.cat(previews, dim=0), "\n".join(detected_batches))


class AutoWatermarkRemover:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "languages": ("STRING", {"default": "en", "tooltip": "Used only when OCR modes are selected."}),
                "detection_mode": (["cv2_only", "ocr_then_cv2", "ocr_only"], {"default": "cv2_only", "tooltip": "How to create the watermark mask before cleanup. cv2_only is usually the best starting point for logos and symbols."}),
                "region": (["full_image", "edges", "corners"], {"default": "full_image", "tooltip": "Limits detection to likely watermark areas. Smaller regions are faster and reduce false hits."}),
                "min_confidence": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Minimum OCR confidence before text is masked. Only matters for OCR modes."}),
                "padding": ("INT", {"default": 10, "min": 0, "max": 256, "step": 1, "tooltip": "Extra pixels around each detected area before cleanup."}),
                "dilate": ("INT", {"default": 5, "min": 0, "max": 256, "step": 1, "tooltip": "Expands the final mask. Increase if the watermark edges are still visible after removal."}),
                "blur": ("INT", {"default": 3, "min": 0, "max": 255, "step": 1, "tooltip": "Softens the mask edge. Lower values keep the repaired region tighter."}),
                "cv2_sensitivity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Sensitivity for non-OCR detection. Raise it for faint overlays, lower it to avoid masking real image detail."}),
                "corner_fallback": (["off", "region_hint", "top_left", "top_right", "bottom_left", "bottom_right", "all_corners"], {"default": "off", "tooltip": "Adds a simple mask in one or more corners when the watermark sits near the border and detection is inconsistent."}),
                "corner_width_ratio": ("FLOAT", {"default": 0.12, "min": 0.02, "max": 0.5, "step": 0.01, "tooltip": "Corner fallback width as a fraction of image width."}),
                "corner_height_ratio": ("FLOAT", {"default": 0.08, "min": 0.02, "max": 0.5, "step": 0.01, "tooltip": "Corner fallback height as a fraction of image height."}),
                "inpaint_method": (_get_inpaint_methods(), {"default": "opencv_telea", "tooltip": "Cleanup backend. Start with opencv_telea for small marks, big_lama for thicker texture overlap, symbol_reverse_blend when you have a watermark_template for a semi-transparent symbol."}),
                "inpaint_model": (_get_inpaint_model_names(), {"default": "big-lama.pt", "tooltip": "Big-LaMa checkpoint name. Used only when inpaint_method=big_lama."}),
                "inpaint_radius": ("INT", {"default": 5, "min": 1, "max": 64, "step": 1, "tooltip": "OpenCV or cleanup radius. Small values preserve detail better; larger values can erase thicker marks."}),
                "gpu": ("BOOLEAN", {"default": False, "tooltip": "Use GPU for OCR when available. Big-LaMa and Diffusers manage their own device placement."}),
                "detail_detector": (["none", "decomposition", "symbol_template"], {"default": "none", "tooltip": "Extra detector for hard masks. Use symbol_template together with watermark_template for known logos or repeated symbol watermarks."}),
                "symbol_match_threshold": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Template match threshold for symbol_template and symbol_reverse_blend. Raise it to be stricter."}),
                "diffusers_model_id": ("STRING", {"default": _DIFFUSERS_DEFAULT_MODEL_ID, "tooltip": "Optional Hugging Face model id or local folder for diffusers_sd_inpaint."}),
                "diffusers_prompt": ("STRING", {"default": _DIFFUSERS_DEFAULT_PROMPT, "multiline": True, "tooltip": "Prompt used only by diffusers_sd_inpaint. Keep it focused on preserving the original subject while removing the watermark."}),
                "diffusers_negative_prompt": ("STRING", {"default": _DIFFUSERS_DEFAULT_NEGATIVE_PROMPT, "multiline": True, "tooltip": "Negative prompt used only by diffusers_sd_inpaint."}),
                "diffusers_strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How strongly Diffusers redraws the cropped area. Lower values preserve more of the source image."}),
                "diffusers_guidance_scale": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "Prompt guidance for diffusers_sd_inpaint."}),
                "diffusers_steps": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1, "tooltip": "Inference steps for diffusers_sd_inpaint."}),
                "diffusers_crop_padding": ("INT", {"default": 48, "min": 0, "max": 512, "step": 1, "tooltip": "Extra context around the masked area when using diffusers_sd_inpaint."}),
            },
            "optional": {
                "watermark_template": ("IMAGE", {"tooltip": "Optional reference image of the watermark to match. Use a tight crop when possible. A clean external template works better than a noisy crop from the same image."}),
                "symbol_reference": ("IMAGE", {"tooltip": "Legacy alias for watermark_template. You only need one of these inputs connected."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "mask", "mask_preview", "detected_text")
    FUNCTION = "remove"
    CATEGORY = "image/cleanup"

    @classmethod
    def IS_CHANGED(
        cls,
        image=None,
        languages=None,
        detection_mode=None,
        region=None,
        min_confidence=None,
        padding=None,
        dilate=None,
        blur=None,
        cv2_sensitivity=None,
        corner_fallback=None,
        corner_width_ratio=None,
        corner_height_ratio=None,
        inpaint_method=None,
        inpaint_model=None,
        inpaint_radius=None,
        gpu=None,
        detail_detector=None,
        symbol_match_threshold=None,
        diffusers_model_id=None,
        diffusers_prompt=None,
        diffusers_negative_prompt=None,
        diffusers_strength=None,
        diffusers_guidance_scale=None,
        diffusers_steps=None,
        diffusers_crop_padding=None,
        watermark_template=None,
        symbol_reference=None,
    ):
        template_input = _resolve_template_input(watermark_template, symbol_reference)
        return (
            languages,
            detection_mode,
            region,
            min_confidence,
            padding,
            dilate,
            blur,
            cv2_sensitivity,
            corner_fallback,
            corner_width_ratio,
            corner_height_ratio,
            inpaint_method,
            inpaint_model,
            inpaint_radius,
            gpu,
            detail_detector,
            symbol_match_threshold,
            diffusers_model_id,
            diffusers_prompt,
            diffusers_negative_prompt,
            diffusers_strength,
            diffusers_guidance_scale,
            diffusers_steps,
            diffusers_crop_padding,
            None if template_input is None else tuple(template_input.shape),
        )

    def remove(
        self,
        image,
        languages,
        detection_mode,
        region,
        min_confidence,
        padding,
        dilate,
        blur,
        cv2_sensitivity,
        corner_fallback,
        corner_width_ratio,
        corner_height_ratio,
        inpaint_method,
        inpaint_model,
        inpaint_radius,
        gpu,
        detail_detector,
        symbol_match_threshold=0.45,
        diffusers_model_id=_DIFFUSERS_DEFAULT_MODEL_ID,
        diffusers_prompt=_DIFFUSERS_DEFAULT_PROMPT,
        diffusers_negative_prompt=_DIFFUSERS_DEFAULT_NEGATIVE_PROMPT,
        diffusers_strength=0.9,
        diffusers_guidance_scale=4.5,
        diffusers_steps=30,
        diffusers_crop_padding=48,
        watermark_template=None,
        symbol_reference=None,
    ):
        template_input = _resolve_template_input(watermark_template, symbol_reference)
        if inpaint_method == "gemini_reverse_alpha":
            return _remove_gemini_sparkle_for_batch(image, min_confidence, inpaint_radius)

        if inpaint_method == "symbol_reverse_blend":
            if template_input is None:
                raise RuntimeError("symbol_reverse_blend requires watermark_template or symbol_reference to be connected.")
            return _remove_symbol_reverse_blend_for_batch(
                image,
                region,
                template_input,
                symbol_match_threshold,
                inpaint_radius,
            )

        rgb_batches, masks, previews, detected_batches = _detect_watermark_mask_for_batch(
            image,
            languages,
            detection_mode,
            detail_detector,
            region,
            min_confidence,
            padding,
            dilate,
            blur,
            cv2_sensitivity,
            gpu,
            corner_fallback,
            corner_width_ratio,
            corner_height_ratio,
            template_input,
            symbol_match_threshold,
        )

        if inpaint_method == "big_lama":
            cleaned = _inpaint_with_lama(image, masks, inpaint_model)
            return (cleaned, torch.stack(masks, dim=0), torch.cat(previews, dim=0), "\n".join(detected_batches))

        if inpaint_method == "diffusers_sd_inpaint":
            cleaned = _inpaint_with_diffusers_crop(
                image,
                masks,
                diffusers_model_id,
                diffusers_prompt,
                diffusers_negative_prompt,
                diffusers_strength,
                diffusers_guidance_scale,
                diffusers_steps,
                diffusers_crop_padding,
            )
            return (cleaned, torch.stack(masks, dim=0), torch.cat(previews, dim=0), "\n".join(detected_batches))

        method = cv2.INPAINT_TELEA if inpaint_method == "opencv_telea" else cv2.INPAINT_NS
        cleaned_images = []

        for rgb, mask_tensor in zip(rgb_batches, masks):
            mask = np.clip(mask_tensor.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            if mask.max() == 0:
                cleaned = rgb
            else:
                cleaned = cv2.inpaint(rgb, mask, float(inpaint_radius), method)
            cleaned_images.append(torch.from_numpy(cleaned.astype(np.float32) / 255.0).unsqueeze(0))

        return (torch.cat(cleaned_images, dim=0), torch.stack(masks, dim=0), torch.cat(previews, dim=0), "\n".join(detected_batches))


NODE_CLASS_MAPPINGS = {
    "AutoWatermarkMaskOCR": AutoWatermarkMaskOCR,
    "AutoWatermarkRemover": AutoWatermarkRemover,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoWatermarkMaskOCR": "Auto Watermark Mask (OCR)",
    "AutoWatermarkRemover": "Auto Watermark Remover",
}
