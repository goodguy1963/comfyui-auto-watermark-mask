import threading

import cv2
import numpy as np
import torch
import torch.nn.functional as F


_READER_CACHE = {}
_READER_LOCK = threading.Lock()
_INPAINT_MODEL_CACHE = {}
_INPAINT_MODEL_LOCK = threading.Lock()
_DETECTION_MAX_SIDE = 1280
_LAMA_TARGET_SIZE = 256


def _tensor_to_uint8(image):
    array = image.detach().cpu().numpy()
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


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
        return ["big-lama.pt"]


def _get_inpaint_model_path(model_name):
    try:
        import folder_paths

        model_path = folder_paths.get_full_path("inpaint", model_name)
        if model_path is not None:
            return model_path
    except Exception:
        pass

    import os

    comfy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_path = os.path.join(comfy_root, "models", "inpaint", model_name)
    if os.path.exists(model_path):
        return model_path
    raise RuntimeError(f"Inpaint model file not found: {model_name}")


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


def _detect_watermark_mask_for_batch(image, languages, detection_mode, region, min_confidence, padding, dilate, blur, cv2_sensitivity, gpu):
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

    for batch_image in image:
        rgb = _tensor_to_uint8(batch_image)
        height, width = rgb.shape[:2]
        detection_rgb, detection_scale = _prepare_detection_rgb(rgb)
        mask = np.zeros((height, width), dtype=np.uint8)
        keep_region = _region_filter(width, height, region)
        detected_text = []

        if reader is not None:
            results = reader.readtext(detection_rgb)
            detected_text = _mask_ocr_results(mask, results, keep_region, min_confidence, padding, width, height, detection_scale)

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
                "detection_mode": (["ocr_then_cv2", "ocr_only", "cv2_only"], {"default": "ocr_then_cv2"}),
                "region": (["full_image", "edges", "corners"], {"default": "full_image"}),
                "min_confidence": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "padding": ("INT", {"default": 12, "min": 0, "max": 256, "step": 1}),
                "dilate": ("INT", {"default": 10, "min": 0, "max": 256, "step": 1}),
                "blur": ("INT", {"default": 7, "min": 0, "max": 255, "step": 1}),
                "cv2_sensitivity": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
                "gpu": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("mask", "mask_preview", "detected_text")
    FUNCTION = "detect"
    CATEGORY = "image/mask"

    def detect(self, image, languages, detection_mode, region, min_confidence, padding, dilate, blur, cv2_sensitivity, gpu):
        _rgb_batches, masks, previews, detected_batches = _detect_watermark_mask_for_batch(
            image, languages, detection_mode, region, min_confidence, padding, dilate, blur, cv2_sensitivity, gpu
        )

        return (torch.stack(masks, dim=0), torch.cat(previews, dim=0), "\n".join(detected_batches))


class AutoWatermarkRemover:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "languages": ("STRING", {"default": "en", "tooltip": "Used only when OCR modes are selected."}),
                "detection_mode": (["cv2_only", "ocr_then_cv2", "ocr_only"], {"default": "cv2_only"}),
                "region": (["full_image", "edges", "corners"], {"default": "full_image"}),
                "min_confidence": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "padding": ("INT", {"default": 10, "min": 0, "max": 256, "step": 1}),
                "dilate": ("INT", {"default": 5, "min": 0, "max": 256, "step": 1}),
                "blur": ("INT", {"default": 3, "min": 0, "max": 255, "step": 1}),
                "cv2_sensitivity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "inpaint_method": (["opencv_telea", "opencv_navier_stokes", "big_lama"], {"default": "opencv_telea"}),
                "inpaint_model": (_get_inpaint_model_names(), {"default": "big-lama.pt"}),
                "inpaint_radius": ("INT", {"default": 5, "min": 1, "max": 64, "step": 1}),
                "gpu": ("BOOLEAN", {"default": False}),
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
        inpaint_method=None,
        inpaint_model=None,
        inpaint_radius=None,
        gpu=None,
    ):
        return (
            languages,
            detection_mode,
            region,
            min_confidence,
            padding,
            dilate,
            blur,
            cv2_sensitivity,
            inpaint_method,
            inpaint_model,
            inpaint_radius,
            gpu,
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
        inpaint_method,
        inpaint_model,
        inpaint_radius,
        gpu,
    ):
        rgb_batches, masks, previews, detected_batches = _detect_watermark_mask_for_batch(
            image, languages, detection_mode, region, min_confidence, padding, dilate, blur, cv2_sensitivity, gpu
        )

        if inpaint_method == "big_lama":
            cleaned = _inpaint_with_lama(image, masks, inpaint_model)
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
