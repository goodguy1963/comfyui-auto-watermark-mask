import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import auto_watermark_mask as awm


class DummyLaMaModel(torch.nn.Module):
    def forward(self, image, mask):
        blurred = torch.nn.functional.avg_pool2d(image, kernel_size=5, stride=1, padding=2)
        return image * (1.0 - mask) + blurred * mask


def build_synthetic_rgb(height, width):
    image = np.full((height, width, 3), 240, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (width - 1, height - 1), (210, 210, 210), 6)
    for row_index in range(6):
        y = 80 + row_index * max(70, height // 9)
        cv2.putText(
            image,
            "WATERMARK SAMPLE",
            (max(20, width // 16), min(height - 40, y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.9, width / 1400.0),
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
    return image


def build_synthetic_gemini_rgb(height, width):
    original = build_synthetic_rgb(height, width)
    size = awm._GEMINI_SMALL_SIZE if min(height, width) <= 1024 else awm._GEMINI_LARGE_SIZE
    margin = awm._GEMINI_SMALL_MARGIN if size <= awm._GEMINI_SMALL_SIZE else awm._GEMINI_LARGE_MARGIN
    x = max(0, width - size - margin)
    y = max(0, height - size - margin)
    template = awm._get_gemini_template(size)

    watermarked = original.astype(np.float32) / 255.0
    roi = watermarked[y : y + size, x : x + size]
    alpha = template["alpha"][..., None]
    logo = template["logo"]
    watermarked[y : y + size, x : x + size] = alpha * logo + (1.0 - alpha) * roi
    return original, np.clip(watermarked * 255.0, 0, 255).astype(np.uint8)


def build_mask(height, width):
    mask = np.zeros((height, width), dtype=np.float32)
    cv2.rectangle(mask, (width // 12, height // 10), (width * 10 // 12, height * 3 // 4), 1.0, thickness=-1)
    return torch.from_numpy(mask)


def timed(label, repeat, fn):
    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    mean_seconds = statistics.mean(timings)
    print(f"{label}: mean={mean_seconds:.6f}s min={min(timings):.6f}s max={max(timings):.6f}s")
    return mean_seconds


def run_detection_benchmark(height, width, repeat, sensitivity, padding):
    rgb = build_synthetic_rgb(height, width)
    keep_region = awm._region_filter(width, height, "full_image")

    def current_detection():
        detection_rgb, detection_scale = awm._prepare_detection_rgb(rgb)
        mask = np.zeros((height, width), dtype=np.uint8)
        awm._mask_cv2_text_like_regions(mask, detection_rgb, keep_region, padding, sensitivity, width, height, detection_scale)
        return mask

    def decomposition_detection():
        detection_rgb, detection_scale = awm._prepare_detection_rgb(rgb)
        mask = np.zeros((height, width), dtype=np.uint8)
        awm._mask_decomposition_regions(mask, detection_rgb, keep_region, padding, sensitivity, width, height, detection_scale)
        return mask

    def legacy_detection():
        mask = np.zeros((height, width), dtype=np.uint8)
        awm._mask_cv2_text_like_regions(mask, rgb, keep_region, padding, sensitivity, width, height, 1.0)
        return mask

    legacy_time = timed("legacy_cv2_detection", repeat, legacy_detection)
    current_time = timed("current_cv2_detection", repeat, current_detection)
    decomposition_time = timed("decomposition_detection", repeat, decomposition_detection)
    print(f"cv2_detection_speedup: {legacy_time / current_time:.2f}x")
    print(f"decomposition_vs_legacy_speed: {legacy_time / decomposition_time:.2f}x")


def legacy_inpaint_with_lama(image, masks):
    model = DummyLaMaModel().cpu()
    image_bchw = image.detach().permute(0, 3, 1, 2).cpu()
    mask_bchw = awm._mask_to_torch(torch.stack(masks, dim=0)).cpu()
    outputs = []

    with torch.no_grad():
        for index in range(image_bchw.shape[0]):
            work_image = image_bchw[index].unsqueeze(0)
            work_mask = mask_bchw[index].unsqueeze(0)
            work_image, work_mask, original_size = awm._resize_square(work_image, work_mask, 256)
            work_mask = (work_mask >= 0.99).to(work_image.dtype)
            result = model(work_image, work_mask)
            result = awm._undo_resize_square(result, original_size)
            original = image_bchw[index].unsqueeze(0)
            original_mask = (mask_bchw[index].unsqueeze(0) >= 0.99).to(original.dtype)
            result = original + (result - original) * original_mask
            outputs.append(result)

    return torch.cat(outputs, dim=0).permute(0, 2, 3, 1)


def run_lama_benchmark(batch_size, height, width, repeat):
    image = np.stack([build_synthetic_rgb(height, width) for _ in range(batch_size)], axis=0)
    image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
    masks = [build_mask(height, width) for _ in range(batch_size)]

    original_loader = awm._load_inpaint_model
    original_device = awm._get_torch_device
    awm._load_inpaint_model = lambda _model_name: DummyLaMaModel().cpu()
    awm._get_torch_device = lambda: torch.device("cpu")
    try:
        legacy_time = timed("legacy_lama_loop", repeat, lambda: legacy_inpaint_with_lama(image_tensor, masks))
        current_time = timed("current_lama_batch", repeat, lambda: awm._inpaint_with_lama(image_tensor, masks, "big-lama.pt"))
    finally:
        awm._load_inpaint_model = original_loader
        awm._get_torch_device = original_device

    print(f"lama_batch_speedup: {legacy_time / current_time:.2f}x")


def run_gemini_benchmark(height, width, repeat, cleanup_radius):
    original_rgb, watermarked_rgb = build_synthetic_gemini_rgb(height, width)
    image_tensor = torch.from_numpy(watermarked_rgb.astype(np.float32) / 255.0).unsqueeze(0)

    detect_time = timed(
        "gemini_detect",
        repeat,
        lambda: awm._detect_gemini_sparkle(watermarked_rgb, 0.25),
    )
    remove_time = timed(
        "gemini_remove",
        repeat,
        lambda: awm._remove_gemini_sparkle_for_batch(image_tensor, 0.25, cleanup_radius),
    )

    cleaned, masks, _previews, detected_text = awm._remove_gemini_sparkle_for_batch(image_tensor, 0.25, cleanup_radius)
    cleaned_rgb = np.clip(cleaned[0].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    mae = np.mean(np.abs(cleaned_rgb.astype(np.float32) - original_rgb.astype(np.float32)))
    mask_ratio = float(masks[0].mean().item())
    print(f"gemini_detect_time: {detect_time:.6f}s")
    print(f"gemini_remove_time: {remove_time:.6f}s")
    print(f"gemini_restore_mae: {mae:.4f}")
    print(f"gemini_mask_ratio: {mask_ratio:.6f}")
    print(f"gemini_detected: {detected_text}")


def collect_input_images(input_dir, limit):
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    candidates = [path for path in sorted(Path(input_dir).iterdir()) if path.is_file() and path.suffix.lower() in extensions]
    if limit is not None:
        candidates = candidates[:limit]
    return candidates


def load_image_tensor(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not load benchmark image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)


def timed_real_mode(label, repeat, paths, fn):
    timings = []
    mask_ratios = []
    hits = 0
    for _ in range(repeat):
        for path in paths:
            image_tensor = load_image_tensor(path)
            start = time.perf_counter()
            mask_ratio, hit = fn(image_tensor)
            timings.append(time.perf_counter() - start)
            mask_ratios.append(mask_ratio)
            hits += int(hit)
    mean_seconds = statistics.mean(timings)
    mean_ratio = statistics.mean(mask_ratios) if mask_ratios else 0.0
    total = max(1, len(paths) * repeat)
    print(f"{label}: mean={mean_seconds:.6f}s hit_rate={hits / total:.2%} mask_ratio={mean_ratio:.6f}")


def run_real_sample_benchmark(input_dir, repeat, limit):
    paths = collect_input_images(input_dir, limit)
    if not paths:
        print("real_sample_benchmark: no supported images found")
        return

    print(f"real_sample_count: {len(paths)}")

    def cv2_mode(image_tensor):
        _rgb_batches, masks, _previews, _text = awm._detect_watermark_mask_for_batch(
            image_tensor,
            "en",
            "cv2_only",
            "none",
            "full_image",
            0.25,
            10,
            5,
            3,
            0.8,
            False,
            "off",
            0.12,
            0.08,
        )
        ratio = float(masks[0].mean().item())
        return ratio, ratio > 0.0

    def decomposition_mode(image_tensor):
        _rgb_batches, masks, _previews, _text = awm._detect_watermark_mask_for_batch(
            image_tensor,
            "en",
            "cv2_only",
            "decomposition",
            "full_image",
            0.25,
            10,
            5,
            3,
            0.8,
            False,
            "off",
            0.12,
            0.08,
        )
        ratio = float(masks[0].mean().item())
        return ratio, ratio > 0.0

    def gemini_mode(image_tensor):
        _cleaned, masks, _previews, detected = awm._remove_gemini_sparkle_for_batch(image_tensor, 0.25, 5)
        ratio = float(masks[0].mean().item())
        return ratio, "not detected" not in detected.lower()

    timed_real_mode("real_cv2_only", repeat, paths, cv2_mode)
    timed_real_mode("real_decomposition", repeat, paths, decomposition_mode)
    timed_real_mode("real_gemini_reverse_alpha", repeat, paths, gemini_mode)


def main():
    parser = argparse.ArgumentParser(description="Benchmark the optimized watermark node against small legacy reference paths.")
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cv2-sensitivity", type=float, default=0.8)
    parser.add_argument("--padding", type=int, default=10)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--input-limit", type=int, default=None)
    args = parser.parse_args()

    print("== Detection benchmark ==")
    run_detection_benchmark(args.height, args.width, args.repeat, args.cv2_sensitivity, args.padding)
    print()
    print("== LaMa benchmark ==")
    run_lama_benchmark(args.batch_size, min(args.height, 1024), min(args.width, 1024), args.repeat)
    print()
    print("== Gemini benchmark ==")
    run_gemini_benchmark(min(args.height, 1536), min(args.width, 1536), args.repeat, cleanup_radius=5)

    if args.input_dir is not None:
        print()
        print("== Real sample benchmark ==")
        run_real_sample_benchmark(args.input_dir, max(1, min(args.repeat, 5)), args.input_limit)


if __name__ == "__main__":
    main()
