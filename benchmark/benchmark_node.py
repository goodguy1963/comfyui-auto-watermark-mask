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

    def legacy_detection():
        mask = np.zeros((height, width), dtype=np.uint8)
        awm._mask_cv2_text_like_regions(mask, rgb, keep_region, padding, sensitivity, width, height, 1.0)
        return mask

    legacy_time = timed("legacy_cv2_detection", repeat, legacy_detection)
    current_time = timed("current_cv2_detection", repeat, current_detection)
    print(f"cv2_detection_speedup: {legacy_time / current_time:.2f}x")


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


def main():
    parser = argparse.ArgumentParser(description="Benchmark the optimized watermark node against small legacy reference paths.")
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cv2-sensitivity", type=float, default=0.8)
    parser.add_argument("--padding", type=int, default=10)
    args = parser.parse_args()

    print("== Detection benchmark ==")
    run_detection_benchmark(args.height, args.width, args.repeat, args.cv2_sensitivity, args.padding)
    print()
    print("== LaMa benchmark ==")
    run_lama_benchmark(args.batch_size, min(args.height, 1024), min(args.width, 1024), args.repeat)


if __name__ == "__main__":
    main()
