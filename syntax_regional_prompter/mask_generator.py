from typing import List

try:
    import torch
except ModuleNotFoundError:
    torch = None


def generate_mask(box: List[int], max_width: int, max_height: int):
    """
    Generate a binary mask tensor with shape [1, max_height, max_width].

    Args:
        box: [x, y, w, h]
        max_width: mask width
        max_height: mask height
    """
    if torch is None:
        raise RuntimeError("PyTorch is not installed in current Python environment")

    if len(box) != 4:
        raise ValueError("box must be [x, y, w, h]")
    if max_width <= 0 or max_height <= 0:
        raise ValueError("max_width and max_height must be positive")

    x, y, w, h = [int(v) for v in box]

    # Clamp to bounds to avoid index errors on malformed input.
    x0 = max(0, min(x, max_width))
    y0 = max(0, min(y, max_height))
    x1 = max(0, min(x + max(0, w), max_width))
    y1 = max(0, min(y + max(0, h), max_height))

    mask = torch.zeros((1, max_height, max_width), dtype=torch.float32)
    if x1 > x0 and y1 > y0:
        mask[:, y0:y1, x0:x1] = 1.0
    return mask


if __name__ == "__main__":
    if torch is None:
        print("PyTorch not found. Please run this script in your ComfyUI Python environment.")
        raise SystemExit(1)

    test_box = [0, 0, 256, 256]
    test_w = 512
    test_h = 512

    test_mask = generate_mask(test_box, test_w, test_h)

    print("Input box:", test_box)
    print("Mask shape:", tuple(test_mask.shape))

    print("Top-left corner (expect 1s):")
    print(test_mask[0, 0:4, 0:4])

    print("Bottom-right corner (expect 0s):")
    print(test_mask[0, 508:512, 508:512])

    print("Boundary check around x=255..257, y=255..257:")
    print(test_mask[0, 254:258, 254:258])
