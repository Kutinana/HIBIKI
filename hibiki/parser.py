import re
from dataclasses import dataclass, field
from typing import List, Tuple


REGION_PATTERN = re.compile(r"\{([^{}]+)\}")


@dataclass
class RegionSpec:
    text: str
    box: List[int]
    span: Tuple[int, int]


@dataclass
class ParseResult:
    global_text: str
    regions: List[RegionSpec] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _normalize_global_text(raw_text: str) -> str:
    text = re.sub(r"\s+", " ", raw_text).strip()
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,+", ",", text)
    return text.strip(" ,")


def _parse_box(raw_box: str) -> List[int]:
    parts = [p.strip() for p in raw_box.split(",")]
    if len(parts) != 4:
        raise ValueError(f"box must contain 4 values, got {len(parts)}")
    try:
        _, _, _, _ = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError("box values must be integers") from exc
    return [int(p) for p in parts]


def _parse_grid_box(
    raw_grid: str,
    canvas_width: int,
    canvas_height: int,
    overlap_ratio: float = 0.0,
) -> List[int]:
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("grid syntax requires canvas_width and canvas_height > 0")

    kv_pairs = {}
    for part in raw_grid.split(","):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        kv_pairs[key.strip().lower()] = value.strip().lower()

    grid_value = kv_pairs.get("grid")
    pos_value = kv_pairs.get("pos")
    if grid_value is None or pos_value is None:
        raise ValueError("grid syntax must include both 'grid' and 'pos'")

    grid_match = re.fullmatch(r"(\d+)\s*[xX]\s*(\d+)", grid_value)
    if not grid_match:
        raise ValueError("grid must use format rowsxcols, e.g. 1x2")

    rows = int(grid_match.group(1))
    cols = int(grid_match.group(2))
    pos = int(pos_value)
    if rows <= 0 or cols <= 0:
        raise ValueError("grid rows and cols must be positive")
    if pos < 1 or pos > rows * cols:
        raise ValueError(f"pos must be within [1, {rows * cols}]")

    index = pos - 1
    row = index // cols
    col = index % cols

    x0 = (canvas_width * col) // cols
    y0 = (canvas_height * row) // rows
    x1 = (canvas_width * (col + 1)) // cols
    y1 = (canvas_height * (row + 1)) // rows

    if overlap_ratio > 0.0:
        cell_w = max(1, x1 - x0)
        cell_h = max(1, y1 - y0)
        dx = int(round(cell_w * overlap_ratio))
        dy = int(round(cell_h * overlap_ratio))
        x0 = max(0, x0 - dx)
        y0 = max(0, y0 - dy)
        x1 = min(canvas_width, x1 + dx)
        y1 = min(canvas_height, y1 + dy)

    return [x0, y0, x1 - x0, y1 - y0]


def parse_hibiki_prompt(
    prompt: str,
    strict: bool = True,
    canvas_width: int = 0,
    canvas_height: int = 0,
    grid_overlap_ratio: float = 0.0,
) -> ParseResult:
    """
    Parse syntax like:
        A street, {a cat | 0,0,256,256}
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")

    regions: List[RegionSpec] = []
    errors: List[str] = []

    global_parts: List[str] = []
    last_idx = 0

    for match in REGION_PATTERN.finditer(prompt):
        start, end = match.span()
        global_parts.append(prompt[last_idx:start])
        last_idx = end

        body = match.group(1).strip()
        if "|" not in body:
            msg = f"Missing '|' in region: {body!r}"
            if strict:
                raise ValueError(msg)
            errors.append(msg)
            continue

        text_part, box_part = [part.strip() for part in body.split("|", 1)]
        if not text_part:
            msg = f"Empty region text in: {body!r}"
            if strict:
                raise ValueError(msg)
            errors.append(msg)
            continue

        try:
            if "grid:" in box_part.lower():
                box = _parse_grid_box(
                    box_part,
                    canvas_width=canvas_width,
                    canvas_height=canvas_height,
                    overlap_ratio=float(grid_overlap_ratio),
                )
            else:
                box = _parse_box(box_part)
        except ValueError as exc:
            if strict:
                raise
            errors.append(f"{exc} in region: {body!r}")
            continue

        regions.append(RegionSpec(text=text_part, box=box, span=(start, end)))

    global_parts.append(prompt[last_idx:])
    global_text = _normalize_global_text(" ".join(global_parts))

    return ParseResult(global_text=global_text, regions=regions, errors=errors)


if __name__ == "__main__":
    demo_prompt = "A street, {a cat | 0,0,256,256}"
    result = parse_hibiki_prompt(demo_prompt, strict=True)
    grid_prompt = "A scene, {a dog | grid: 1x2, pos: 1}, {a cat | grid: 1x2, pos: 2}"
    grid_result = parse_hibiki_prompt(grid_prompt, strict=True, canvas_width=1024, canvas_height=512)
    grid_overlap_result = parse_hibiki_prompt(
        grid_prompt,
        strict=True,
        canvas_width=1024,
        canvas_height=512,
        grid_overlap_ratio=0.1,
    )

    print("Input:", demo_prompt)
    print("Global:", result.global_text)
    print("Regions:")
    for i, region in enumerate(result.regions, start=1):
        print(f"  {i}. text={region.text!r}, box={region.box}, span={region.span}")
    print("Errors:", result.errors)
    print("---")
    print("Grid Input:", grid_prompt)
    for i, region in enumerate(grid_result.regions, start=1):
        print(f"  {i}. text={region.text!r}, box={region.box}, span={region.span}")
    print("Grid Input (overlap=0.1):", grid_prompt)
    for i, region in enumerate(grid_overlap_result.regions, start=1):
        print(f"  {i}. text={region.text!r}, box={region.box}, span={region.span}")

