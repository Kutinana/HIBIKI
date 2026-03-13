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


def parse_hibiki_prompt(prompt: str, strict: bool = True) -> ParseResult:
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

    print("Input:", demo_prompt)
    print("Global:", result.global_text)
    print("Regions:")
    for i, region in enumerate(result.regions, start=1):
        print(f"  {i}. text={region.text!r}, box={region.box}, span={region.span}")
    print("Errors:", result.errors)

