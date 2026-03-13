import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


ANCHOR_PATTERN = re.compile(r"\{([^{}|]+)\|([^{}]+)\}")


@dataclass
class AnchorGroup:
    anchor: str
    attributes: List[str]


@dataclass
class AnchorParseResult:
    global_text: str
    groups: List[AnchorGroup]
    flattened_text: str


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,+", ",", text)
    return text.strip(" ,")



def parse_anchor_syntax(prompt: str) -> AnchorParseResult:
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")

    groups: List[AnchorGroup] = []
    global_parts: List[str] = []
    last_idx = 0

    for match in ANCHOR_PATTERN.finditer(prompt):
        start, end = match.span()
        global_parts.append(prompt[last_idx:start])
        last_idx = end

        anchor = _normalize_text(match.group(1))
        attrs_raw = match.group(2)
        attributes = [_normalize_text(x) for x in attrs_raw.split(",")]
        attributes = [x for x in attributes if x]
        if not anchor or not attributes:
            continue
        groups.append(AnchorGroup(anchor=anchor, attributes=attributes))

    global_parts.append(prompt[last_idx:])
    global_text = _normalize_text(" ".join(global_parts))

    flat_parts: List[str] = []
    if global_text:
        flat_parts.append(global_text)
    for group in groups:
        flat_parts.append(group.anchor)
        flat_parts.extend(group.attributes)
    flattened_text = ", ".join([p for p in flat_parts if p])

    return AnchorParseResult(global_text=global_text, groups=groups, flattened_text=flattened_text)


def _pick_primary_stream(tokenized: Dict) -> Tuple[str, List]:
    if "l" in tokenized:
        return "l", tokenized["l"]
    if "g" in tokenized:
        return "g", tokenized["g"]
    key = next(iter(tokenized.keys()))
    return key, tokenized[key]


def _extract_token_ids(token_batches: List[List[Tuple]]) -> List[int]:
    token_ids: List[int] = []
    for batch in token_batches:
        for item in batch:
            if not item:
                continue
            token_ids.append(int(item[0]))
    return token_ids


def _get_stream_batches(tokenized: Dict, stream_name: str) -> List[List[Tuple]]:
    return tokenized.get(stream_name) or next(iter(tokenized.values()))


def _get_content_token_ids(clip, text: str, stream_name: str) -> List[int]:
    """Extract content BPE token IDs (excluding BOS/EOS/padding) for *text*."""
    text_tok = clip.tokenize(text, return_word_ids=False)
    empty_tok = clip.tokenize("", return_word_ids=False)
    text_ids = _extract_token_ids(_get_stream_batches(text_tok, stream_name))
    empty_ids = _extract_token_ids(_get_stream_batches(empty_tok, stream_name))
    content: List[int] = []
    for i in range(len(text_ids)):
        if i < len(empty_ids):
            if text_ids[i] != empty_ids[i]:
                content.append(text_ids[i])
        else:
            content.append(text_ids[i])
    return content


def _find_subsequence(full_ids: List[int], sub_ids: List[int], start_from: int = 0) -> List[int]:
    """Find the first occurrence of *sub_ids* within *full_ids* starting at *start_from*.

    Returns the list of matching positions, or [] if not found.
    """
    if not sub_ids:
        return []
    sub_len = len(sub_ids)
    for i in range(start_from, len(full_ids) - sub_len + 1):
        if full_ids[i : i + sub_len] == sub_ids:
            return list(range(i, i + sub_len))
    return []


def build_anchor_token_map(clip, parsed: AnchorParseResult, precomputed_tokens: Dict = None) -> Dict:
    """Build a mapping from anchor/attribute text to token positions.

    Uses subsequence matching: tokenise each part individually, then locate
    its BPE token sequence inside the full tokenised flattened text.  This
    correctly skips comma-separator tokens and is robust against BPE
    merging and word-id assignment issues.

    If *precomputed_tokens* is provided (the return value of
    ``clip.tokenize(parsed.flattened_text)``), it will be reused directly
    instead of re-tokenising.  This guarantees that the token map is built
    against the **exact same** token stream used for conditioning, avoiding
    any theoretical inconsistency from double-tokenisation.
    """
    tokenized = precomputed_tokens if precomputed_tokens is not None else clip.tokenize(parsed.flattened_text, return_word_ids=False)
    stream_name, _ = _pick_primary_stream(tokenized)
    full_ids = _extract_token_ids(_get_stream_batches(tokenized, stream_name))

    # Advance search cursor past global text tokens.
    search_cursor = 0
    if parsed.global_text:
        global_content = _get_content_token_ids(clip, parsed.global_text, stream_name)
        if global_content:
            pos = _find_subsequence(full_ids, global_content, search_cursor)
            if pos:
                search_cursor = pos[-1] + 1

    group_maps: List[Dict] = []
    for group in parsed.groups:
        # --- anchor ---
        anchor_content = _get_content_token_ids(clip, group.anchor, stream_name)
        anchor_positions = _find_subsequence(full_ids, anchor_content, search_cursor)
        anchor_idx = list(anchor_positions)
        if anchor_positions:
            search_cursor = anchor_positions[-1] + 1

        # --- attributes ---
        attribute_indices: List[List[int]] = []
        for attr in group.attributes:
            attr_content = _get_content_token_ids(clip, attr, stream_name)
            attr_positions = _find_subsequence(full_ids, attr_content, search_cursor)
            attribute_indices.append(list(attr_positions))
            if attr_positions:
                search_cursor = attr_positions[-1] + 1

        group_maps.append(
            {
                "anchor_text": group.anchor,
                "attribute_texts": group.attributes,
                "anchor_idx": anchor_idx,
                "attribute_indices": attribute_indices,
            }
        )

    return {
        "flattened_text": parsed.flattened_text,
        "token_stream": stream_name,
        "groups": group_maps,
    }

