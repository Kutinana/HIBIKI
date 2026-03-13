from .mask_generator import generate_mask
from .parser import parse_syntax


def _set_values(conditioning, values):
    out = []
    for cond_entry in conditioning:
        tensor = cond_entry[0]
        meta = cond_entry[1].copy()
        meta.update(values)
        out.append([tensor, meta])
    return out


class SyntaxRegionalPrompter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "clip": ("CLIP",),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "build_conditioning"
    CATEGORY = "conditioning"

    def _encode_text(self, clip, text):
        import nodes as comfy_nodes

        return comfy_nodes.CLIPTextEncode().encode(clip, text)[0]

    def _infer_canvas_size(self, regions):
        max_x = 64
        max_y = 64
        for region in regions:
            x, y, w, h = region.box
            max_x = max(max_x, x + max(0, w))
            max_y = max(max_y, y + max(0, h))
        # Match area node assumptions: multiples of 8.
        width = ((max_x + 7) // 8) * 8
        height = ((max_y + 7) // 8) * 8
        return width, height

    def build_conditioning(self, text, clip, strength=1.0, set_cond_area="default"):
        parsed = parse_syntax(text, strict=False)
        out = []

        if parsed.global_text:
            out.extend(self._encode_text(clip, parsed.global_text))

        if not parsed.regions:
            if out:
                return (out,)
            return (self._encode_text(clip, text),)

        canvas_w, canvas_h = self._infer_canvas_size(parsed.regions)
        set_area_to_bounds = set_cond_area != "default"

        for region in parsed.regions:
            x, y, w, h = region.box
            region_cond = self._encode_text(clip, region.text)
            region_mask = generate_mask(region.box, canvas_w, canvas_h)

            values = {
                "area": (h // 8, w // 8, y // 8, x // 8),
                "strength": float(strength),
                "mask": region_mask,
                "mask_strength": float(strength),
                "set_area_to_bounds": set_area_to_bounds,
            }
            out.extend(_set_values(region_cond, values))

        return (out,)


NODE_CLASS_MAPPINGS = {
    "SyntaxRegionalPrompter": SyntaxRegionalPrompter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SyntaxRegionalPrompter": "Syntax Regional Prompter",
}
