from .mask_generator import generate_mask
from .parser import parse_hibiki_prompt


def _set_values(conditioning, values):
    out = []
    for cond_entry in conditioning:
        tensor = cond_entry[0]
        meta = cond_entry[1].copy()
        meta.update(values)
        out.append([tensor, meta])
    return out


class HIBIKIRegionalPrompter:
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
                "image_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "image_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "mask_blur_radius": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "mask_blur_sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 32.0, "step": 0.1}),
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
        width = ((max_x + 7) // 8) * 8
        height = ((max_y + 7) // 8) * 8
        return width, height

    def _resolve_canvas_size(self, regions, image_width=0, image_height=0):
        if image_width > 0 and image_height > 0:
            width = max(64, ((int(image_width) + 7) // 8) * 8)
            height = max(64, ((int(image_height) + 7) // 8) * 8)
            return width, height
        return self._infer_canvas_size(regions)

    def _clamp_box(self, box, canvas_w, canvas_h):
        x, y, w, h = [int(v) for v in box]
        x0 = max(0, min(x, canvas_w))
        y0 = max(0, min(y, canvas_h))
        x1 = max(0, min(x + max(0, w), canvas_w))
        y1 = max(0, min(y + max(0, h), canvas_h))
        return x0, y0, max(0, x1 - x0), max(0, y1 - y0)

    def build_conditioning(
        self,
        text,
        clip,
        strength=1.0,
        set_cond_area="default",
        image_width=0,
        image_height=0,
        mask_blur_radius=0,
        mask_blur_sigma=1.0,
    ):
        has_grid_syntax = "grid:" in text.lower()
        if has_grid_syntax and (int(image_width) <= 0 or int(image_height) <= 0):
            raise ValueError("grid syntax requires image_width and image_height inputs")

        if has_grid_syntax:
            canvas_w, canvas_h = self._resolve_canvas_size([], image_width=image_width, image_height=image_height)
            parsed = parse_hibiki_prompt(
                text,
                strict=False,
                canvas_width=canvas_w,
                canvas_height=canvas_h,
            )
        else:
            parsed = parse_hibiki_prompt(text, strict=False)
        out = []

        if parsed.global_text:
            out.extend(self._encode_text(clip, parsed.global_text))

        if not parsed.regions:
            if out:
                return (out,)
            return (self._encode_text(clip, text),)

        if has_grid_syntax:
            canvas_w, canvas_h = self._resolve_canvas_size([], image_width=image_width, image_height=image_height)
        else:
            canvas_w, canvas_h = self._resolve_canvas_size(
                parsed.regions, image_width=image_width, image_height=image_height
            )
        set_area_to_bounds = set_cond_area != "default"

        for region in parsed.regions:
            x, y, w, h = self._clamp_box(region.box, canvas_w, canvas_h)
            region_cond = self._encode_text(clip, region.text)
            region_mask = generate_mask(
                [x, y, w, h],
                canvas_w,
                canvas_h,
                blur_radius=int(mask_blur_radius),
                blur_sigma=float(mask_blur_sigma),
            )

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
    "HIBIKIRegionalPrompter": HIBIKIRegionalPrompter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HIBIKIRegionalPrompter": "HIBIKI Regional Prompter",
}

