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
                "local_mask_base_ratio": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "local_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "local_end_percent": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.001}),
                "set_cond_area": (["default", "mask bounds"],),
                "region_mode": (["mask_only", "area_and_mask"],),
                "image_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "image_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "grid_overlap_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.45, "step": 0.01}),
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
        local_mask_base_ratio=0.8,
        local_start_percent=0.0,
        local_end_percent=0.4,
        set_cond_area="default",
        region_mode="mask_only",
        image_width=0,
        image_height=0,
        grid_overlap_ratio=0.1,
        mask_blur_radius=0,
        mask_blur_sigma=1.0,
    ):
        local_start_percent = float(local_start_percent)
        local_end_percent = float(local_end_percent)
        if local_start_percent > local_end_percent:
            raise ValueError("local_start_percent must be <= local_end_percent")

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
                grid_overlap_ratio=float(grid_overlap_ratio),
            )
        else:
            parsed = parse_hibiki_prompt(text, strict=False)
        out = []

        # Always include a global/base conditioning to stabilize scene semantics.
        global_prompt = parsed.global_text if parsed.global_text else ""
        out.extend(self._encode_text(clip, global_prompt))

        if not parsed.regions:
            return (out,)

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
                "strength": float(strength),
                "mask": region_mask,
                "mask_strength": float(strength) * float(local_mask_base_ratio),
                "set_area_to_bounds": set_area_to_bounds,
                "start_percent": local_start_percent,
                "end_percent": local_end_percent,
            }
            if region_mode == "area_and_mask":
                values["area"] = (h // 8, w // 8, y // 8, x // 8)
            out.extend(_set_values(region_cond, values))

        return (out,)


class HIBIKIAttentionPatcher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_model"
    CATEGORY = "advanced/model"

    def patch_model(self, model):
        patched_model = model.clone()
        has_logged = {"value": False}

        def my_custom_patch(q, k, v, extra_options):
            # Dummy hook for Step 3.1: verify hook path without changing attention behavior.
            if not has_logged["value"]:
                print("Hook triggered")
                has_logged["value"] = True
            return q, k, v

        patched_model.set_model_attn2_patch(my_custom_patch)
        return (patched_model,)


NODE_CLASS_MAPPINGS = {
    "HIBIKIRegionalPrompter": HIBIKIRegionalPrompter,
    "HIBIKIAttentionPatcher": HIBIKIAttentionPatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HIBIKIRegionalPrompter": "HIBIKI Regional Prompter",
    "HIBIKIAttentionPatcher": "HIBIKI Attention Patcher",
}

