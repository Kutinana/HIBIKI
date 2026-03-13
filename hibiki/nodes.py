import json
import math

import torch

from .anchor_parser import build_anchor_token_map, parse_anchor_syntax


class HIBIKIAttentionPatcher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                "logit_floor": ("FLOAT", {"default": -30.0, "min": -100000.0, "max": -1.0, "step": 1.0}),
                "anchor_quantile": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "attr_inside_boost": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "anchor_inside_boost": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "other_anchor_suppress": ("FLOAT", {"default": -4.0, "min": -20.0, "max": 0.0, "step": 0.1}),
                "cond_only": ("BOOLEAN", {"default": True}),
                "exclusive_attribute_isolation": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("model", "conditioning", "flattened_text", "token_map_json")
    FUNCTION = "patch_model"
    CATEGORY = "advanced/model"

    def patch_model(
        self,
        model,
        clip,
        text,
        start_percent=0.0,
        end_percent=0.5,
        logit_floor=-30.0,
        anchor_quantile=0.45,
        attr_inside_boost=4.0,
        anchor_inside_boost=1.5,
        other_anchor_suppress=-4.0,
        cond_only=True,
        exclusive_attribute_isolation=True,
    ):
        if float(start_percent) > float(end_percent):
            raise ValueError("start_percent must be <= end_percent")

        parsed = parse_anchor_syntax(text)
        if not parsed.groups:
            raise ValueError("No anchor structure found. Expected syntax like: A street, {cat | white, fluffy}")

        # ---- SINGLE tokenisation: used for BOTH token-map AND conditioning ----
        tokens = clip.tokenize(parsed.flattened_text)
        token_map = build_anchor_token_map(clip, parsed, precomputed_tokens=tokens)
        group_maps = []
        for group in token_map.get("groups", []):
            anchor_idx = [int(i) for i in group.get("anchor_idx", [])]
            attribute_indices = [[int(i) for i in grp] for grp in group.get("attribute_indices", [])]
            attr_flat = sorted({i for grp in attribute_indices for i in grp})
            group_maps.append(
                {
                    "anchor_text": group.get("anchor_text", ""),
                    "attribute_texts": group.get("attribute_texts", []),
                    "anchor_idx": anchor_idx,
                    "attribute_indices": attribute_indices,
                    "attr_flat": attr_flat,
                }
            )

        # Diagnostic: print token mapping and budget warning.
        total_attr_tokens = sum(len(g["attr_flat"]) for g in group_maps)
        max_anchor_idx = max((max(g["anchor_idx"]) if g["anchor_idx"] else 0) for g in group_maps)
        max_attr_idx = max(
            (max(g["attr_flat"]) if g["attr_flat"] else 0) for g in group_maps
        )
        last_content_pos = max(max_anchor_idx, max_attr_idx)

        for g in group_maps:
            print(f"[HIBIKI] Token map: anchor={g['anchor_text']!r} idx={g['anchor_idx']}, "
                  f"attrs={list(zip(g['attribute_texts'], g['attribute_indices']))}")

        if last_content_pos >= 70:
            print(f"[HIBIKI] ⚠ Token budget tight: last content token at position {last_content_pos}/76. "
                  "Consider shortening the prompt to avoid truncation.")
        print(f"[HIBIKI] Total attribute tokens: {total_attr_tokens}, "
              f"groups: {len(group_maps)}, last content pos: {last_content_pos}")

        patched_model = model.clone()
        model_sampling = patched_model.get_model_object("model_sampling")
        sigma_start = float(model_sampling.percent_to_sigma(float(start_percent)))
        sigma_end = float(model_sampling.percent_to_sigma(float(end_percent)))

        has_replace_logged = {"value": False}

        # Capture tuning parameters for use inside the closure.
        _attr_inside_boost = float(attr_inside_boost)
        _anchor_inside_boost = float(anchor_inside_boost)
        _other_anchor_suppress = float(other_anchor_suppress)
        _logit_floor = float(logit_floor)
        _anchor_quantile = max(0.0, min(1.0, float(anchor_quantile)))
        _cond_only = bool(cond_only)
        _exclusive = bool(exclusive_attribute_isolation)

        enable_replace = any(g["anchor_idx"] and g["attr_flat"] for g in group_maps)
        if not enable_replace:
            print("[HIBIKI] Warning: token mapping is empty — attention patching disabled.")

        def _fallback_attention(q, k, v, extra_options):
            """Compute standard attention when the patch should not be active."""
            from comfy.ldm.modules.attention import optimized_attention

            n_heads = int(extra_options.get("n_heads", 1))
            attn_precision = extra_options.get("attn_precision", q.dtype)
            return optimized_attention(q, k, v, n_heads, attn_precision=attn_precision)

        def my_attn2_replace(q, k, v, extra_options):
            eps = 1e-6

            # ---- gate: check sigma range ----
            sigmas = extra_options.get("sigmas", None)
            if sigmas is None:
                return _fallback_attention(q, k, v, extra_options)
            current_sigma = float(sigmas[0])
            if not (sigma_end <= current_sigma <= sigma_start):
                return _fallback_attention(q, k, v, extra_options)

            batch_size, q_tokens, inner_dim = q.shape
            k_tokens = k.shape[1]
            n_heads = int(extra_options.get("n_heads", 1))
            dim_head = int(extra_options.get("dim_head", inner_dim // max(1, n_heads)))
            if n_heads <= 0 or dim_head <= 0 or n_heads * dim_head != inner_dim:
                return _fallback_attention(q, k, v, extra_options)

            valid_groups = []
            for group in group_maps:
                valid_anchor = [i for i in group["anchor_idx"] if 0 <= i < k_tokens]
                valid_attrs = [i for i in group["attr_flat"] if 0 <= i < k_tokens]
                if valid_anchor and valid_attrs:
                    valid_groups.append({"anchor": valid_anchor, "attrs": valid_attrs})

            if not valid_groups:
                return _fallback_attention(q, k, v, extra_options)

            # De-duplicate shared anchor tokens across groups.
            if len(valid_groups) > 1:
                shared_anchor = set(valid_groups[0]["anchor"])
                for g in valid_groups[1:]:
                    shared_anchor &= set(g["anchor"])
                if shared_anchor:
                    for g in valid_groups:
                        filtered = [t for t in g["anchor"] if t not in shared_anchor]
                        if filtered:
                            g["anchor"] = filtered

            # Per-group effective boost: normalise so that the COLLECTIVE
            # attribute attention stays roughly constant (~60-70%) regardless
            # of how many attribute tokens a group has.
            # Reference: 2 attr tokens at the base boost.  For more tokens
            # the per-token boost shrinks by ln(2/n); for fewer it grows.
            _N_REF = 2.0
            group_eff_boost = []
            for group in valid_groups:
                n_attr = max(len(group["attrs"]), 1)
                eff = _attr_inside_boost + math.log(_N_REF / n_attr)
                group_eff_boost.append(max(eff, 0.5))

            # Reshape to multi-head: [B, H, T, D]
            qh = q.view(batch_size, q_tokens, n_heads, dim_head).permute(0, 2, 1, 3)
            kh = k.view(batch_size, k_tokens, n_heads, dim_head).permute(0, 2, 1, 3)
            vh = v.view(batch_size, k_tokens, n_heads, dim_head).permute(0, 2, 1, 3)

            scores = torch.matmul(qh.float(), kh.float().transpose(-1, -2)) * (dim_head ** -0.5)

            cond_or_uncond = extra_options.get("cond_or_uncond", None)
            n_groups = len(valid_groups)

            for b in range(batch_size):
                if _cond_only and cond_or_uncond is not None and b < len(cond_or_uncond) and cond_or_uncond[b] != 0:
                    continue

                # Step 1: Compute RAW mean attention to each group's anchor → [G, H, Q]
                raw_anchor_attn = []
                for group in valid_groups:
                    attn = scores[b, :, :, group["anchor"]].mean(dim=-1)  # [H, Q]
                    raw_anchor_attn.append(attn)

                if _exclusive and n_groups > 1:
                    # --- MULTI-GROUP EXCLUSIVE MODE ---
                    # Z-score each group's anchor attention INDEPENDENTLY
                    # before comparing.  This removes the positional bias
                    # where earlier tokens get systematically higher raw
                    # scores, causing one group to dominate assignment.
                    z_per_group = []
                    for raw_attn in raw_anchor_attn:
                        mu = raw_attn.mean(dim=-1, keepdim=True)    # [H, 1]
                        sigma = raw_attn.std(dim=-1, keepdim=True).clamp(min=eps)
                        z_per_group.append((raw_attn - mu) / sigma)  # [H, Q]

                    stacked_z = torch.stack(z_per_group, dim=0)    # [G, H, Q]
                    assigned_group = torch.argmax(stacked_z, dim=0) # [H, Q]

                    # "active" zone: the winning anchor's z-score is
                    # above-quantile → position is near SOME anchor.
                    best_z = stacked_z.max(dim=0).values            # [H, Q]
                    thresh = torch.quantile(best_z, q=_anchor_quantile, dim=-1, keepdim=True)
                    active = best_z >= thresh                       # [H, Q]

                    # Precompute per-group masks
                    owner_masks = []
                    for g_idx in range(n_groups):
                        owner_masks.append((assigned_group == g_idx) & active)

                    # any_active = union of all groups' regions
                    any_active = active

                    for g_idx, group in enumerate(valid_groups):
                        is_owner = owner_masks[g_idx]       # [H, Q]
                        is_rival = any_active & ~is_owner   # [H, Q]
                        # is_neutral = ~any_active (implicitly: do nothing)

                        eff_boost = group_eff_boost[g_idx]

                        # Boost own anchor in own region.
                        for token_idx in group["anchor"]:
                            scores[b, :, :, token_idx] = torch.where(
                                is_owner,
                                scores[b, :, :, token_idx] + _anchor_inside_boost,
                                scores[b, :, :, token_idx],
                            )

                        # Suppress rival anchors in own region.
                        for j, other_group in enumerate(valid_groups):
                            if j == g_idx:
                                continue
                            for token_idx in other_group["anchor"]:
                                scores[b, :, :, token_idx] = torch.where(
                                    is_owner,
                                    scores[b, :, :, token_idx] + _other_anchor_suppress,
                                    scores[b, :, :, token_idx],
                                )

                        # THREE-ZONE attribute handling (with normalised boost):
                        #   own region  → boost (scaled by group attr count)
                        #   rival region → suppress (logit_floor)
                        #   neutral zone → UNCHANGED (let model decide)
                        for token_idx in group["attrs"]:
                            original = scores[b, :, :, token_idx]
                            scores[b, :, :, token_idx] = torch.where(
                                is_owner,
                                original + eff_boost,
                                torch.where(
                                    is_rival,
                                    torch.full_like(original, _logit_floor),
                                    original,  # neutral: unchanged
                                ),
                            )
                else:
                    # --- SINGLE-GROUP MODE ---
                    anchor_attn = raw_anchor_attn[0]  # [H, Q]
                    centered = anchor_attn - anchor_attn.mean(dim=-1, keepdim=True)
                    z_scored = centered / (centered.std(dim=-1, keepdim=True) + eps)
                    thresh = torch.quantile(z_scored, q=_anchor_quantile, dim=-1, keepdim=True)
                    contour = z_scored >= thresh  # [H, Q]
                    outside = ~contour

                    eff_boost = group_eff_boost[0]
                    group = valid_groups[0]
                    for token_idx in group["anchor"]:
                        scores[b, :, :, token_idx] = torch.where(
                            contour,
                            scores[b, :, :, token_idx] + _anchor_inside_boost,
                            scores[b, :, :, token_idx],
                        )

                    for token_idx in group["attrs"]:
                        scores[b, :, :, token_idx] = torch.where(
                            outside,
                            torch.full_like(scores[b, :, :, token_idx], _logit_floor),
                            scores[b, :, :, token_idx] + eff_boost,
                        )

            attn = torch.softmax(scores, dim=-1).to(vh.dtype)
            out = torch.matmul(attn, vh)
            out = out.permute(0, 2, 1, 3).reshape(batch_size, q_tokens, inner_dim)

            if not has_replace_logged["value"]:
                s_mean = scores.mean().item()
                s_std = scores.std().item()
                s_min = scores.min().item()
                s_max = scores.max().item()
                print(f"[HIBIKI] Attention replace active — groups: {len(valid_groups)}")
                print(f"[HIBIKI] Score stats (first call): mean={s_mean:.2f} std={s_std:.2f} "
                      f"min={s_min:.2f} max={s_max:.2f}")
                print(f"[HIBIKI] Base boost params: attr={_attr_inside_boost:.1f} anchor={_anchor_inside_boost:.1f} "
                      f"suppress={_other_anchor_suppress:.1f} floor={_logit_floor:.1f}")
                for g_idx, group in enumerate(valid_groups):
                    n_attr = len(group["attrs"])
                    eff = group_eff_boost[g_idx]
                    pct_str = ""
                    if _exclusive and n_groups > 1:
                        pct = owner_masks[g_idx].float().mean().item() * 100
                        pct_str = f"  region={pct:.1f}%"
                    print(f"[HIBIKI]   Group {g_idx}: {n_attr} attr tokens, "
                          f"eff_boost={eff:.2f}{pct_str}")
                has_replace_logged["value"] = True
            return out

        if enable_replace:
            for block_id in range(32):
                patched_model.set_model_attn2_replace(my_attn2_replace, "input", block_id)
                patched_model.set_model_attn2_replace(my_attn2_replace, "output", block_id)
            patched_model.set_model_attn2_replace(my_attn2_replace, "middle", 0)

        # Encode conditioning from the SAME tokenisation used for the token map.
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        token_map_json = json.dumps(
            {
                "groups": [
                    {
                        "anchor_text": g["anchor_text"],
                        "attribute_texts": g["attribute_texts"],
                        "anchor_idx": g["anchor_idx"],
                        "attribute_indices": g["attribute_indices"],
                    }
                    for g in group_maps
                ]
            },
            ensure_ascii=False,
        )

        return (patched_model, conditioning, parsed.flattened_text, token_map_json)


NODE_CLASS_MAPPINGS = {
    "HIBIKIAttentionPatcher": HIBIKIAttentionPatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HIBIKIAttentionPatcher": "HIBIKI Attention Patcher",
}

