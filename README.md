# HIBIKI: Hierarchical Instruction Binding for Image Key-token Intervention

A ComfyUI custom node that mitigates concept drift (attribute leakage) in multi-subject Stable Diffusion generation by patching cross-attention at the token level.

## Syntax

```
global description, {subjectA | attrA1, attrA2, ...}, {subjectB | attrB1, attrB2, ...}, ...
```

## Example

Vanilla Stable Diffusion struggles to assign distinct attributes to multiple subjects. For instance, the following prompt:

```
1boy, black hair, very long hair, blue eyes, sitting,
1girl, white hair, short hair, yellow eyes, standing,
blank background, white background, masterpiece, 8k, high quality, best quality, absurd resolution, very awa
```

Conflicting attributes (hair colour, hair length, eye colour, pose) are randomly distributed across the two subjects. Both images below were generated directly from this prompt:

| Without HIBIKI | Without HIBIKI |
|:-:|:-:|
| ![](./samples/not_applied/ComfyUI_temp_knxqi_00005_.png) | ![](./samples/not_applied/ComfyUI_temp_knxqi_00007_.png) |

After applying HIBIKI's syntax and attention patching (only the special brackets were added to the same prompt):

```
{1boy | black hair, very long hair, blue eyes, sitting},
{1girl | white hair, short hair, yellow eyes, standing},
blank background, white background, masterpiece, 8k, high quality, best quality, absurd resolution, very awa
```

The probability of generating correctly attributed subjects increases significantly:

| With HIBIKI | With HIBIKI |
|:-:|:-:|
| ![](./samples/applied_hibiki/ComfyUI_temp_knxqi_00001_.png) | ![](./samples/applied_hibiki/ComfyUI_temp_knxqi_00003_.png) |
