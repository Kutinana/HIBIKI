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
| Boy: black short hair, yellow eyes, standing | Boy: white short hair, blue eyes, sitting |
| Girl: white long hair, blue eyes, sitting | Girl: black long hair, yellow eyes, standing |

After applying HIBIKI's syntax and attention patching (only the special brackets were added to the same prompt):

```
{1boy | black hair, very long hair, blue eyes, sitting},
{1girl | white hair, short hair, yellow eyes, standing},
blank background, white background, masterpiece, 8k, high quality, best quality, absurd resolution, very awa
```

The probability of generating correctly attributed subjects increases significantly:

| With HIBIKI | With HIBIKI |
|:-:|:-:|
| ![](./samples/applied_hibiki/ComfyUI_temp_knxqi_00009_.png) | ![](./samples/applied_hibiki/ComfyUI_temp_knxqi_00003_.png) |

## Implementation

### Problem Formulation

In the standard Stable Diffusion U-Net, every denoising step computes cross-attention between spatial queries and text-token keys.  Let $Q \in \mathbb{R}^{H \times N \times d}$ denote the multi-head query from the spatial latent (where $N = h \times w$ is the number of spatial positions and $H$ the number of attention heads), and let $K, V \in \mathbb{R}^{H \times L \times d}$ denote the key/value projections from the $L$-token CLIP embedding.  The standard attention output is:

$$
A = \text{softmax}\!\Bigl(\frac{Q K^\top}{\sqrt{d}}\Bigr), \quad O = A \, V
$$

Because softmax normalises across the *entire* key axis, attribute tokens from different subjects compete in the same distribution.  This causes **attribute leakage**.

### Core Approach: Three-Zone Spatially-Selective Score Modification

HIBIKI replaces the cross-attention function in all transformer blocks of the U-Net (via ComfyUI's `set_model_attn2_replace`) during a configurable fraction of the denoising schedule.  Rather than altering the model weights, it directly manipulates the **pre-softmax attention logits** $S = Q K^\top / \sqrt{d}$ on a per-head, per-spatial-position basis.

#### Step 1 — BPE-Level Token Mapping

The user prompt is parsed into anchor groups. To resolve the prompt text to exact BPE token positions, HIBIKI tokenises each segment independently and locates its BPE subsequence within the full tokenised prompt via linear scan.

#### Step 2 — Per-Group Z-Scored Region Assignment

For each group $g$, the mean attention of all spatial positions to its anchor tokens is computed:

$$
\bar{s}_g^{(h)}(n) = \frac{1}{|\mathcal{A}_g|} \sum_{j \in \mathcal{A}_g} S^{(h)}(n, j)
$$

where $\mathcal{A}_g$ is the set of anchor token indices for group $g$, and $h$ indexes the attention head.  To eliminate positional bias (earlier tokens in CLIP's causal embedding tend to have systematically higher raw scores), each group's anchor attention is independently z-normalised:

$$
z_g^{(h)}(n) = \frac{\bar{s}_g^{(h)}(n) - \mu_g^{(h)}}{\sigma_g^{(h)} + \epsilon}
$$

Region ownership is then assigned by $\text{argmax}_g \, z_g^{(h)}(n)$, and positions whose winning z-score falls below a configurable quantile threshold $q$ are marked as **neutral** (unmodified).  This yields three disjoint zones per group at every head:

| Zone | Definition | Action |
|:---|:---|:---|
| **Owner** | Position belongs to group $g$ and is above the activity threshold | Boost anchor & attribute scores |
| **Rival** | Position belongs to another group $g' \neq g$ and is above the threshold | Suppress attribute scores to logit floor |
| **Neutral** | Position is below the activity threshold (background / shared context) | Leave scores unmodified |

#### Step 3 — Score Modification with Adaptive Boost Scaling

Within each group's **owner** zone, the logits are additively modified:

- **Own anchor tokens** receive $+ \beta_{\text{anchor}}$
- **Own attribute tokens** receive $+ \beta_{\text{attr}}^{(g)}$, an *adaptive* per-group boost
- **Rival anchor tokens** receive $+ \gamma_{\text{suppress}}$ (negative)

Within each group's **rival** zone:

- **Other groups' attribute tokens** are hard-clamped to a **logit floor** $f \ll 0$ (default −30), effectively zeroing their post-softmax contribution

Within the **neutral** zone, all scores are left untouched, preserving the model's native compositional reasoning for background elements and global context.

**Adaptive boost scaling.**  A fixed per-token boost causes attribute dilution when many attributes are present: $k$ boosted tokens each contribute $\sim e^{\beta}$ in the softmax numerator, collectively consuming a share that grows with $k$ and squeezes out structural tokens. HIBIKI compensates by scaling the per-token boost as:

$$
\beta_{\text{attr}}^{(g)} = \beta_{\text{base}} + \ln\frac{k_{\text{ref}}}{k_g}
$$

This ensures the collective softmax share of all attribute tokens in a group remains approximately constant (~60–65%) regardless of how many attributes are specified, because:

$$
k_g \cdot e^{\beta_{\text{attr}}^{(g)}} = k_g \cdot e^{\beta_{\text{base}}} \cdot \frac{k_{\text{ref}}}{k_g} = k_{\text{ref}} \cdot e^{\beta_{\text{base}}} = \text{const}
$$

### Compatibility

The algorithm operates exclusively on BPE token positions and pre-softmax logits.  It is architecture-agnostic with respect to the text encoder: SDXL's dual-encoder setup (CLIP-L + CLIP-G with a shared tokeniser) is fully supported because both streams share the same token-position indexing, and their embeddings are concatenated along the feature axis—not the token axis.