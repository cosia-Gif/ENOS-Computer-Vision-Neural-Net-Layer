# The ENOS Layer: Scientific Documentation

> ENOS — **N**orth-**E**ast-**O**uest-**S**ud (FR) / **N**orth-**E**ast-**W**est-**S**outh (EN).
> A non-parametric, non-differentiable spatial-feature layer that augments
> the conventional convolutional stack with directional activation-distribution
> statistics.

This document describes (a) what the layer in
[`ENOSNet.py`](../ENOSNet.py) actually computes, (b) every test case
exercised against it in [`tests/`](../tests/), with the result, (c) when
adding ENOS to a model is and is not justified, and (d) how to slot it
into a real object-detection pipeline.

The reasoning is structured to be falsifiable: every quantitative claim
in this document is regenerable by running the test files committed
alongside it.

---

## 1. Operational definition

Given a 4-D input tensor `x` of shape `(B, H, W, C)`:

1. **Threshold** to a binary mask:
   `mask = (x > threshold).astype(int32)` — default `threshold = 0.5`.
2. **Compute four directional cumulative counts** along the spatial axes:
   ```python
   north = cumsum(mask, axis=H) - mask
   south = cumsum(mask[:, ::-1, :, :], axis=H)[:, ::-1, :, :] - mask
   west  = cumsum(mask, axis=W) - mask
   east  = cumsum(mask[:, :, ::-1, :], axis=W)[:, :, ::-1, :] - mask
   ```
3. **Concatenate** along the channel axis:
   `output = concat([north, south, west, east], axis=-1)` →
   shape `(B, H, W, 4·C)`, dtype `float32`.

| Property              | Value                                                                |
| --------------------- | -------------------------------------------------------------------- |
| Trainable parameters  | 0                                                                    |
| Differentiable        | No — `backward()` returns `zeros_like(input)`                       |
| Channel expansion     | ×4                                                                   |
| Hyperparameters       | `threshold` (default 0.5)                                            |
| Asymptotic complexity | O(B · H · W · C) (four prefix sums over a 4-D tensor)                |
| Memory overhead       | ~5× the input mask size during forward (mask + four count tensors)   |

---

## 2. **Critical finding — what the layer actually computes vs. what is often claimed**

The layer is frequently described — including in informal write-ups
linked to this repository — as computing the **count of *consecutive*
activated pixels** extending from each location in each cardinal
direction. **The implementation in `ENOSNet.py` does not do this.**

The expression `cumsum(mask, axis) - mask` evaluates the **total
number of activated pixels strictly above** (resp. below / left / right
of) position `(i, j)` in the same column or row. This is the
*cumulative* count, not the *consecutive-run* count. The two are
**equal only when the activation pattern is gap-free** along the
direction of measurement (for example a fully-active plane, or a fully-
active column for the N/S directions on that column).

This is verified mechanically by two independent oracles in
[`tests/test_enos_layer.py`](../tests/test_enos_layer.py):

* `enos_implementation_oracle` — nested-loop reproduction of the
  `cumsum - mask` semantics.
* `enos_consecutive_intent_oracle` — nested-loop reproduction of the
  consecutive-run semantics.

The tests `TestENOSImplementationSemantics::*` confirm the vectorized
implementation matches the first oracle exactly on randomized inputs
(threshold sweep included). The tests in
`TestENOSIntentVsImplementation::*` confirm the implementation
**diverges** from the second oracle whenever the mask contains gaps.

### 2.1 Quantified divergence

Running the implementation and the consecutive-intent oracle on
randomized binary masks of shape `(4, 32, 32, 4)` at varying densities
gives:

| Activation density | Position-wise disagreement | Mean (consecutive) | Mean (implementation) | Implementation / consecutive |
| -----------------: | -------------------------: | -----------------: | --------------------: | ---------------------------: |
|                0.1 |                      66.1% |              0.106 |                 1.523 |                       14.4× |
|                0.3 |                      84.5% |              0.417 |                 4.657 |                       11.2× |
|                0.5 |                      87.8% |              0.944 |                 7.779 |                        8.2× |
|                0.7 |                      84.9% |              2.102 |                10.854 |                        5.2× |
|                0.9 |                      67.9% |              6.057 |                13.899 |                        2.3× |

Reproducible from the snippet in §6.

### 2.2 Why this matters

The two semantics are different *features* and they encode different
priors:

* **Consecutive-run** counts are a *connectivity* feature. They tell
  the downstream layer how far an unbroken structure (edge, stroke,
  blob boundary) extends from `(i, j)`. They are insensitive to remote
  activations that are not connected to `(i, j)` along the cardinal
  axis.
* **Cumulative** counts are a *density* feature. They tell the
  downstream layer the integrated activation in each half-row/column
  starting from `(i, j)`. Disconnected activations contribute equally
  to nearby ones.

Both are legitimate features. But a downstream convolution that was
designed assuming consecutive-run inputs (e.g. expecting small,
locally-supported numbers) will be exposed to outputs that grow
linearly with the *spatial extent of the receptive half-plane*, not
with the local structure size. On a saturating mask this causes the
ENOS output magnitude to scale as O(H) or O(W), not O(local-feature-
size).

This is the dominant finding to internalize before tuning the
threshold or planning where to insert the layer.

---

## 3. Test cases and results

All tests live in `tests/test_enos_layer.py`. Twenty-two tests across
ten test classes, all passing on the current implementation.

```
$ python3 -m unittest tests.test_enos_layer -v
...
Ran 22 tests in 0.108s
OK
```

The tests are intentionally each one-claim, one-assertion; the table
below maps every test to the property it verifies and the observed
outcome.

### 3.1 Shape and dtype contract — `TestENOSShapeAndDtype`

| Test                                      | Verified property                                                                | Result |
| ----------------------------------------- | -------------------------------------------------------------------------------- | -----: |
| `test_output_shape_4d`                    | `(B,H,W,C)` → `(B,H,W,4C)` for batched input                                     |   pass |
| `test_output_shape_3d`                    | `(H,W,C)` single image → `(H,W,4C)`                                              |   pass |
| `test_output_dtype_is_float32`            | Output dtype is `float32` even when input is `float64`                           |   pass |
| `test_channel_expansion_factor_is_four`   | Output channel count is exactly `4 × C` for `C ∈ {1,3,8,16,32}`                 |   pass |

### 3.2 Implementation correctness — `TestENOSImplementationSemantics`

| Test                                                       | Verified property                                                                                | Result |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | -----: |
| `test_matches_implementation_oracle_random`                | Vectorized impl ≡ nested-loop oracle of cumulative semantics on shapes `(1,5,5,1)`, `(2,7,9,3)`, `(3,16,16,4)` |   pass |
| `test_matches_implementation_oracle_with_varied_thresholds`| Equivalence holds across thresholds `{0.1, 0.3, 0.5, 0.7, 0.9}`                                  |   pass |

### 3.3 Intent vs. implementation — `TestENOSIntentVsImplementation`

| Test                                                       | Verified property                                                                                | Result |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | -----: |
| `test_isolated_pixel_pair_diverges_from_consecutive_intent`| On column `[1,0,0,0,1]`, impl returns N=1 at the bottom pixel; consecutive intent returns N=0 |   pass |
| `test_intent_and_impl_agree_on_gap_free_patterns`          | On a fully-active plane, the two semantics produce identical output                              |   pass |
| `test_quantify_divergence_on_random_input`                 | Position-wise disagreement on a random `(1,16,16,1)` mask exceeds 10%                            |   pass |

### 3.4 Edge cases — `TestENOSEdgeCases`

| Test                                                          | Verified property                                                                                                          | Result |
| ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | -----: |
| `test_all_zeros_yields_all_zeros`                             | `x = 0` → output is identically zero                                                                                       |   pass |
| `test_all_ones_counts_distance_to_edge`                       | On a fully-active 6×6 plane: `north[i,j] = i`, `south[i,j] = H-1-i`, `west[i,j] = j`, `east[i,j] = W-1-j`                  |   pass |
| `test_single_activated_pixel_under_cumulative_semantics`      | Isolated active pixel at `(2,2)` produces value `1` at all positions on the same row/column on the appropriate side       |   pass |
| `test_vertical_line_under_cumulative_semantics`               | On an active column `j=3`: N/S grow linearly along that column; W/E are `1` exactly on every column to the right/left of 3 |   pass |
| `test_horizontal_line_under_cumulative_semantics`             | Symmetric statement for a horizontal line                                                                                  |   pass |
| `test_threshold_changes_mask_and_output`                      | Outputs at thresholds `{0.1, 0.5, 0.9}` are pairwise unequal; at threshold 0.9 the output is identically zero              |   pass |

### 3.5 Self-exclusion — `TestENOSSelfExclusion`

| Test                                  | Verified property                                                                                       | Result |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------- | -----: |
| `test_center_pixel_excludes_itself`   | Center of a 5×5 fully-active plane: `(N,S,W,E) = (2,2,2,2)` — the active pixel itself is not counted |   pass |

### 3.6 Channel independence — `TestENOSChannelIndependence`

| Test                                  | Verified property                                                                                                                          | Result |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | -----: |
| `test_channels_processed_independently` | The N block of the output for two stacked channels `[a, b]` equals the N output of `a` concatenated with the N output of `b`, and similarly for S, W, E |   pass |

### 3.7 Translation behaviour — `TestENOSTranslationEquivariance`

| Test                                                                           | Verified property                                                                                                                                                         | Result |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -----: |
| `test_cumulative_counts_inside_active_region_are_translation_invariant`        | For an isolated 3×3 active block in a 12×12 zero plane, the values *inside the support of the block* depend only on relative position, not absolute location              |   pass |

> ⚠️ Note: the layer is **not** translation-equivariant in general,
> because the cumulative count at any pixel depends on the absolute
> distance to the boundary (see §2). The test above documents the
> precisely-defined regime in which translation equivariance does hold.

### 3.8 Backward pass — `TestENOSBackward`

| Test                                            | Verified property                                                                       | Result |
| ----------------------------------------------- | --------------------------------------------------------------------------------------- | -----: |
| `test_backward_returns_zero_gradient`           | `backward(d_output)` returns `zeros_like(input)` regardless of `d_output`               |   pass |
| `test_backward_blocks_upstream_gradient_flow`   | Even an upstream gradient of magnitude 10⁶ is zeroed out — no signal leaks through ENOS |   pass |

### 3.9 3D input handling — `TestENOS3DInputHandling`

| Test                            | Verified property                                                       | Result |
| ------------------------------- | ----------------------------------------------------------------------- | -----: |
| `test_single_image_round_trip`  | Calling `forward` on `(H,W,C)` matches the batched call on `(1,H,W,C)`  |   pass |

### 3.10 Pipeline integration — `TestENOSPipelineIntegration`

| Test                                          | Verified property                                                                                       | Result |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------- | -----: |
| `test_conv_relu_pool_enos_conv_chain`         | A real chain `Conv(4)→ReLU→MaxPool(2)→ENOS→Conv(8)` produces the expected shapes throughout (smoke)    |   pass |

---

## 4. Performance characterization

Run with `python3 tests/benchmark_enos.py`. Wall times are the median
of five forward passes after one warmup, on the host's CPU.

### 4.1 Shape sweep (threshold = 0.5, uniform random input)

| Shape                | Elements    | Forward (ms) | Output max | Output mean |
| -------------------- | ----------: | -----------: | ---------: | ----------: |
| `(1, 14, 14, 8)`     |       1 568 |        0.042 |         11 |       3.20  |
| `(1, 28, 28, 16)`    |      12 544 |        0.190 |         23 |       6.74  |
| `(8, 28, 28, 16)`    |     100 352 |        2.13  |         23 |       6.79  |
| `(8, 56, 56, 32)`    |     802 816 |       31.83  |         43 |      13.74  |
| `(16, 56, 56, 32)`   |   1 605 632 |       63.13  |         43 |      13.74  |
| `(8, 112, 112, 32)`  |   3 211 264 |      146.05  |         78 |      27.74  |
| `(4, 224, 224, 16)`  |   3 211 264 |      152.08  |        146 |      55.71  |

Empirical scaling is consistent with O(B·H·W·C). Doubling batch size
at fixed `(H,W,C)` exactly doubled wall time
(`(8→16, 56, 56, 32): 31.83 → 63.13 ms`).

The **output magnitude grows with the spatial dimension** even on a
50%-density input: at `(4, 224, 224, 16)` the maximum directional
count is **146** and the mean is **55.7** — feature scales that any
downstream `Dense` or `Conv` layer must be prepared to absorb. See §5
on normalization.

### 4.2 Threshold sweep (4×56×56×16 ReLU-like input)

| Threshold | Sparsity (fraction zero) | Output max | Output mean |
| --------: | -----------------------: | ---------: | ----------: |
|     0.05  |                  0.597   |         36 |      11.083 |
|     0.10  |                  0.687   |         30 |       8.609 |
|     0.25  |                  0.888   |         17 |       3.089 |
|     0.50  |                  0.992   |          5 |       0.220 |
|     0.75  |                  1.000   |          1 |       0.005 |
|     0.90  |                  1.000   |          1 |       0.001 |

The threshold has a sharp effect on the output magnitude. For a
ReLU-rectified normal input rescaled to `[0,1]`, threshold 0.5 already
zeroes 99.2% of activations, and threshold 0.75+ effectively turns the
layer into a near-zero feature map. **The threshold must be calibrated
to the empirical activation distribution at the layer's insertion
point**, not chosen by analogy with image-pixel intensities.

### 4.3 Vectorized vs. nested-loop oracle

On `(1, 16, 16, 4)`:

| Implementation                | Time (ms) |
| ----------------------------- | --------: |
| Vectorized (`cumsum - mask`)  |     0.038 |
| Nested-loop oracle            |     4.590 |
| **Speedup**                   |  **121×** |

The cumsum implementation is the right way to compute this feature.
The asymptotic ratio grows as Θ(H+W) per pixel.

---

## 5. Strengths

1. **Zero parameters, zero training cost.** ENOS introduces no weights
   to learn and adds no terms to the loss. It is purely a
   *representational augmentation*.
2. **Captures non-local spatial statistics in one operator.** A
   single ENOS forward pass yields, at every pixel, four scalars that
   summarize the entire row and column passing through that pixel.
   Reproducing the same receptive-field reach with stacked
   convolutions would require log(max(H, W)) layers or dilated kernels
   with carefully chosen rates.
3. **Linear-time, vectorizable.** O(B·H·W·C) with small constant
   factor; ~120× faster than the obvious nested-loop reference on a
   16×16×4 tile. Trivially parallelizable per-batch and per-channel.
4. **Interpretable.** Each output channel has a precise, human-readable
   meaning: "directional activation density above / below / left /
   right of this pixel."
5. **Independent of channel count.** The four cumsum operations are
   decoupled across channels, so doubling `C` doubles work but does
   not change the algorithm.

---

## 6. Weaknesses (elaborated)

The strengths above are real. The weaknesses below are equally real
and warrant elaboration because most of them are not obvious from the
two-line description of the operation.

### 6.1 Non-differentiable: gradient flow is severed

`backward()` returns zeros. The implication is structural, not
cosmetic:

> **Any layer placed *upstream* of an ENOS layer cannot be trained by
> any loss term applied *downstream* of it.**

In the reference architecture
([`Building_NN.py`](../Building_NN.py)),
`Conv₁ → ReLU → MaxPool → ENOS → Conv₂ → … → Softmax`, the gradient of
the cross-entropy loss with respect to the parameters of `Conv₁`
is identically zero. `Conv₁` remains pinned to its Glorot-uniform
initialization for the entire training run. This is verified in
`test_backward_blocks_upstream_gradient_flow`.

This has three consequences:

* **Don't put ENOS early in the network.** Anything before it is
  effectively a fixed random feature extractor.
* **The first layer after ENOS does the heavy lifting.** All
  representational learning that would normally have happened upstream
  is concentrated into the immediately-downstream learnable layer.
  This pushes a lot of capacity demand onto a single Conv.
* **Skip connections are mandatory if you want to train upstream
  weights.** A residual `x + ENOS(x)` (with appropriate channel
  matching) lets the gradient bypass ENOS through the identity branch.

### 6.2 Hard threshold: gradient discontinuity *and* hyperparameter sensitivity

The thresholding step `(x > threshold)` is a hard step function. Two
problems compound:

* It is non-differentiable and step-shaped, so even a soft surrogate
  (e.g. straight-through estimator) would still produce zero gradient
  almost everywhere.
* The choice of threshold is critical and is not learnable. From §4.2,
  varying the threshold from 0.05 to 0.5 on the same input takes the
  output mean from 11.08 to 0.22 — two orders of magnitude. A
  threshold mis-calibrated by 0.2 can effectively kill the layer.

The right way to use the threshold is to *measure* the activation
distribution at the insertion point on a held-out batch and pick a
threshold near the median (or some target sparsity). This is brittle
to distribution shift across training.

### 6.3 Cumulative-vs-consecutive semantic mismatch (§2)

The implementation computes cumulative directional sums, not
consecutive runs. On naturally sparse feature maps the two diverge on
60–88% of positions and differ in mean magnitude by 2× to 14×. If you
were expecting "stroke length" semantics, you are getting "half-row
density" semantics. Choose your threshold and your downstream
interpretation accordingly.

### 6.4 Output magnitude grows with the spatial dimension

Cumulative counts are bounded above by `H` (for N/S) and `W` (for
W/E). On a 224×224 feature map at 50% density, the maximum N count
observed in §4.1 was 146. Without normalization, this dominates any
nearby learnable Conv weights initialized with Glorot uniform (whose
output scale is ~1).

**Mandatory mitigation:** divide the ENOS output by the corresponding
spatial dimension (or apply a `BatchNorm` / `LayerNorm` immediately
after). The reference architecture in this repo does *not* do this,
which contributes to the slow training reported in
`Animations/Model_1000_images.png`.

### 6.5 4× channel inflation cascades downstream cost

ENOS quadruples the channel count. The Conv layer immediately
downstream therefore has 4× more input channels than it would without
ENOS. For a `Conv(out=K, k=3)` receiving the output of an ENOS layer
on `C` input channels, the parameter count is `K · 4C · 9` instead of
`K · C · 9`. In the reference network, the second Conv takes input
`(14, 14, 32)` instead of `(14, 14, 8)` — 4× the FLOPs and 4× the
weights for that layer.

### 6.6 Not boundary-aware

The cumsum is taken over the entire row/column with no notion of
"object" or "segment." A long active stripe in another part of the
image affects the count at every pixel of the same row, even if those
pixels are perceptually unrelated. The layer cannot, by construction,
distinguish "two separate objects on the same row" from "one wide
object."

### 6.7 Ill-posed on non-spatial channel layouts

The layer assumes that `axis=1` is height and `axis=2` is width. It
should not be applied to any tensor whose spatial axes have been
flattened, or to any tensor whose channel-last layout has been broken
by an upstream transpose. The class does not validate this; it will
silently produce garbage. The test suite covers the contracted layout
only.

### 6.8 No batched threshold; one threshold per layer

The implementation supports a single scalar threshold for the entire
batch and all channels. Per-channel thresholding (which would be the
correct way to handle heterogeneous activation distributions across
filters) requires modifying the layer.

### 6.9 Memory: five tensors of mask shape are alive at once

`mask`, `north_counts`, `south_counts`, `west_counts`, `east_counts`
all coexist before the final `concatenate`. For a `(16, 56, 56, 32)`
input that's about 8 MB of int32 per tensor, ~40 MB transient before
the concat returns ~32 MB of float32 output. Fine on CPU; on GPU, the
extra activation memory matters during deep stacks.

---

## 7. Implementation recipe for an object-detection pipeline

The layer is best inserted **after a learnable feature extractor that
produces sparse, well-thresholded activations**, and **before a
learnable head**, with a normalization step in between to control
magnitude. Concretely, drop it into a YOLO-/SSD-style architecture as
follows:

```python
# Pseudocode in this library's idiom (see ENOSNet.py).
backbone = NeuralNet()
backbone.add(Conv2DLayer(num_filters=32, kernel_size=3, input_shape=(H, W, 3),  padding=1))
backbone.add(ReLU())
backbone.add(MaxPoolLayer(2, 2))
backbone.add(Conv2DLayer(num_filters=64, kernel_size=3, input_shape=(H//2, W//2, 32), padding=1))
backbone.add(ReLU())
backbone.add(MaxPoolLayer(2, 2))
# >>> Insertion point. Activations at this depth are sparse positive ReLU outputs.
backbone.add(ENOSLayer(threshold=t_star))   # see §7.1 for choosing t_star
# (Add a normalization layer here in any framework that has one.)
# Channels just went from 64 → 256. Compensate in the next conv:
backbone.add(Conv2DLayer(num_filters=128, kernel_size=3, input_shape=(H//4, W//4, 256), padding=1))
backbone.add(ReLU())
# ... detection head (objectness, bbox regression, class logits) ...
```

Five concrete recommendations specific to detection:

1. **Insert ENOS at a mid-depth feature scale, not at input.** Raw
   image pixels rarely have a meaningful threshold and there are no
   weights upstream to be wasted by the gradient block.
2. **Use a threshold near the median of the post-ReLU activation
   distribution** at the insertion point, measured on a calibration
   batch. From §4.2: a threshold giving ~50% sparsity yields
   discriminative outputs; >90% sparsity collapses the signal.
3. **Normalize.** Divide ENOS output by `max(H, W)` or follow with
   BatchNorm/LayerNorm. Without this the values dominate downstream
   Conv outputs and the model becomes hard to train (§6.4).
4. **Use ENOS as an auxiliary branch, not the trunk.** Concatenate
   `[features, ENOS(features)]` rather than replacing `features` with
   `ENOS(features)`. This preserves the gradient path through the
   trunk and exposes the directional counts to the head as additional
   evidence.
5. **For bounding-box regression specifically**, the cumulative
   directional counts are a useful prior for *width* and *height*
   estimation. At a pixel inside an object, `west[i,j] + east[i,j]`
   gives a lower bound on the object's horizontal extent in
   activation space; `north[i,j] + south[i,j]` likewise for vertical
   extent. This signal is most useful for elongated, axis-aligned
   classes (vehicles, doors, vertical signage). It is *less* useful
   for diagonal or curved structures, where the directional projection
   onto cardinal axes loses information.

### 7.1 Choosing the threshold

A reproducible recipe:

```python
# 1. Run a forward pass with the threshold disabled (e.g. very small).
calib_input = ...                       # one batch
features = backbone_up_to_enos(calib_input)
# 2. Pick the threshold to hit a target sparsity (e.g. 50%).
t_star = float(np.quantile(features, 0.5))
enos = ENOSLayer(threshold=t_star)
# 3. Optionally re-measure once training has shifted the activation distribution
#    and re-set the threshold. ENOS has no parameters so this is free.
```

### 7.2 Using ENOS at multiple scales (FPN-style)

Because ENOS is parameter-free, it costs nothing to attach an ENOS
branch at every level of a feature pyramid. The directional counts at
coarse scales encode large-scale layout (e.g. "this pixel is below a
long horizontal structure") while at fine scales they encode local
extent. A simple multi-scale head can concatenate
`[fpn_level_l, ENOS(fpn_level_l)]` at each level before classification
and regression heads.

---

## 8. Reproducing every number in this document

```bash
git clone https://github.com/cosia-Gif/ENOS-Computer-Vision-Neural-Net-Layer.git
cd ENOS-Computer-Vision-Neural-Net-Layer
pip install numpy tqdm

# Functional tests (22 cases, ~0.1 s):
python3 -m unittest tests.test_enos_layer -v

# Performance / threshold / impl-vs-oracle benchmarks:
python3 tests/benchmark_enos.py

# Quantified intent-vs-implementation divergence table from §2.1:
python3 - <<'PY'
import sys, numpy as np
sys.path.insert(0, '.')
from ENOSNet import ENOSLayer
from tests.test_enos_layer import enos_consecutive_intent_oracle
rng = np.random.default_rng(99)
for d in [0.1, 0.3, 0.5, 0.7, 0.9]:
    x = (rng.random((4, 32, 32, 4)) < d).astype(np.float32)
    impl = ENOSLayer(threshold=0.5).forward(x)
    intent = enos_consecutive_intent_oracle(x, threshold=0.5)
    print(f'd={d}  disagree={np.mean(impl!=intent):.3f}  '
          f'intent_mean={intent.mean():.3f}  impl_mean={impl.mean():.3f}')
PY
```

---

## 9. Summary

ENOS is a non-parametric, non-differentiable, channel-quadrupling
layer that encodes directional activation density. Its strengths
(zero parameters, linear-time vectorized implementation, large
effective receptive field, interpretability) are real; its weaknesses
(severed upstream gradient, hard threshold, magnitude that scales with
H/W, 4× downstream cost, and a substantive divergence between its
documented intent and its actual semantics) are also real and demand
deliberate placement, normalization, and threshold calibration.

Used as an *auxiliary* branch — concatenated with, not replacing, a
trainable feature trunk — at a mid-depth feature scale, with a
threshold chosen from a calibration batch and a normalization step
afterwards, ENOS provides a free directional-extent prior that is
useful for axis-aligned bounding-box regression and for any task where
"how far does this active region extend in each cardinal direction"
is a meaningful question.

Used as a drop-in trunk component without normalization, as in the
reference architecture of this repository, it will train slowly and
will leave its upstream layers stuck at initialization.
