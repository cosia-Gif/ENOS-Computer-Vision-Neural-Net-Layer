"""
Test suite for the ENOS layer.

Each test probes one falsifiable claim. Tests are grouped by claim:

1. Shape / dtype contract.
2. Semantic correctness of the *vectorized cumsum* implementation, verified
   against a brute-force oracle that mirrors the implementation's actual
   semantics (cumulative directional count of activated pixels, exclusive
   of self — see `enos_implementation_oracle`).
3. Documented divergence between the implementation's *actual* semantics
   and the often-stated *intent* of "count of consecutive activated pixels
   in each direction" (see `enos_consecutive_intent_oracle`). The two agree
   only when activations form gap-free runs from the boundary.
4. Edge cases (all zeros, all ones, single pixel, line shapes, threshold sweeps).
5. Self-exclusion (a pixel never counts itself).
6. Channel independence.
7. Translation equivariance (with care: under cumulative semantics, only
   the values *inside* the support of an isolated active region match
   exactly under translation — see test).
8. Backward pass (zero gradient guarantee).
9. 3D vs 4D input handling.
10. Integration: ENOS slotted into a Conv->ReLU->Pool->ENOS->Conv pipeline.

Run from repo root:
    python3 -m unittest tests.test_enos_layer -v
or:
    python3 tests/test_enos_layer.py
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ENOSNet import (  # noqa: E402
    Conv2DLayer,
    ENOSLayer,
    MaxPoolLayer,
    ReLU,
)


# ---------------------------------------------------------------------------
# Two oracles, deliberately written as obvious nested loops.
#
# `enos_implementation_oracle` mirrors what `cumsum(mask, axis) - mask` actually
# computes: the *total number of activated pixels* strictly above / below /
# left / right of (i, j) in the same column or row.
#
# `enos_consecutive_intent_oracle` mirrors the often-stated *intent*: the
# length of the unbroken run of activated pixels extending from (i, j) in
# each cardinal direction (excluding (i, j) itself).
#
# These two diverge whenever the mask has any gap. Both are tested to make
# the discrepancy explicit and quantifiable.
# ---------------------------------------------------------------------------
def enos_implementation_oracle(x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Reference matching what `cumsum(mask) - mask` computes."""
    if x.ndim == 3:
        x = x[None, ...]
    B, H, W, C = x.shape
    mask = (x > threshold).astype(np.int32)
    north = np.zeros_like(mask)
    south = np.zeros_like(mask)
    west = np.zeros_like(mask)
    east = np.zeros_like(mask)
    for b in range(B):
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    north[b, i, j, c] = mask[b, :i, j, c].sum()
                    south[b, i, j, c] = mask[b, i + 1 :, j, c].sum()
                    west[b, i, j, c] = mask[b, i, :j, c].sum()
                    east[b, i, j, c] = mask[b, i, j + 1 :, c].sum()
    return np.concatenate([north, south, west, east], axis=-1).astype(np.float32)


def enos_consecutive_intent_oracle(x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Reference matching the stated *intent*: consecutive-run length."""
    if x.ndim == 3:
        x = x[None, ...]
    B, H, W, C = x.shape
    mask = (x > threshold).astype(np.int32)
    north = np.zeros_like(mask)
    south = np.zeros_like(mask)
    west = np.zeros_like(mask)
    east = np.zeros_like(mask)
    for b in range(B):
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    n = 0
                    for k in range(i - 1, -1, -1):
                        if mask[b, k, j, c]:
                            n += 1
                        else:
                            break
                    s = 0
                    for k in range(i + 1, H):
                        if mask[b, k, j, c]:
                            s += 1
                        else:
                            break
                    w = 0
                    for k in range(j - 1, -1, -1):
                        if mask[b, i, k, c]:
                            w += 1
                        else:
                            break
                    e = 0
                    for k in range(j + 1, W):
                        if mask[b, i, k, c]:
                            e += 1
                        else:
                            break
                    north[b, i, j, c] = n
                    south[b, i, j, c] = s
                    west[b, i, j, c] = w
                    east[b, i, j, c] = e
    return np.concatenate([north, south, west, east], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Shape / dtype contract
# ---------------------------------------------------------------------------
class TestENOSShapeAndDtype(unittest.TestCase):
    def test_output_shape_4d(self):
        layer = ENOSLayer()
        x = np.random.rand(2, 8, 8, 3).astype(np.float32)
        y = layer.forward(x)
        self.assertEqual(y.shape, (2, 8, 8, 12))

    def test_output_shape_3d(self):
        layer = ENOSLayer()
        x = np.random.rand(8, 8, 3).astype(np.float32)
        y = layer.forward(x)
        self.assertEqual(y.shape, (8, 8, 12))

    def test_output_dtype_is_float32(self):
        layer = ENOSLayer()
        x = np.random.rand(1, 4, 4, 1).astype(np.float64)
        y = layer.forward(x)
        self.assertEqual(y.dtype, np.float32)

    def test_channel_expansion_factor_is_four(self):
        layer = ENOSLayer()
        for C in (1, 3, 8, 16, 32):
            x = np.random.rand(1, 5, 5, C).astype(np.float32)
            y = layer.forward(x)
            self.assertEqual(y.shape[-1], 4 * C)


# ---------------------------------------------------------------------------
# 2. Implementation correctness vs the cumulative-semantics oracle
# ---------------------------------------------------------------------------
class TestENOSImplementationSemantics(unittest.TestCase):
    """The vectorized cumsum implementation must agree with an obviously-
    correct nested-loop implementation of the same cumulative semantics."""

    def test_matches_implementation_oracle_random(self):
        rng = np.random.default_rng(0)
        layer = ENOSLayer()
        for shape in [(1, 5, 5, 1), (2, 7, 9, 3), (3, 16, 16, 4)]:
            x = rng.random(shape).astype(np.float32)
            np.testing.assert_array_equal(
                layer.forward(x), enos_implementation_oracle(x),
                err_msg=f"vectorized impl disagrees with oracle on shape {shape}",
            )

    def test_matches_implementation_oracle_with_varied_thresholds(self):
        rng = np.random.default_rng(1)
        x = rng.random((2, 8, 8, 2)).astype(np.float32)
        for t in (0.1, 0.3, 0.5, 0.7, 0.9):
            layer = ENOSLayer(threshold=t)
            np.testing.assert_array_equal(
                layer.forward(x), enos_implementation_oracle(x, t)
            )


# ---------------------------------------------------------------------------
# 3. Documented divergence between intent and implementation
# ---------------------------------------------------------------------------
class TestENOSIntentVsImplementation(unittest.TestCase):
    """The implementation computes *cumulative* (total) counts, NOT
    *consecutive run length*. This test makes the discrepancy explicit."""

    def test_isolated_pixel_pair_diverges_from_consecutive_intent(self):
        """Two activated pixels separated by a gap reveal the difference."""
        x = np.zeros((1, 5, 1, 1), dtype=np.float32)
        x[0, 0, 0, 0] = 1.0  # row 0
        x[0, 4, 0, 0] = 1.0  # row 4
        impl = ENOSLayer().forward(x)            # cumulative semantics
        intent = enos_consecutive_intent_oracle(x)  # consecutive-run semantics

        # At (i=4, j=0): north under cumulative = #1s in rows 0..3 = 1.
        # Under consecutive intent: row 3 is 0 -> run length stops at 0.
        self.assertEqual(impl[0, 4, 0, 0], 1.0)
        self.assertEqual(intent[0, 4, 0, 0], 0.0)

    def test_intent_and_impl_agree_on_gap_free_patterns(self):
        """When activations form an unbroken run from the boundary,
        the two semantics coincide."""
        x = np.ones((1, 6, 6, 1), dtype=np.float32)  # fully active -> no gaps
        np.testing.assert_array_equal(
            ENOSLayer().forward(x),
            enos_consecutive_intent_oracle(x),
        )

    def test_quantify_divergence_on_random_input(self):
        """Quantify the fraction of positions where the two semantics
        disagree on a random sparse mask. This number is reported in
        Findings/ENOS_Layer_Documentation.md."""
        rng = np.random.default_rng(99)
        x = (rng.random((1, 16, 16, 1)) > 0.5).astype(np.float32)
        impl = ENOSLayer().forward(x)
        intent = enos_consecutive_intent_oracle(x)
        disagreement_fraction = float(np.mean(impl != intent))
        # We don't assert a specific value; we expect substantial divergence.
        self.assertGreater(disagreement_fraction, 0.10)


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------
class TestENOSEdgeCases(unittest.TestCase):
    def test_all_zeros_yields_all_zeros(self):
        x = np.zeros((1, 8, 8, 2), dtype=np.float32)
        y = ENOSLayer().forward(x)
        self.assertTrue(np.all(y == 0))

    def test_all_ones_counts_distance_to_edge(self):
        """Both cumulative and consecutive semantics agree here:
        north = i, south = H-1-i, west = j, east = W-1-j."""
        H = W = 6
        x = np.ones((1, H, W, 1), dtype=np.float32)
        y = ENOSLayer().forward(x)
        north, south, west, east = np.split(y[0, :, :, :], 4, axis=-1)
        for i in range(H):
            for j in range(W):
                self.assertEqual(north[i, j, 0], i)
                self.assertEqual(south[i, j, 0], H - 1 - i)
                self.assertEqual(west[i, j, 0], j)
                self.assertEqual(east[i, j, 0], W - 1 - j)

    def test_single_activated_pixel_under_cumulative_semantics(self):
        """An isolated active pixel at the center of a zero plane: the
        implementation places `1` at the corresponding edge-direction
        positions (not zeros, as the consecutive-intent reading would)."""
        x = np.zeros((1, 5, 5, 1), dtype=np.float32)
        x[0, 2, 2, 0] = 1.0
        y = ENOSLayer().forward(x)
        north, south, west, east = np.split(y[0, :, :, :], 4, axis=-1)
        # Below the active pixel (rows 3,4) at column 2: one activated pixel above.
        self.assertEqual(north[3, 2, 0], 1)
        self.assertEqual(north[4, 2, 0], 1)
        # Above the active pixel (rows 0,1) at column 2: one activated pixel below.
        self.assertEqual(south[0, 2, 0], 1)
        self.assertEqual(south[1, 2, 0], 1)
        # Right of (2,2): one activated pixel to the west.
        self.assertEqual(west[2, 3, 0], 1)
        self.assertEqual(west[2, 4, 0], 1)
        # The pixel itself contributes 0 in every direction.
        self.assertEqual(north[2, 2, 0], 0)
        self.assertEqual(south[2, 2, 0], 0)
        self.assertEqual(west[2, 2, 0], 0)
        self.assertEqual(east[2, 2, 0], 0)

    def test_vertical_line_under_cumulative_semantics(self):
        """Active column j=3. In every row i, west[i, j>3] sees 1 active
        pixel; east[i, j<3] sees 1 active pixel. north/south on column 3
        equal the distance to the top/bottom boundary."""
        H = W = 7
        x = np.zeros((1, H, W, 1), dtype=np.float32)
        x[0, :, 3, 0] = 1.0
        y = ENOSLayer().forward(x)
        north, south, west, east = np.split(y[0, :, :, :], 4, axis=-1)
        # On the active column itself:
        for i in range(H):
            self.assertEqual(north[i, 3, 0], i)
            self.assertEqual(south[i, 3, 0], H - 1 - i)
        # Off-column N/S must be 0 (no activations in those columns).
        for j in (0, 1, 2, 4, 5, 6):
            for i in range(H):
                self.assertEqual(north[i, j, 0], 0)
                self.assertEqual(south[i, j, 0], 0)
        # West counts: one activation to the west iff j > 3.
        for i in range(H):
            for j in range(W):
                expected = 1 if j > 3 else 0
                self.assertEqual(west[i, j, 0], expected)
                expected_e = 1 if j < 3 else 0
                self.assertEqual(east[i, j, 0], expected_e)

    def test_horizontal_line_under_cumulative_semantics(self):
        H = W = 7
        x = np.zeros((1, H, W, 1), dtype=np.float32)
        x[0, 3, :, 0] = 1.0
        y = ENOSLayer().forward(x)
        north, south, west, east = np.split(y[0, :, :, :], 4, axis=-1)
        for j in range(W):
            self.assertEqual(west[3, j, 0], j)
            self.assertEqual(east[3, j, 0], W - 1 - j)
        for i in range(H):
            for j in range(W):
                expected_n = 1 if i > 3 else 0
                expected_s = 1 if i < 3 else 0
                self.assertEqual(north[i, j, 0], expected_n)
                self.assertEqual(south[i, j, 0], expected_s)

    def test_threshold_changes_mask_and_output(self):
        x = np.broadcast_to(
            np.array([[[[0.2], [0.4], [0.6], [0.8]]]], dtype=np.float32),
            (1, 4, 4, 1),
        ).copy()
        y_low = ENOSLayer(threshold=0.1).forward(x)
        y_mid = ENOSLayer(threshold=0.5).forward(x)
        y_high = ENOSLayer(threshold=0.9).forward(x)
        self.assertFalse(np.array_equal(y_low, y_mid))
        self.assertFalse(np.array_equal(y_mid, y_high))
        self.assertTrue(np.all(y_high == 0))


# ---------------------------------------------------------------------------
# 5. Self-exclusion
# ---------------------------------------------------------------------------
class TestENOSSelfExclusion(unittest.TestCase):
    def test_center_pixel_excludes_itself(self):
        x = np.ones((1, 5, 5, 1), dtype=np.float32)
        y = ENOSLayer().forward(x)
        n, s, w, e = y[0, 2, 2, 0], y[0, 2, 2, 1], y[0, 2, 2, 2], y[0, 2, 2, 3]
        self.assertEqual((n, s, w, e), (2, 2, 2, 2))


# ---------------------------------------------------------------------------
# 6. Channel independence
# ---------------------------------------------------------------------------
class TestENOSChannelIndependence(unittest.TestCase):
    def test_channels_processed_independently(self):
        rng = np.random.default_rng(2)
        x_a = rng.random((1, 6, 6, 1)).astype(np.float32)
        x_b = rng.random((1, 6, 6, 1)).astype(np.float32)
        x_combined = np.concatenate([x_a, x_b], axis=-1)
        y_combined = ENOSLayer().forward(x_combined)
        y_a = ENOSLayer().forward(x_a)
        y_b = ENOSLayer().forward(x_b)

        # Combined output has 2 channels of N, 2 of S, 2 of W, 2 of E:
        n_c = y_combined[..., 0:2]
        s_c = y_combined[..., 2:4]
        w_c = y_combined[..., 4:6]
        e_c = y_combined[..., 6:8]
        n_a, s_a, w_a, e_a = np.split(y_a, 4, axis=-1)
        n_b, s_b, w_b, e_b = np.split(y_b, 4, axis=-1)
        np.testing.assert_array_equal(n_c[..., :1], n_a)
        np.testing.assert_array_equal(n_c[..., 1:], n_b)
        np.testing.assert_array_equal(s_c[..., :1], s_a)
        np.testing.assert_array_equal(s_c[..., 1:], s_b)
        np.testing.assert_array_equal(w_c[..., :1], w_a)
        np.testing.assert_array_equal(w_c[..., 1:], w_b)
        np.testing.assert_array_equal(e_c[..., :1], e_a)
        np.testing.assert_array_equal(e_c[..., 1:], e_b)


# ---------------------------------------------------------------------------
# 7. Translation equivariance (with caveat under cumulative semantics)
# ---------------------------------------------------------------------------
class TestENOSTranslationEquivariance(unittest.TestCase):
    def test_cumulative_counts_inside_active_region_are_translation_invariant(self):
        """Inside the support of an isolated active block, the local N/S/W/E
        cumulative counts depend only on the position relative to the block
        (not on the block's location), because everything outside the block
        is zero. This is the strongest equivariance statement that holds for
        cumulative semantics."""
        H = W = 12
        x = np.zeros((1, H, W, 1), dtype=np.float32)
        x[0, 1:4, 2:5, 0] = 1.0  # 3x3 block

        x_shift = np.zeros_like(x)
        x_shift[0, 5:8, 7:10, 0] = 1.0  # shifted by (4, 5)

        y = ENOSLayer().forward(x)
        y_shift = ENOSLayer().forward(x_shift)
        np.testing.assert_array_equal(
            y[0, 1:4, 2:5, :], y_shift[0, 5:8, 7:10, :]
        )


# ---------------------------------------------------------------------------
# 8. Backward pass
# ---------------------------------------------------------------------------
class TestENOSBackward(unittest.TestCase):
    def test_backward_returns_zero_gradient(self):
        layer = ENOSLayer()
        x = np.random.rand(2, 5, 5, 3).astype(np.float32)
        _ = layer.forward(x)
        upstream = np.random.rand(2, 5, 5, 12).astype(np.float32)
        grad = layer.backward(upstream)
        self.assertEqual(grad.shape, x.shape)
        self.assertTrue(np.all(grad == 0))

    def test_backward_blocks_upstream_gradient_flow(self):
        """Anything upstream of an ENOS layer cannot be trained by gradients
        passing through ENOS. We make this concrete by sending a large
        upstream gradient and showing zero comes out."""
        layer = ENOSLayer()
        x = np.ones((1, 4, 4, 1), dtype=np.float32)
        layer.forward(x)
        grad = layer.backward(1e6 * np.ones((1, 4, 4, 4), dtype=np.float32))
        self.assertTrue(np.array_equal(grad, np.zeros_like(x)))


# ---------------------------------------------------------------------------
# 9. 3D-input convenience wrapping (single image vs batch)
# ---------------------------------------------------------------------------
class TestENOS3DInputHandling(unittest.TestCase):
    def test_single_image_round_trip(self):
        rng = np.random.default_rng(3)
        x3 = rng.random((6, 6, 2)).astype(np.float32)
        x4 = x3[None, ...]
        layer3 = ENOSLayer()
        layer4 = ENOSLayer()
        y3 = layer3.forward(x3)
        y4 = layer4.forward(x4)
        self.assertEqual(y3.shape, (6, 6, 8))
        np.testing.assert_array_equal(y3, y4[0])


# ---------------------------------------------------------------------------
# 10. End-to-end integration with the rest of the library
# ---------------------------------------------------------------------------
class TestENOSPipelineIntegration(unittest.TestCase):
    def test_conv_relu_pool_enos_conv_chain(self):
        rng = np.random.default_rng(7)
        x = rng.random((2, 16, 16, 1)).astype(np.float32)
        c1 = Conv2DLayer(num_filters=4, kernel_size=3, input_shape=(16, 16, 1), padding=1)
        relu = ReLU()
        pool = MaxPoolLayer(pool_size=2, stride=2)
        enos = ENOSLayer(threshold=0.0)  # threshold>0 of ReLU output = "any positive activation"
        c2 = Conv2DLayer(num_filters=8, kernel_size=3, input_shape=(8, 8, 16), padding=1)

        y = c1.forward(x)
        y = relu.forward(y)
        y = pool.forward(y)
        self.assertEqual(y.shape, (2, 8, 8, 4))
        y = enos.forward(y)
        self.assertEqual(y.shape, (2, 8, 8, 16))
        y = c2.forward(y)
        self.assertEqual(y.shape, (2, 8, 8, 8))


def _demo_numeric_example():
    print("\n--- Numeric demo: 5x5 'plus sign' input (cumulative semantics) ---")
    x = np.zeros((1, 5, 5, 1), dtype=np.float32)
    x[0, 2, :, 0] = 1.0
    x[0, 1:4, 2, 0] = 1.0
    print("input mask:")
    print(x[0, :, :, 0].astype(int))
    y = ENOSLayer().forward(x)
    print("north:")
    print(y[0, :, :, 0].astype(int))
    print("south:")
    print(y[0, :, :, 1].astype(int))
    print("west:")
    print(y[0, :, :, 2].astype(int))
    print("east:")
    print(y[0, :, :, 3].astype(int))


if __name__ == "__main__":
    _demo_numeric_example()
    unittest.main(verbosity=2)
