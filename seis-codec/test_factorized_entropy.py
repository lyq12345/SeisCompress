import unittest

import numpy as np
import torch

from factorized_entropy import (
    FactorizedCategoricalEntropyModel,
    probabilities_to_quantized_cdf,
)
from first_order_entropy import FirstOrderCategoricalEntropyModel
from quantize_stable import ResidualVectorQuantizeStable
from rans_codec import FactorizedRansCodec, FirstOrderRansCodec, STREAM_HEADER


class FactorizedEntropyModelTest(unittest.TestCase):
    def test_quantized_cdf_is_positive_and_exact(self):
        probabilities = np.array(
            [
                [0.7, 0.2, 0.09, 0.01],
                [0.25, 0.25, 0.25, 0.25],
            ],
            dtype=np.float64,
        )
        cdf = probabilities_to_quantized_cdf(probabilities, precision=12)
        self.assertEqual(cdf.shape, (2, 5))
        np.testing.assert_array_equal(cdf[:, 0], 0)
        np.testing.assert_array_equal(cdf[:, -1], 1 << 12)
        self.assertTrue(np.all(np.diff(cdf.astype(np.int64), axis=1) > 0))

    def test_uniform_model_matches_fixed_width(self):
        model = FactorizedCategoricalEntropyModel(3, 8)
        codes = torch.randint(0, 8, (4, 3, 11))
        expected = torch.full((4,), 3 * 11 * 3.0)
        torch.testing.assert_close(model.estimate_bits(codes), expected)

    def test_calibrate_from_counts_sets_smoothed_mle(self):
        model = FactorizedCategoricalEntropyModel(2, 3)
        counts = torch.tensor([[7, 1, 1], [0, 2, 4]])
        probabilities = model.calibrate_from_counts(counts, smoothing=1.0)
        expected = torch.tensor(
            [[8 / 12, 2 / 12, 2 / 12], [1 / 9, 3 / 9, 5 / 9]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(probabilities.cpu(), expected)
        torch.testing.assert_close(model.probabilities().double(), expected)

    def test_calibration_rejects_invalid_counts(self):
        model = FactorizedCategoricalEntropyModel(2, 3)
        with self.assertRaises(ValueError):
            model.calibrate_from_counts(torch.ones(2, 2))
        with self.assertRaises(ValueError):
            model.calibrate_from_counts(torch.tensor([[1, -1, 2], [1, 1, 1]]))
        with self.assertRaises(ValueError):
            model.calibrate_from_counts(torch.ones(2, 3), smoothing=0.0)

    def test_rate_surrogate_reaches_prior_encoder_and_codebook(self):
        torch.manual_seed(7)
        quantizer = ResidualVectorQuantizeStable(
            input_dim=4,
            n_codebooks=2,
            codebook_size=8,
            codebook_dim=2,
        )
        entropy_model = FactorizedCategoricalEntropyModel(2, 8)
        with torch.no_grad():
            entropy_model.logits.copy_(torch.randn_like(entropy_model.logits))

        latent = torch.randn(3, 4, 5, requires_grad=True)
        _, codes, _, _, _, rate_bits = quantizer(
            latent,
            entropy_log_probs=entropy_model.log_probabilities(),
            entropy_temperature=0.2,
        )
        torch.testing.assert_close(
            rate_bits.detach(),
            entropy_model.estimate_bits(codes).detach(),
        )
        rate_bits.mean().backward()
        self.assertGreater(float(latent.grad.abs().sum()), 0.0)
        self.assertGreater(float(entropy_model.logits.grad.abs().sum()), 0.0)
        codebook_grad = quantizer.quantizers[0].codebook.weight.grad
        self.assertIsNotNone(codebook_grad)
        self.assertGreater(float(codebook_grad.abs().sum()), 0.0)


class FactorizedRansCodecTest(unittest.TestCase):
    def test_random_stream_round_trip(self):
        rng = np.random.default_rng(19)
        probabilities = rng.random((9, 1024))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        cdf = probabilities_to_quantized_cdf(probabilities, precision=16)
        codes = np.stack(
            [rng.choice(1024, size=376, p=probabilities[idx]) for idx in range(9)]
        )
        codec = FactorizedRansCodec(cdf, precision=16)
        stream = codec.encode(codes, original_length=3001)
        decoded, original_length = codec.decode(stream)
        self.assertGreater(len(stream), STREAM_HEADER.size)
        self.assertEqual(original_length, 3001)
        np.testing.assert_array_equal(decoded, codes)

    def test_wrong_cdf_is_rejected(self):
        probabilities = np.array([[0.6, 0.3, 0.1]], dtype=np.float64)
        codec = FactorizedRansCodec(
            probabilities_to_quantized_cdf(probabilities, precision=12),
            precision=12,
        )
        stream = codec.encode(np.array([[0, 1, 0, 2]], dtype=np.int64), original_length=4)
        other = FactorizedRansCodec(
            probabilities_to_quantized_cdf(
                np.array([[0.2, 0.3, 0.5]], dtype=np.float64),
                precision=12,
            ),
            precision=12,
        )
        with self.assertRaises(ValueError):
            other.decode(stream)

    def test_truncated_stream_is_rejected(self):
        model = FactorizedCategoricalEntropyModel(1, 16, cdf_precision=12)
        codec = FactorizedRansCodec(model.quantized_cdf(), precision=12)
        stream = codec.encode(np.arange(16, dtype=np.int64)[None, :], original_length=16)
        with self.assertRaises(ValueError):
            codec.decode(stream[:-1])


class FirstOrderEntropyModelTest(unittest.TestCase):
    def _calibrated_model(self):
        model = FirstOrderCategoricalEntropyModel(2, 4, cdf_precision=12)
        marginal_counts = torch.tensor(
            [[100, 80, 60, 40], [50, 50, 50, 50]],
            dtype=torch.int64,
        )
        transitions = torch.ones(2, 4, 4, dtype=torch.int64)
        for codebook in range(2):
            for symbol in range(4):
                transitions[codebook, symbol, symbol] += 100
        model.calibrate_from_counts(
            marginal_counts,
            transitions,
            backoff_concentration=4.0,
        )
        return model

    def test_calibrated_cdfs_are_valid(self):
        model = self._calibrated_model()
        marginal, conditional = model.quantized_cdfs()
        self.assertEqual(marginal.shape, (2, 5))
        self.assertEqual(conditional.shape, (2, 4, 5))
        np.testing.assert_array_equal(marginal[:, -1], 1 << 12)
        np.testing.assert_array_equal(conditional[:, :, -1], 1 << 12)
        self.assertTrue(np.all(np.diff(conditional.astype(np.int64), axis=-1) > 0))

    def test_repeated_sequence_costs_less_than_uniform(self):
        model = self._calibrated_model()
        codes = torch.tensor(
            [[[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]]],
            dtype=torch.long,
        )
        self.assertLess(float(model.estimate_bits(codes)[0]), 20.0)

    def test_first_order_rans_round_trip(self):
        model = self._calibrated_model()
        marginal, conditional = model.quantized_cdfs()
        codec = FirstOrderRansCodec(marginal, conditional, precision=12)
        codes = np.array(
            [[0, 0, 1, 1, 1, 3, 3], [2, 2, 2, 0, 0, 0, 1]],
            dtype=np.int64,
        )
        stream = codec.encode(codes, original_length=51)
        decoded, original_length = codec.decode(stream)
        self.assertEqual(original_length, 51)
        np.testing.assert_array_equal(decoded, codes)

    def test_first_order_wrong_cdf_is_rejected(self):
        model = self._calibrated_model()
        marginal, conditional = model.quantized_cdfs()
        codec = FirstOrderRansCodec(marginal, conditional, precision=12)
        stream = codec.encode(np.array([[0, 0, 1, 1]]), original_length=4)
        conditional = conditional.copy()
        conditional[0, 0] = marginal[0]
        other = FirstOrderRansCodec(marginal, conditional, precision=12)
        with self.assertRaises(ValueError):
            other.decode(stream)


if __name__ == "__main__":
    unittest.main()
