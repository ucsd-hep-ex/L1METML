"""
Unit tests for loss functions
Run with cmd:
    python -m pytest loss/test_loss.py -v
"""

import unittest

import numpy as np
import tensorflow as tf
from loss import custom_loss_wrapper


class TestCustomLoss(unittest.TestCase):
    """Test cases for custom loss functions"""

    def setUp(self):
        """Set up test data before each test"""
        self.batch_size = 32
        self.normFac = 100

        # Create reproducible test data
        np.random.seed(42)
        tf.random.set_seed(42)

        # Ground truth: px, py components
        self.y_true = tf.constant(np.random.randn(self.batch_size, 2), dtype=tf.float32)

        # Predictions: px, py components
        self.y_pred = tf.constant(np.random.randn(self.batch_size, 2), dtype=tf.float32)

        # Edge case: zeros
        self.y_true_zeros = tf.zeros((self.batch_size, 2), dtype=tf.float32)
        self.y_pred_zeros = tf.zeros((self.batch_size, 2), dtype=tf.float32)

    def test_standard_loss_computation(self):
        """Test that standard loss computes without errors"""
        loss_fn = custom_loss_wrapper(normFac=self.normFac, use_symmetry=False)

        # Should not raise any exceptions
        loss_value = loss_fn(self.y_true, self.y_pred)

        # Loss should be a scalar tensor
        self.assertEqual(loss_value.shape, ())

        # Loss should be finite and positive
        self.assertTrue(tf.math.is_finite(loss_value))
        self.assertGreaterEqual(loss_value.numpy(), 0.0)

    def test_symmetry_loss_computation(self):
        """Test that symmetry loss computes without errors"""
        loss_fn = custom_loss_wrapper(
            normFac=self.normFac, use_symmetry=True, symmetry_weight=0.5
        )

        # Should not raise any exceptions
        loss_value = loss_fn(self.y_true, self.y_pred)

        # Loss should be a scalar tensor
        self.assertEqual(loss_value.shape, ())

        # Loss should be finite and positive
        self.assertTrue(tf.math.is_finite(loss_value))
        self.assertGreaterEqual(loss_value.numpy(), 0.0)

    def test_symmetry_increases_loss(self):
        """Test that symmetry penalty increases the loss"""
        standard_loss_fn = custom_loss_wrapper(normFac=self.normFac, use_symmetry=False)
        symmetry_loss_fn = custom_loss_wrapper(
            normFac=self.normFac, use_symmetry=True, symmetry_weight=1.0
        )

        standard_loss = standard_loss_fn(self.y_true, self.y_pred)
        symmetry_loss = symmetry_loss_fn(self.y_true, self.y_pred)

        # Symmetry loss should be >= standard loss (equal if perfectly symmetric)
        self.assertGreaterEqual(symmetry_loss.numpy(), standard_loss.numpy())

    def test_symmetry_weight_scaling(self):
        """Test that symmetry weight properly scales the penalty"""
        loss_fn_low = custom_loss_wrapper(
            normFac=self.normFac, use_symmetry=True, symmetry_weight=0.1
        )
        loss_fn_high = custom_loss_wrapper(
            normFac=self.normFac, use_symmetry=True, symmetry_weight=2.0
        )

        loss_low = loss_fn_low(self.y_true, self.y_pred)
        loss_high = loss_fn_high(self.y_true, self.y_pred)

        # Higher weight should give higher loss
        self.assertGreater(loss_high.numpy(), loss_low.numpy())

    def test_perfect_predictions(self):
        """Test loss when predictions match truth exactly"""
        loss_fn = custom_loss_wrapper(normFac=self.normFac, use_symmetry=False)

        # Perfect predictions should give low loss
        loss_value = loss_fn(self.y_true, self.y_true)

        # Should be very small (close to zero, accounting for numerical precision)
        self.assertLess(loss_value.numpy(), 1e-3)

    def test_normfac_scaling(self):
        """Test that normFac properly scales the loss"""
        loss_fn_norm1 = custom_loss_wrapper(normFac=1, use_symmetry=False)
        loss_fn_norm100 = custom_loss_wrapper(normFac=100, use_symmetry=False)

        loss_norm1 = loss_fn_norm1(self.y_true, self.y_pred)
        loss_norm100 = loss_fn_norm100(self.y_true, self.y_pred)

        # Higher normFac should scale the MSE component
        # (The relationship isn't exactly linear due to the dev term)
        self.assertNotEqual(loss_norm1.numpy(), loss_norm100.numpy())

    def test_gradient_computation(self):
        """Test that gradients can be computed (for training)"""
        loss_fn = custom_loss_wrapper(normFac=self.normFac, use_symmetry=True)

        with tf.GradientTape() as tape:
            # Make predictions trainable
            y_pred_var = tf.Variable(self.y_pred)
            tape.watch(y_pred_var)
            loss_value = loss_fn(self.y_true, y_pred_var)

        # Should be able to compute gradients
        gradients = tape.gradient(loss_value, y_pred_var)

        self.assertIsNotNone(gradients)
        self.assertEqual(gradients.shape, y_pred_var.shape)
        self.assertTrue(tf.reduce_any(tf.math.is_finite(gradients)))


class TestLossIntegration(unittest.TestCase):
    """Integration tests for loss functions with actual model training"""

    def test_with_simple_model(self):
        """Test loss function in a simple model training setup"""
        # Create a simple model
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(16, activation="relu", input_shape=(4,)),
                tf.keras.layers.Dense(2),  # Output: px, py
            ]
        )

        # Create dummy training data
        X = np.random.randn(100, 4).astype(np.float32)
        y = np.random.randn(100, 2).astype(np.float32)

        # Test compilation with both loss types
        for use_symmetry in [False, True]:
            loss_fn = custom_loss_wrapper(
                normFac=100, use_symmetry=use_symmetry, symmetry_weight=0.5
            )

            # Should compile without errors
            model.compile(optimizer="adam", loss=loss_fn, metrics=["mae"])

            # Should be able to run one training step
            history = model.fit(X, y, epochs=1, verbose=0)

            # Should produce finite loss
            final_loss = history.history["loss"][0]
            self.assertTrue(np.isfinite(final_loss))


if __name__ == "__main__":
    unittest.main(verbosity=2)
