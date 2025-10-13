#!/usr/bin/env python3
"""
Pruning utilities for TensorFlow models.

"""
from pathlib import Path

from tensorflow.keras import callbacks, layers, models
from tensorflow_model_optimization.sparsity.keras import (
    ConstantSparsity,
    PolynomialDecay,
    PruningSummaries,
    UpdatePruningStep,
    prune_low_magnitude,
)

from config import Config


def get_pruning_config(config: Config):
    """Configure pruning params from the training config."""

    if not config.get("pruning.prune", False):
        return None

    pruning_schedule = config.get("pruning.pruning_schedule")
    target_sparsity = config.get("pruning.target_sparsity")
    begin_step = config.get("pruning.begin_step")
    frequency = config.get("pruning.frequency")

    if pruning_schedule == "polynomial":
        return PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=target_sparsity,
            begin_step=begin_step,
            end_step=config.get("pruning.end_step"),
            frequency=frequency,
            power=3.0,
        )
    elif pruning_schedule == "constant":
        return ConstantSparsity(
            target_sparsity=target_sparsity, begin_step=begin_step, frequency=frequency
        )
    else:
        raise ValueError(f"Unsupported pruning schedule: {pruning_schedule}")


def apply_model_pruning(keras_model, pruning_config, model_type):
    """Apply pruning to the keras model."""

    def apply_pruning_to_dense(layer):
        if isinstance(layer, layers.Dense):
            return prune_low_magnitude(
                layer,
                pruning_schedule=pruning_config,
            )
        return layer

    if model_type == "dense_embedding":
        return models.clone_model(
            keras_model,
            clone_function=apply_pruning_to_dense,
        )
    else:
        raise ValueError(f"Model type {model_type} not implemented for pruning.")


def get_pruning_callbacks(path_out, target_sparsity):
    """Get pruning callbacks for training."""
    pruning_callbacks = [
        UpdatePruningStep(),
        PruningSummaries(str(Path(path_out) / "pruning_logs")),
        SparsityCallback(target_sparsity, str(Path(path_out) / "pruning_logs")),
    ]
    return pruning_callbacks


class SparsityCallback(callbacks.Callback):
    """Callback to monitor and log sparsity during training."""

    def __init__(self, target_sparsity, log_dir):
        super().__init__()
        self.target_sparsity = target_sparsity
        self.log_dir = Path(log_dir)
        self.sparsity_log = []

        self.log_dir.mkdir(parents=True, exist_ok=True)

    def calculate_model_sparsity(self):
        """Calculate current model sparsity."""
        total_weights = 0
        zero_weights = 0

        for layer in self.model.layers:
            if hasattr(layer, "kernel") and layer.get_weights():
                weights = layer.get_weights()[0]
                total_weights += weights.size
                zero_weights += (weights == 0).sum()

        return zero_weights / total_weights if total_weights > 0 else 0.0

    def on_epoch_end(self, epoch, logs=None):
        # Calculate current sparsity
        current_sparsity = self.calculate_model_sparsity()
        self.sparsity_log.append(
            {
                "epoch": epoch,
                "sparsity": current_sparsity,
                "target": self.target_sparsity,
            }
        )

        # Log sparsity information
        print(
            f"Epoch {epoch}: Current Sparsity: {current_sparsity:.4f}, Target Sparsity: {self.target_sparsity:.4f}"
        )


def analyse_pruned_model(original_model, pruned_model, test_data, path_out):

    original_pred = original_model.predict(test_data)
    pruned_pred = pruned_model.predict(test_data)

    # TODO: impolement these metrics in pruning/analysis.py

    metrics = {
        #'model_size_reduction': calculate_size_reduction(original_model, pruned_model),
        #'inference_speedup': measure_inference_speed(original_model, pruned_model, test_data),
        #'accuracy_retention': calculate_accuracy_retention(original_pred, pruned_pred),
        #'memory_usage': measure_memory_usage(original_model, pruned_model),
    }

    # save_pruning_report(metrics, path_out) #TODO: implement in pruning/analysis.py

    return metrics
