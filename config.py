import argparse
from typing import Any, Dict

import yaml


class Config:
    """Configuration management class"""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    def get(self, key: str, default=None):
        """Get nested config value using dot notation"""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """Set nested config value using dot notation"""
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Return full config as dictionary"""
        return self._config

    def save(self, path: str):
        """Save config to YAML file"""
        with open(path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False)


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)


def merge_config_with_args(config: Config, args: argparse.Namespace) -> Config:
    """Override config values with command line arguments"""

    # Override with command line arguments if provided
    if hasattr(args, "epochs") and args.epochs is not None:
        config.set("training.epochs", args.epochs)
    if hasattr(args, "batch_size") and args.batch_size is not None:
        config.set("training.batch_size", args.batch_size)
    if hasattr(args, "input") and args.input is not None:
        config.set("paths.input", args.input)
    if hasattr(args, "output") and args.output is not None:
        config.set("paths.output", args.output)
    if hasattr(args, "mode") and args.mode is not None:
        config.set("training.mode", args.mode)
    if hasattr(args, "model") and args.model is not None:
        config.set("model.type", args.model)
    if hasattr(args, "units") and args.units is not None:
        config.set("model.units", list(map(int, args.units)))
    if hasattr(args, "maxNPF") and args.maxNPF is not None:
        config.set("data.maxNPF", args.maxNPF)
    if hasattr(args, "normFac") and args.normFac is not None:
        config.set("training.normFac", args.normFac)
    if hasattr(args, "loss_symmetry") and args.loss_symmetry:
        config.set("loss.use_symmetry", True)
    if hasattr(args, "symmetry_weight") and args.symmetry_weight is not None:
        config.set("loss.symmetry_weight", args.symmetry_weight)
    if hasattr(args, "quantized") and args.quantized is not None:
        config.set("quantization.enabled", True)
        config.set("quantization.total_bits", int(args.quantized[0]))
        config.set("quantization.int_bits", int(args.quantized[1]))
    if hasattr(args, "compute_edge_feat") and args.compute_edge_feat is not None:
        config.set("data.compute_edge_feat", args.compute_edge_feat)
    if hasattr(args, "edge_features") and args.edge_features is not None:
        config.set("data.edge_features", args.edge_features)

    return config


def create_default_config() -> Config:
    """Create default configuration"""
    default_config = {
        "model": {
            "type": "dense_embedding",
            "units": [64, 32, 16],
            "activation": "tanh",
            "with_bias": False,
            "emb_out_dim": 8,
        },
        "data": {
            "maxNPF": 128,
            "n_features_pf": 6,
            "n_features_pf_cat": 2,
            "compute_edge_feat": 0,
            "edge_features": [],
            "preprocessed": True,
            "normFac": 100,
        },
        "training": {
            "workflow_type": "dataGenerator",
            "epochs": 100,
            "batch_size": 1024,
            "mode": 0,
            "normFac": 100,
        },
        "loss": {"use_symmetry": False, "symmetry_weight": 1.0},
        "quantization": {"enabled": False, "total_bits": 7, "int_bits": 2},
        "callbacks": {
            "early_stopping": {"patience": 40, "monitor": "val_loss"},
            "reduce_lr": {"factor": 0.5, "patience": 4, "min_lr": 0.000001},
            "cyclical_lr": {"base_lr": 0.0003, "max_lr": 0.001, "mode": "triangular2"},
        },
        "optimizer": {"type": "adam", "learning_rate": 1.0, "clipnorm": 1.0},
    }
    return Config(default_config)
