from __future__ import annotations

from typing import List, Optional

from hydra import compose, initialize
from omegaconf import DictConfig

CONFIG_PATH = "../configs"


def _load_config(overrides: Optional[List[str]] = None) -> DictConfig:
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize(version_base="1.3", config_path=CONFIG_PATH):
        cfg = compose(config_name="main", overrides=overrides or [])
    return cfg


def train(overrides: Optional[List[str]] = None) -> None:
    cfg = _load_config(overrides)

    from . import training as training_module

    training_module.train(cfg)


def infer(overrides: Optional[List[str]] = None) -> None:
    cfg = _load_config(overrides)

    try:
        from . import training as training_module

        training_module._ensure_dataset(cfg)
    except Exception:
        pass

    from . import main as inference_main

    inference_main.run_online_inference(cfg)
