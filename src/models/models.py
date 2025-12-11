from __future__ import annotations

from multiprocessing import Process, Queue
from pathlib import Path
from typing import Iterable, List, Sequence

from rknnlite.api import RKNNLite

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_PATH = PROJECT_ROOT / "checkpoints"


def _get_models_path() -> Path:
    import os

    env_path = os.getenv("RKNN_MODELS_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_MODELS_PATH


MODELS_PATH = _get_models_path()


RKNN_MODEL_NAMES = {
    "YOLO": "yolo8s__3.rknn",
}


def get_model_paths(model_list: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for name in model_list:
        filename = RKNN_MODEL_NAMES.get(name)
        if filename is None:
            raise KeyError(
                f"Unknown RKNN model name '{name}'. "
                f"Known names: {sorted(RKNN_MODEL_NAMES.keys())}"
            )
        paths.append(MODELS_PATH / filename)
    return paths


class RKNNModelLoader:

    @staticmethod
    def _check_paths(model_paths: Sequence[Path]) -> None:
        missing = [p for p in model_paths if not p.exists()]
        if missing:
            msg = (
                "RKNN model files not found: "
                + ", ".join(str(p) for p in missing)
                + f"\nMake sure you've copied .rknn files to '{MODELS_PATH}' "
                "or set RKNN_MODELS_PATH."
            )
            raise FileNotFoundError(msg)

    @classmethod
    def load_weights(cls, cores: int, model_paths: Sequence[Path]) -> RKNNLite:
        cls._check_paths(model_paths)

        rknn = RKNNLite()
        model_path = model_paths[0]
        ret = rknn.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f"RKNN load_rknn failed for {model_path}, code={ret}")

        ret = rknn.init_runtime(core_mask=cores)
        if ret != 0:
            raise RuntimeError(f"RKNN init_runtime failed, code={ret}")

        return rknn


class Inference(Process):

    def __init__(self, q_input: Queue, rknn: RKNNLite):
        super().__init__()
        self.input: Queue = q_input
        self._rknn = rknn
        self.q_out: Queue = Queue(maxsize=3)

    def run(self) -> None:
        while True:
            data = self.input.get()
            if data is None:
                self.q_out.put(None)
                break

            if isinstance(data, tuple) and len(data) == 2:
                frame, gps_data = data
            else:
                frame = data
                gps_data = None

            outputs = self._rknn.inference(inputs=[frame])
            self.q_out.put((frame, outputs, gps_data))


class Net:

    def __init__(self, model_name: str, cores: int, q_input: Queue):
        model_paths = get_model_paths([model_name])
        rknn = RKNNModelLoader.load_weights(cores, model_paths)
        self.inference = Inference(q_input, rknn)
