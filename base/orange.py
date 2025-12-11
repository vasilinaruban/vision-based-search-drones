from __future__ import annotations


from typing import Any

try:
    from rknnlite.api import RKNNLite
except ImportError:  
    RKNNLite = None  

from src.models.models import Net


class NeuroModule:

    _PROCESSES_NUMBER = 3

    def __init__(self, model_name: str, cores_list: int, q_input: Any):
        self.net = Net(model_name, cores_list, q_input)

    def run_inference(self) -> None:
        self.net.inference.start()


class RK3588:

    _CORES = [
        getattr(RKNNLite, "NPU_CORE_0", 0),
        getattr(RKNNLite, "NPU_CORE_1", 1),
        getattr(RKNNLite, "NPU_CORE_2", 2),
        getattr(RKNNLite, "NPU_CORE_AUTO", 0),
        getattr(RKNNLite, "NPU_CORE_0_1", 0),
        getattr(RKNNLite, "NPU_CORE_0_1_2", 0),
    ]

    def __init__(self, model_name: str, camera: Any):
        if RKNNLite is None:
            raise ImportError(
                "rknnlite is not installed. Online inference on RK3588 is "
                "available only on the target board with RKNN SDK installed."
            )

        self._camera = camera
        self._neuro = NeuroModule(model_name, self._CORES[5], self._camera._queue)

    @property
    def net(self) -> Net:
        return self._neuro.net

    def run_inference(self) -> None:
        self._neuro.run_inference()
