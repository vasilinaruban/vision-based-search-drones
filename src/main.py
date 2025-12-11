# src/main.py
from __future__ import annotations


from multiprocessing import Queue
from typing import Optional

from omegaconf import DictConfig

from utils import PostProcess, Visualizer
from base import Camera, RK3588


RKNK_POSTPROCESS_CFG = {
    "img_size": 640,
    "scales": [24, 48, 96, 192, 384],
    "aspect_ratios": [1, 0.5, 2],
    "top_k": 200,
    "max_detections": 100,
    "nms_score_thre": 0.5,
    "nms_iou_thre": 0.5,
    "visual_thre": 0.5,
}


def run_online_inference(cfg: Optional[DictConfig] = None) -> None:


    source = 0
    model_name = "YOLO"

    if cfg is not None and "inference" in cfg:
        inf_cfg = cfg.inference
        source = getattr(inf_cfg, "source", source)
        model_name = getattr(inf_cfg, "model_name", model_name)


    q_cam_to_npu: Queue = Queue(maxsize=3)

    camera = Camera(source=source, queue=q_cam_to_npu)

    npu = RK3588(model_name=model_name, camera=camera)

    postprocess = PostProcess(
        npu.net.inference.q_out,
        RKNK_POSTPROCESS_CFG,
        onnx=False,
        result_saver_kwargs=dict(
            eps=cfg.inference.clustering.eps,
            min_samples=cfg.inference.clustering.min_samples,
            metric=cfg.inference.clustering.metric,
            drone_height_m=cfg.inference.drone_height,
            fov_deg=cfg.inference.fov_degrees,
        ),
    )
    visualizer = Visualizer()

    camera.start()
    npu.run_inference()
    postprocess.run()

    while True:
        data = postprocess.get_outputs()
        if data is None:
            break
        frame, outputs = data
        visualizer.show_results(frame, outputs)


if __name__ == "__main__":
    run_online_inference()
