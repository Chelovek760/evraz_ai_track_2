from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    yolo_weights_path: str
    yolo_frame_size: int

    confidence_threshold: float = 0.8
    iou_threshold: float = 0.5

    yolo_max_batch_size: int = 10

    max_dist: float = 0.2
    min_confidence: float = 0.3
    nn_budget: int = 100
    n_init: int = 5


@dataclass
class GaleatiConfig:
    model: ModelConfig
    source: str = "/app/data/test/images/"
    dest: str = "/app/data_res/images/test/"


cs = ConfigStore.instance()
cs.store(name="base_config", node=GaleatiConfig)
