import glob
import time
import logging
from pathlib import Path

import cv2
import torch
import hydra

from post_proc import draw_results
from zone import StaticZone
from config import GaleatiConfig
from g_models.yolo import Yolo
from g_utils import chunks_generator, CocoPresent
from frame_meta import FrameMeta
import numpy as np

logger = logging.getLogger(__name__)


class App:
    def __init__(self, config: GaleatiConfig):
        self.config = config
        self.coco_maker = CocoPresent(".")
        self.model_config = config.model
        self.source = self.config.source + "*.jpg"
        self.dest = Path(self.config.dest)
        self.dest.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda")
        # self.zones_type=self.config.zones_type
        # self.zones={}
        # for zone in  self.zones_type:
        #     self.zones[zone]=StaticZone(zone)
        self.yolo = Yolo(
            weight=self.config.model.yolo_weights_path,
            model_frame_size=self.config.model.yolo_frame_size,
            device=self.device,
            confidence_threshold=self.config.model.confidence_threshold,
            iou_threshold=self.config.model.iou_threshold,
        )

    def start(self):
        logger.info("Files from source %s", self.source)
        files = glob.glob(self.source)
        logger.info("Files count %s", len(list(files)))
        batched = chunks_generator(files, 10)
        for batch in batched:
            start = time.perf_counter()
            batch_list_pic = []
            for pic in batch:
                batch_list_pic.append(cv2.imread(pic))
            frames_meta = []

            frames = torch.from_numpy(np.stack(batch_list_pic)).to(self.device)

            frames = frames[:, :, :, [2, 1, 0]].permute(0, 3, 1, 2)
            print(frames.shape)
            predicts = self.yolo(frames)
            print(predicts)
            for i, detection in enumerate(predicts):
                frame_meta = FrameMeta(
                    model_config=self.config.model,
                    frame=batch_list_pic[i],
                    frame_ix=i,
                    raw_detections=detection,
                    file_name=batch[i],
                )
                frames_meta.append(frame_meta)

            for frame_meta in frames_meta:
                if frame_meta.is_empty():
                    continue
                for detection in frame_meta.detections:
                    detection.make_crop()
            for frame_meta in frames_meta:
                file_path_post = Path(frame_meta.file_name)
                for detection in frame_meta.detections:
                    self.coco_maker.add_an(file_path_post.name, detection.get_bbox())
                marked_frame = draw_results(frame_meta)
                cv2.imwrite(str(self.dest / file_path_post.stem) + "_res.jpg", marked_frame)
            delta = time.perf_counter() - start
            logger.info("Batch time  %s ", str(delta))

        self.coco_maker.create_json()


@hydra.main(config_path="../config", config_name="config")
def main(galeati_config: GaleatiConfig):
    app = App(galeati_config)

    try:
        app.start()
    finally:
        logger.info("Stop")


if __name__ == "__main__":
    main()
