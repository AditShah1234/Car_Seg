import torch
import sys
import torchvision

sys.path.append("../")

from util.gaussian_blur import gaussian_blur
import logging
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SSD:
    def __init__(self, sub_model=None, conf_threshold=0.2):
        self.model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
        self.class_id = 3  # Class ID for 'car' in COCO dataset
        self.conf_threshold = conf_threshold

    def infer(self, frame_list):
        self.model.eval()
        frame = [
            torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            for image in frame_list
        ]
        # frame = torch.tensor(frame)
        results = self.model(frame)
        return results

    def infer_batch(self, image_list, batch_size=16):
        self.model.eval()
        results = []

        start_time = time.time()
        no_images = len(image_list)
        for i in tqdm(range(0, no_images, batch_size)):
            if i + batch_size > no_images - 1:
                f = image_list[i:]
            else:
                f = image_list[i : i + batch_size]
            r = self.infer(f)
            results.extend(r)
        end_time = time.time()
        self.results = results
        logger.info(
            "Avg time for prediction per frame done {}".format(
                (end_time - start_time) / no_images
            )
        )
        logger.info("Prediction done")
        return results

    def apply_gausian_blur(self, image_list, results):
        if results == None:
            results = self.results
        output = []
        for i in zip(image_list, results):
            frame, result = i[0], i[1]

            car_boxes = []
            for i, label in enumerate(result["labels"]):
                if label == self.class_id and result["scores"][i] > self.conf_threshold:
                    car_boxes.append(result["boxes"][i].tolist())

            # Apply Gaussian blur on detected cars
            if car_boxes:
                gb = gaussian_blur()
                frame = gb.apply_gaussian_blur(frame, car_boxes)
                output.append(frame)

        return output
