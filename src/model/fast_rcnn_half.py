import torch
import sys
import torchvision

sys.path.append("../")
from util.gaussian_blur import gaussian_blur
import logging
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


class fast_rcnn_half:
    def __init__(self, sub_model="resnet50", conf_threshold=0.2):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True
        )
        self.model.half()
        self.classe_id = 3  # Class ID for 'car' in COCO dataset
        self.conf_threshold = conf_threshold

    def infer(self, frame_list):
        """
        For final inference

        Args:
            frame_list : List of all the frames

        Returns:
            result :Output of the model

        """
        self.model.eval()
        frame = [
            torch.from_numpy(image).permute(2, 0, 1).half() / 255.0
            for image in frame_list
        ]  # Convert the image to tensor
        results = self.model(frame)  # Infer
        return results

    def infer_batch(self, image_list, batch_size=16):
        """
        For batch inference converting frames to batch

        Args:
            image_list : List of all the frames
            batch_size: Batch Size

        Returns:
            result: concatenated output of all the batchs

        """
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
            results.append(r)
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
        """
        Applying gausian blur

        Args:
            image_list : List of all the frames
            result: Output of the model

        Returns:
            output: Result after putting the gaussian blur on the cars

        """
        if results == None:
            results = self.results
        output = []
        for i in zip(image_list, results):
            frame, result = i[0], i[1]

            car_boxes = []
            for i, label in enumerate(result["labels"]):
                if (
                    label == self.classe_id
                    and result["scores"][i] > self.conf_threshold
                ):  # Allowing car class only andallowing only prediction above the threshold
                    car_boxes.append(result["boxes"][i].tolist())

            # Apply Gaussian blur on detected cars
            if car_boxes:
                gb = gaussian_blur()
                frame = gb.apply_gaussian_blur(frame, car_boxes)  # adding Gausian Blur
                output.append(frame)

        return output
