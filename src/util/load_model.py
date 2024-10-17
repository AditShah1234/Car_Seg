from model.yolo import YOLO
from model.fast_rcnn import FastRCNN
from model.ssd import SSD
from model.fast_rcnn_half import fast_rcnn_half
from model.onnx import YOLO_onnx
import logging

logger = logging.getLogger(__name__)


def load_model(model_name, confidence):
    if str(model_name) == "YOLOv5":
        model = YOLO(conf_threshold=confidence)
    elif str(model_name) == "fastRCNN":
        model = FastRCNN(conf_threshold=confidence)
    elif str(model_name) == "SSD":
        model = SSD(conf_threshold=confidence)
    elif str(model_name) == "Fast_rcnn_half":
        model = fast_rcnn_half(conf_threshold=confidence)
    elif str(model_name) == "onnx":
        model = YOLO_onnx(conf_threshold=confidence)
    else:
        logger.warning("Wrong Model Selected")
        assert False
    return model
