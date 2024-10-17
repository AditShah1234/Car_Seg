import torchvision.transforms as T
import logging
import argparse
from util.utils import video_utils
from model.yolo import YOLO
from model.fast_rcnn import FastRCNN
from model.ssd import SSD
from model.fast_rcnn_half import fast_rcnn_half
from util import log_file

log_file.file_logging("compare_run.log", logging.INFO)
logger = logging.getLogger(__name__)


def infer(args):
    if args.skip_frame < 1:
        logger.warning("Selected skip frame les then 1 {}".format(args.skip_frame))
    else:
        skip_frame = args.skip_frame
    if str(args.input_file).endswith(("mp4", "mkv", "mov")):
        video_util = video_utils(frame_skip=skip_frame)
    if args.confidence > 1 or args.confidence < 0:
        logger.warning("Confidence out of range {}".format(args.confidence))
    else:
        confidence = args.confidence

    if str(args.model_name) == "YOLOv5":
        model = YOLO(conf_threshold=confidence)
    elif str(args.model_name) == "fastRCNN":
        model = FastRCNN(conf_threshold=confidence)
    elif str(args.model_name) == "SSD":
        model = SSD(conf_threshold=confidence)
    elif str(args.model_name) == "Fast_rcnn_half":
        model = fast_rcnn_half(conf_threshold=confidence)

    else:

        logger.warning("Wrong Model Selected")

    if args.batch_size < 1:
        logger.warning("Selected batch size less then 1 {}".format(args.skip_frame))
    else:
        batch_size = args.batch_size

    logger.info("Selected Input {}".format(str(args.input_file)))
    logger.info("Selected Model {}".format(str(args.model_name)))
    logger.info("Selected Output {}".format(str(args.output_file)))
    logger.info("Selected Confidence {}".format(str(args.confidence)))
    logger.info("Selected Skip Frame {}".format(str(args.skip_frame)))
    logger.info("Selected Batch Size {}".format(str(args.batch_size)))

    frames = video_util.load_video(args.input_file)
    result = model.infer_batch(frames, batch_size)
    output = model.apply_gausian_blur(frames, result)
    video_util.save_video(output, args.output_file)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script for running a model on a given input video or image."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["YOLOv5", "fastRCNN", "SSD", "all"],
        default="all",
        help="Name of the model",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/hanoi-traffic-clip.mov",
        help="Path to the input video or image file.",
    )

    # Additional optional arguments
    parser.add_argument(
        "--output_file",
        type=str,
        default="output.mp4",
        help="Path to save the output file (default: output.mp4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run the model on (default: cpu).",
    )

    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size should be more than 0"
    )

    parser.add_argument(
        "--skip_frame", type=int, default=16, help="To make it fast we skip the frame"
    )

    parser.add_argument(
        "--confidence",
        type=int,
        default=0.4,
        help="Depending upon requirement set it in range of [0-1]",
    )

    args = parser.parse_args()
    return args


def compare(args):
    if str(args.model_name) == "all":
        for model in ["YOLOv5", "fastRCNN", "SSD"]:
            args.model_name = model
            args.output_file = model + ".mp4"
            infer(args)
    else:
        infer(args)


if __name__ == "__main__":
    args = parse_arguments()

    compare(args)
