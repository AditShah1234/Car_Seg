import torchvision.transforms as T
import logging
import argparse
from util.utils import video_utils
from util import load_model, pipline, log_file


log_file.file_logging("demo.log", logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    """
    main function validates the args and send it to infer

    Args:
        args : Containing info for inference like(Batch Size, Confidence , Skip Frame,..etc)

    Returns:
        None

    """
    if args.skip_frame < 1:  # checking for skip frame and if less then logging
        logger.warning("Selected skip frame less then 1 {}".format(args.skip_frame))
        assert False
    else:
        skip_frame = args.skip_frame
    if str(args.input_file).endswith(("mp4", "mkv", "mov")):
        video_util = video_utils(frame_skip=skip_frame)
        input_file = args.input_file
    if args.confidence > 1 or args.confidence < 0:  # Checking for confidence level
        logger.warning("Confidence out of range {}".format(args.confidence))
        assert False
    else:
        confidence = args.confidence

    if args.batch_size < 1:  # Checking for batch size
        logger.warning("Selected batch size less then 1 {}".format(args.skip_frame))
        assert False
    else:
        batch_size = args.batch_size

    output_file = args.output_file

    logger.info("Selected Input {}".format(str(args.input_file)))
    logger.info("Selected Model {}".format(str(args.model_name)))
    logger.info("Selected Output {}".format(str(args.output_file)))
    logger.info("Selected Confidence {}".format(str(args.confidence)))
    logger.info("Selected Skip Frame {}".format(str(args.skip_frame)))
    logger.info("Selected Batch Size {}".format(str(args.batch_size)))

    model = load_model.load_model(args.model_name, confidence)
    pipline.infer_video_batch(model, input_file, video_util, batch_size, output_file)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script for running a model on a given input video or image."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["YOLOv5", "fastRCNN", "Fast_rcnn_half", "SSD"],
        default="YOLOv5",
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
        default="output_yolo_4.mp4",
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
        "--skip_frame", type=int, default=1, help="To make it fast we skip the frame"
    )

    parser.add_argument(
        "--confidence",
        type=int,
        default=0.4,
        help="Depending upon requirement set it in range of [0-1]",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    main(args)
