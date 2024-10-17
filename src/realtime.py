import torchvision.transforms as T
import logging
import argparse
from util import log_file, pipline, load_model

import cv2

log_file.file_logging("real_time.log", logging.INFO)
logger = logging.getLogger(__name__)


class video_utils:
    def __init__(
        self,
        model,
        output_video_path=None,
        frame_width=None,
        frame_height=None,
        fps=None,
        frame_skip=64,
    ):
        self.model = model
        self.output_video_path = output_video_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.frame_skip = frame_skip

    def video_process(self, input_video_path=0):

        video = cv2.VideoCapture(input_video_path)
        out = []

        if not self.frame_width:
            self.frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        if not self.frame_height:
            self.frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not self.fps:
            self.fps = video.get(cv2.CAP_PROP_FPS)

        frame_idx = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            if frame_idx % self.frame_skip == 0:
                frame = pipline.infer_realtime(self.model, frame)[0]

            frame_idx += 1
            out.append(frame)

        video.release()

        cv2.destroyAllWindows()
        return out

    def save_video(self, list_frame):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            self.output_video_path,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height),
        )
        for frame in list_frame:
            out.write(frame)
        logger.info("video saved")
        out.release()


def main(args):

    if args.skip_frame < 1:  # checking for skip frame and if less then logging
        logger.warning("Selected skip frame less then 1 {}".format(args.skip_frame))
        assert False
    else:
        skip_frame = args.skip_frame

    if args.confidence > 1 or args.confidence < 0:  # Checking for confidence level
        logger.warning("Confidence out of range {}".format(args.confidence))
        assert False
    else:
        confidence = args.confidence

    output_file = args.output_file

    logger.info("Selected Input {}".format(str(args.input_file)))
    logger.info("Selected Model {}".format(str(args.model_name)))
    logger.info("Selected Output {}".format(str(args.output_file)))
    logger.info("Selected Confidence {}".format(str(args.confidence)))
    logger.info("Selected Skip Frame {}".format(str(args.skip_frame)))
    model = load_model.load_model(args.model_name, confidence)
    process = video_utils(model, output_video_path=output_file, frame_skip=skip_frame)
    out = process.video_process(input_video_path=args.input_file)
    process.save_video(out)


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
        default="output_YOLOv5.mp4",
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
        "--skip_frame", type=int, default=128, help="To make it fast we skip the frame"
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
