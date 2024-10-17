import torchvision.transforms as T
import logging
import argparse
import demo


def main(args):
    demo.main(args=args)


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
        "--batch_size", type=int, default=2, help="Batch size should be more than 0"
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
            main(args)
    else:
        main(args)


if __name__ == "__main__":
    args = parse_arguments()

    compare(args)
