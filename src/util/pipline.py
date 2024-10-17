import logging

logger = logging.getLogger(__name__)


def infer_video_batch(model, input_file, video_util, batch_size, output_file):
    """
    Infer function works on inference

    Args:
        model : Model object for inference
        input_file : Input file path
        video_util :Video read object
        batch_size : Batch Size
        output_file : Output file paths

    Returns:
        None

    """
    frames = video_util.load_video(input_file)  # convert video to frame
    result = model.infer_batch(frames, batch_size)  # Inference as per the model
    output = model.apply_gausian_blur(frames, result)  # Applying Gausian Blur
    video_util.save_video(output, output_file)  # Converting frame back to video


def infer_realtime(model, frames):
    """
    Infer function works on inference

    Args:
        model : Model object for inference
        frame : Input file path



    Returns:
        output

    """

    result = model.infer([frames])  # Inference as per the model
    if not isinstance(result, list):
        result = [result]
    output = model.apply_gausian_blur([frames], result)  # Applying Gausian Blur
    return output
