import librosa
import argparse
import torch
from tqdm import tqdm

import model_engine
import utils


def init_parser(**parser_kwargs):
    """
    This function initializes the parser and adds arguments to it
    :return: The parser object is being returned.
    """
    parser = argparse.ArgumentParser(description="Image caption CLI")
    parser.add_argument(
        "-i", "--input", help="Input directory path, such as ./sample.wav)"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory, such as output.csv",
        default="output.csv",
    )
    parser.add_argument("-b", "--block", help="Block length", default=1280, type=int)
    parser.add_argument(
        "-f", "--freq", help="Audio file frequency", default=16000, type=int
    )

    return parser


def extract_embeddings(audio_path, model, processor, block_length, target_sr=16000):
    """
    It takes an audio file, splits it into chunks, and then extracts embeddings for each chunk
    
    :param audio_path: The path to the audio file you want to extract embeddings from
    :param model: the model we're using to extract embeddings
    :param processor: The pre-processing function that will be used to convert the audio into a tensor
    :param block_length: The number of frames to process at a time
    :param target_sr: The sampling rate of the audio, defaults to 16000 (optional)
    :return: A list of embeddings
    """
    sr = librosa.get_samplerate(audio_path)

    # Set the frame parameters to be equivalent to the librosa defaults
    # in the file's native sampling rate
    frame_length = (2048 * sr) // 22050
    hop_length = (512 * sr) // 22050

    # Stream the data, working on 128 frames at a time
    stream = librosa.stream(
        audio_path,
        mono=True,
        block_length=block_length,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    # Audio splits
    with torch.no_grad():
        embeddings = []

        for y in tqdm(list(stream)):
            resampled_audio = librosa.resample(y=y, orig_sr=sr, target_sr=target_sr)
            input_values = processor(
                resampled_audio, sampling_rate=16000, return_tensors="pt"
            ).input_values
            audio_features = model.feature_extractor(input_values)
            aggregated_features = model.feature_aggregator(audio_features)
            embeddings.append(aggregated_features)
        return embeddings


def main():
    # CLI Requirements
    parser = init_parser()
    opt = parser.parse_args()

    # Model
    model = model_engine.get_model()
    model.eval()

    # Processor
    processor = model_engine.get_processor()

    # Extracting
    embeddings = extract_embeddings(opt.input, model, processor, opt.block, opt.freq)
    scaled_np = utils.concat_and_rescale(embeddings)
    utils.save_embeddings(opt.output, scaled_np)


if __name__ == "__main__":
    main()
