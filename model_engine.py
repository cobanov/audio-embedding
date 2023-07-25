import fairseq
from transformers import Wav2Vec2Processor


def get_model(model_path="model/wav2vec_large.pt"):
    """
    It loads the model from the path you provide

    :param model_path: The path to the model you want to use, defaults to wav2vec_large.pt (optional)
    :return: The model is being returned.
    """

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [model_path]
    )
    return model[0]


def get_processor(model_path="facebook/wav2vec2-base-960h"):
    """
    It loads a pre-trained model from Hugging Face's model hub

    :param model_path: The path to the pretrained model, defaults to facebook/wav2vec2-base-960h
    (optional)
    :return: A Wav2Vec2Processor object
    """
    return Wav2Vec2Processor.from_pretrained(model_path)
