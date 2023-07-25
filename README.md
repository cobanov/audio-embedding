# audio-embedding

A simple Python script for extracting audio embeddings.

## Requirements

- PyTorch
- Transformers
- fairseq

## Model

Download the provided model

- [Download Link](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt)
- [Google Drive]
- [Huggingface]

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Here's an example of how you can use audio_embeddings:

```
python audio_embedding.py -i demo/sample_audio.wav -o outputs/short.npy -b 1280 -f 16000
```

```
usage: audio_embedding.py [-h] [-i INPUT] [-o OUTPUT] [-b BLOCK] [-f FREQ]

Image caption CLI

optional arguments:
  -h, --help                        show this help message and exit
  -i INPUT, --input INPUT           Input directory path, such as ./sample.wav)
  -o OUTPUT, --output OUTPUT        Output directory, such as output.csv
  -b BLOCK, --block BLOCK           Block length
  -f FREQ, --freq FREQ              Audio file frequency
```

## To-Do

- Clipping silence
- Model downloader
- Drop duplicate rows and columns

## License

This project is licensed under the [Apache Licence 2.0](https://opensource.org/licenses/apache).
