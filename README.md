# audio-embedding

A simple Python script for extracting audio embeddings.

## Requirements

- PyTorch
- Transformers
- fairseq

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Here's an example of how you can use audio_embeddings:

```
python audio_embedding.py -i short_sample.wav -o short.npy -b 1280 -f 16000
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

## License

This project is licensed under the [Apache Licence 2.0](https://opensource.org/licenses/apache).
