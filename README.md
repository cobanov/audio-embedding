# audio-embedding

A simple Python script for extracting audio embeddings.

## Requirements

- PyTorch
- Transformers
- fairseq

## Model

Download the provided model

- [Download Link](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt)
- [Google Drive](https://drive.google.com/file/d/1s9MpdjX41jfJQwzj1FrrfD5NJhTS8ZqM/view?usp=sharing)
- [Huggingface](https://huggingface.co/mertcobanov/cobanov-weights/resolve/main/wav2vec_large.pt)

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
### Functional way

```python
from audio_embedding import extract_embeddings
from model_engine import get_model, get_processor
from utils import concat_and_rescale, save_embeddings

import pandas as pd
import numpy as np
import uuid

AUDIO_PATH = r"./demo/sample_audio.wav"
OUTPUT_PATH = f"./outputs/embedding_{uuid.uuid4()}"

BLOCK_LENGTH = 1280
TARGET_SR = 16000

model = get_model()
processor = get_processor()

# Extract Embeddings
raw_embeddings = extract_embeddings(
    audio_path=AUDIO_PATH,
    model=model,
    processor=processor,
    block_length=BLOCK_LENGTH,
    target_sr=TARGET_SR,
)

# Embedding post-processing
embeddings = concat_and_rescale(raw_embeddings)
print(embeddings.shape)

# Save Embeddings
save_embeddings(OUTPUT_PATH, embeddings)
```

## To-Do

- Clipping silence
- Model downloader
- Drop duplicate rows and columns

## License

This project is licensed under the [Apache Licence 2.0](https://opensource.org/licenses/apache).
