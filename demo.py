from audio_embedding import extract_embeddings
from model_engine import get_model, get_processor
from utils import concat_and_rescale, save_embeddings

from tqdm import tqdm
import pandas as pd
import numpy as np
import uuid
import os


model = get_model()
processor = get_processor()


BLOCK_LENGTH = 1280
TARGET_SR = 16000


# AUDIO_PATH = r"D:\RAS\DVORAK_UMAP\dvorak_01.wav"
# OUTPUT_PATH = f"./outputs/embedding_{uuid.uuid4()}"

root_path = r"D:\RAS\DVORAK_UMAP"

for filename in tqdm(os.listdir(root_path)[13:]):
    # File
    ABS_PATH = os.path.join(root_path, filename)
    OUTPUT_PATH = f"./outputs/embedding_{os.path.splitext(filename)[0]}.npy"

    # Extract Embeddings
    raw_embeddings = extract_embeddings(
        audio_path=ABS_PATH,
        model=model,
        processor=processor,
        block_length=BLOCK_LENGTH,
        target_sr=TARGET_SR,
    )
    embeddings = concat_and_rescale(raw_embeddings)

    # Save Embeddings
    save_embeddings(OUTPUT_PATH, embeddings)
