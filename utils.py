from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


def concat_and_rescale(embeddings):
    """
    It takes a list of tensors, converts them to numpy arrays, concatenates them, scales them, and
    returns a pandas dataframe

    :param embeddings: a list of embeddings
    :return: A dataframe with the scaled embeddings.
    """

    # Concat all embeddings
    print("Concatenating all embeddings together.")
    embeddings_np = np.vstack(
        [i.squeeze().detach().numpy().transpose() for i in embeddings]
    )

    # Min Max Scaling
    print("\nMin Max Scaling")
    scaler = MinMaxScaler()
    scaled_np = scaler.fit_transform(embeddings_np)

    # Drop duplicate rows
    unique_rows = np.unique(scaled_np, axis=0)

    print(f"{scaled_np.shape[0] - unique_rows.shape[0]} rows deleted.")
    print(unique_rows.shape)

    return unique_rows




def save_embeddings(output_path, np_array):
    """
    It takes a dataframe and an output path, and saves the dataframe to the output path

    :param dataframe: The dataframe containing the embeddings
    :param output_path: The path to the output file
    """

    print("\nSaving process started.")
    print(np_array.shape)
    np.save(output_path, np_array, allow_pickle=True)
    print("Saved!")
