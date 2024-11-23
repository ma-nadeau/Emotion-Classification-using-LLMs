import torch # type: ignore
import os
import sys


# TODO: write this function
def fine_tune_gpt2(model_path, tokenized_dataset, output_dir):
    """
    Fine-tunes a GPT model for text generation.

    Args:
        model_path (str): Path to the pre-trained GPT model.
        tokenized_dataset (dict): The preprocessed dataset.
        output_dir (str): Directory to save the fine-tuned model.
    """
    