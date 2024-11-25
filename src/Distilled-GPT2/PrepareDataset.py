import numpy as np  # type: ignore
import torch  # type: ignore
import os
import sys
from datasets import Dataset  # type: ignore


# Add the path to the parent directory to augment search for module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from Utils import (
    load_model_and_tokenizer,
    get_single_label_dataset,
    tokenize_dataset,
    format_datasets_for_pytorch,
    train_model,
    compute_model_accuracy,
)

from PlotHelper import plot_confusion_matrix


# GLOBAL VARIABLES
SAVING_PATH = "../../Results-Distilled-GPT2"
MODEL_PATH = "/opt/models/distilgpt2"


def prepare_datasets(tokenizer):
    """
    Prepare the datasets for training and evaluation.

    Returns:
        tuple: A tuple containing the training, evaluation, and test datasets.
    """
    ds_train, ds_validation, ds_test = get_single_label_dataset()

    tokenized_train = tokenize_dataset(ds_train, tokenizer)
    tokenized_validation = tokenize_dataset(ds_validation, tokenizer)
    tokenized_test = tokenize_dataset(ds_test, tokenizer)

    train_dataset, eval_dataset, test_dataset = format_datasets_for_pytorch(
        tokenized_train, tokenized_validation, tokenized_test
    )
    return train_dataset, eval_dataset, test_dataset


if __name__ == "__main__":

    tokenizer, model = load_model_and_tokenizer(MODEL_PATH)

    train_dataset, eval_dataset, test_dataset = prepare_datasets(tokenizer)

    trained_model = train_model(model, train_dataset, eval_dataset)

    accuracy = compute_model_accuracy(trained_model, test_dataset, batch_size=32)
    print(accuracy)

    plot_confusion_matrix(
        trained_model, test_dataset, batch_size=32, saving_path=SAVING_PATH
    )
