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
    define_training_arguments,
    initialize_trainer,
)


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


def prepare_and_train_model(
    model, tokenizer, train_dataset, eval_dataset, test_dataset
):
    """
    Prepare the datasets for PyTorch, define training arguments, and train the model.

    Args:
        model (PreTrainedModel): The model to train.
        tokenizer (PreTrainedTokenizer): The tokenizer used for tokenization.
        train_dataset (Dataset): The training dataset.
        eval_dataset (Dataset): The evaluation dataset.
        test_dataset (Dataset): The test dataset.

    Returns:
        PreTrainedModel: The trained model.
    """
    training_args = define_training_arguments(output_dir=SAVING_PATH)
    trained_model = initialize_trainer(
        model, training_args, train_dataset, eval_dataset
    )
    return trained_model


if __name__ == "__main__":

    tokenizer, model = load_model_and_tokenizer(MODEL_PATH)

    train_dataset, eval_dataset, test_dataset = prepare_datasets(tokenizer)

    trained_model = prepare_and_train_model(
        model, tokenizer, train_dataset, eval_dataset, test_dataset
    )
