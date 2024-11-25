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
    format_dataset_for_pytorch,
    format_datasets_for_pytorch,
    prepare_datasets_for_training,
    define_training_arguments,
    initialize_trainer,
    train_and_evaluate_model,
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


def get_trained_model(tokenizer, model, train_dataset, eval_dataset, test_dataset):

    # Define training arguments
    training_args = define_training_arguments()

    # Initialize trainer
    trainer = initialize_trainer(
        model, training_args, train_dataset, eval_dataset, tokenizer
    )

    # Train and evaluate model
    trained_model = train_and_evaluate_model(trainer, test_dataset)
    return trained_model


if __name__ == "__main__":

    tokenizer, model = load_model_and_tokenizer(MODEL_PATH)

    train_dataset, eval_dataset, test_dataset = prepare_datasets(tokenizer)

    trained_model = get_trained_model(tokenizer, model, train_dataset, eval_dataset, test_dataset)
