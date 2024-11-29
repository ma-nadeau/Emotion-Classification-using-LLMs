import numpy as np  # type: ignore
import torch  # type: ignore
import os
import sys
from datasets import Dataset  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
import string

from LLM import train_evaluate_hyperparams
from PlotHelper import plot_train_vs_validation_accuracy

# Add the path to the parent directory to augment search for module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from Utils import (
    load_model_and_tokenizer,
    get_single_label_dataset,
    tokenize_dataset,
    oversample_dataset,
    undersample_features,
    remove_label,
    format_datasets_for_pytorch,
)

from LLM import (
    train_model_trainer,
    predict_trainer,
)

from PlotHelper import plot_confusion_matrix, plot_distribution_of_datasets

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


def compute_accuracy(prediction, labels, model_name):
    accuracy = accuracy_score(labels, prediction)
    print(f"Accuracy {model_name}: {accuracy}")
    return accuracy


if __name__ == "__main__":
    tokenizer, model = load_model_and_tokenizer(MODEL_PATH)

    train_dataset, eval_dataset, test_dataset = prepare_datasets(tokenizer)

    # train_dataset = undersample_features(train_dataset)
    # train_dataset = oversample_dataset(train_dataset)

    print(len(test_dataset["labels"]))

    # plot_distribution_of_datasets(
    #     train_dataset, eval_dataset, test_dataset, saving_path=SAVING_PATH
    # )

    # untrainded_model_prediction = predict_trainer(model, test_dataset, batch_size=16)
    #
    # trained_model = train_model_trainer(model, train_dataset, eval_dataset=eval_dataset)
    #
    # prediction = predict_trainer(trained_model, test_dataset, batch_size=32)
    #
    # prediction_train = predict_trainer(trained_model, train_dataset, batch_size=32)
    #
    # labels_test = test_dataset["labels"]
    # labels_train = train_dataset["labels"]
    #
    # compute_accuracy(prediction, labels_test, "test")
    # compute_accuracy(prediction_train, labels_train, "train")
    # compute_accuracy(untrainded_model_prediction, labels_test, "untrained")
    #
    # plot_confusion_matrix(prediction, labels_test, saving_path=SAVING_PATH)

    batch_sizes = [8, 16, 32, 64, 128]
    epochs = [3, 5, 10, 15, 20]
    learning_rates = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4]

    results = train_evaluate_hyperparams(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        test_dataset,
        batch_sizes,
        epochs,
        learning_rates,
    )

    # Use the same output directory as the Trainer
    output_dir = "./output"

    # Plot Train vs Validation Accuracy for different hyperparameter pairs
    plot_train_vs_validation_accuracy(
        results, param_x="learning_rate", param_y="batch_size", output_dir=output_dir
    )

    plot_train_vs_validation_accuracy(
        results, param_x="batch_size", param_y="epochs", output_dir=output_dir
    )

    plot_train_vs_validation_accuracy(
        results, param_x="epochs", param_y="learning_rate", output_dir=output_dir
    )
