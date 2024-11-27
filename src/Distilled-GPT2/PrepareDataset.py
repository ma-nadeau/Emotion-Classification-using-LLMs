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
    predict,
)

from PlotHelper import plot_confusion_matrix, plot_distribution_of_datasets


# GLOBAL VARIABLES
SAVING_PATH = "../Results-Distilled-GPT2"
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


def test(test_dataset):

    labels = test_dataset["labels"]
    prediction = labels

    accuracy = np.mean(np.array(prediction) == np.array(labels))
    print(f"Accuracy: {accuracy}")

    plot_confusion_matrix(prediction, labels, saving_path=SAVING_PATH)


def compute_accuracy(prediction, labels, saving_path, model_name):
    labels = test_dataset["labels"]
    accuracy = np.mean(np.array(prediction) == np.array(labels))
    print(f"Accuracy {model_name}: {accuracy}")


if __name__ == "__main__":

    tokenizer, model = load_model_and_tokenizer(MODEL_PATH)

    train_dataset, eval_dataset, test_dataset = prepare_datasets(tokenizer)

    plot_distribution_of_datasets(
        train_dataset, eval_dataset, test_dataset, saving_path=SAVING_PATH
    )

    trained_model = train_model(model, train_dataset, eval_dataset)

    prediction = predict(trained_model, test_dataset, batch_size=32)
    
    # model = train(model, train_dataset, eval_dataset)
    # prediction = predict_trainer(model, test_dataset, batch_size=32)
    
    labels_test = test_dataset["labels"]
    labels_train = train_dataset["labels"]
    
    compute_accuracy(prediction, labels_test, SAVING_PATH, "test")
    compute_accuracy(prediction, labels_train, SAVING_PATH, "train")
    

    plot_confusion_matrix(prediction, labels_test, saving_path=SAVING_PATH)

