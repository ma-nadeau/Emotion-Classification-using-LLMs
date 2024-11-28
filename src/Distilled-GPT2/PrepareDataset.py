import numpy as np  # type: ignore
import torch  # type: ignore
import os
import sys
from datasets import Dataset  # type: ignore
from sklearn.metrics import accuracy_score # type: ignore


# Add the path to the parent directory to augment search for module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from Utils import (
    load_model_and_tokenizer,
    get_single_label_dataset,
    tokenize_dataset,
    format_datasets_for_pytorch,
)
from LLM import (
    train_model_trainer,
    predict_trainer,
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


def compute_accuracy(prediction, labels, model_name):
    accuracy = accuracy_score(labels, prediction)
    print(f"Accuracy {model_name}: {accuracy}")
    return accuracy
    



if __name__ == "__main__":

    tokenizer, model = load_model_and_tokenizer(MODEL_PATH)
    

    train_dataset, eval_dataset, test_dataset = prepare_datasets(tokenizer)

    print(len(test_dataset["labels"]))


    plot_distribution_of_datasets(
        train_dataset, eval_dataset, test_dataset, saving_path=SAVING_PATH
    )

    untrainded_model_prediction =  predict_trainer(model, test_dataset, batch_size=16)
    
    trained_model = train_model_trainer(model, train_dataset,eval_dataset=eval_dataset)
    
    prediction = predict_trainer(trained_model, test_dataset, batch_size=32)
    
    prediction_train = predict_trainer(trained_model, train_dataset, batch_size=32)
    
    labels_test = test_dataset["labels"]
    labels_train = train_dataset["labels"]

    compute_accuracy(prediction, labels_test, "test")
    compute_accuracy(prediction_train, labels_train, "train")
    compute_accuracy(untrainded_model_prediction, labels_test, "untrained")

    plot_confusion_matrix(prediction, labels_test, saving_path=SAVING_PATH)
