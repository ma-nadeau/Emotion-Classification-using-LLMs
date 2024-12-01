import os
import sys
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch  # type: ignore
from datasets import Dataset  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
import string


# Add the path to the parent directory to augment search for module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from LLM import train_evaluate_hyperparams
from PlotHelper import plot_train_vs_validation_accuracy

from Utils import (
    load_model_and_tokenizer,
    prepare_datasets,
    prepare_multilabel_datasets,
    load_model_and_tokenizer_multilabel,
    load_model_and_tokenizer_with_attention,
    get_single_label_dataset,
    tokenize_dataset,
    oversample_dataset,
    undersample_features,
    remove_label,
    format_datasets_for_pytorch,
    delete_CSV,
)

from LLM import (
    train_model_trainer,
    predict_trainer,
    multilabel_predict_trainer,
    multilabel_train_model_trainer,
)

from PlotHelper import (
    plot_confusion_matrix,
    plot_distribution_of_datasets,
    plot_distribution_of_datasets_binary_vector_labels,
    plot_attention_weights,
    plot_all_attention_weights,
)

from MetricsHelper import (
    compute_accuracy,
    compute_recall,
    compute_f1,
    compute_precision,
    compute_classification_report,
)

# GLOBAL VARIABLES
SAVING_PATH = "../Results-Distilled-GPT2"
MODEL_PATH = "/opt/models/distilgpt2"
MODEL_NAME = "Distilled-GPT2"


def compute_model_accuracy(prediction, labels, model_name):
    accuracy = accuracy_score(labels, prediction)
    print(f"Accuracy {model_name}: {accuracy}")
    return accuracy


def over_and_undersample_dataset(train_dataset):
    # train_ds = undersample_features(train_dataset)
    train_ds = oversample_dataset(train_dataset)
    return train_ds


if __name__ == "__main__":

    """Single Label Classification"""

    # tokenizer, model = load_model_and_tokenizer(MODEL_PATH)

    # train_dataset, eval_dataset, test_dataset = prepare_datasets(tokenizer)

    # plot_distribution_of_datasets(
    #     train_dataset, eval_dataset, test_dataset, saving_path=SAVING_PATH
    # )

    # # untrainded_model_prediction =  predict_trainer(model, test_dataset, batch_size=16)

    # trained_model = train_model_trainer(model, train_dataset, eval_dataset=eval_dataset)

    # prediction = predict_trainer(trained_model, test_dataset, batch_size=32)

    # # prediction_train = predict_trainer(trained_model, train_dataset, batch_size=32)

    # labels_test = test_dataset["labels"]
    # # labels_train = train_dataset["labels"]

    # compute_accuracy(prediction, labels_test, "test", MODEL_NAME)
    # compute_recall(prediction, labels_test, "test", MODEL_NAME)
    # compute_precision(prediction, labels_test, "test", MODEL_NAME)
    # compute_classification_report(prediction, labels_test, "test", MODEL_NAME)
    # # compute_accuracy(prediction_train, labels_train, "train")
    # # compute_accuracy(untrainded_model_prediction, labels_test, "untrained")

    # plot_confusion_matrix(prediction, labels_test, saving_path=SAVING_PATH)

    """ Multilabel Classification """

    tokenizer, model = load_model_and_tokenizer_multilabel(MODEL_PATH)
    train_dataset, eval_dataset, test_dataset = prepare_multilabel_datasets(tokenizer)

    # plot_distribution_of_datasets_binary_vector_labels(
    #     train_dataset, eval_dataset, test_dataset, saving_path=SAVING_PATH
    # )

    trained_model = multilabel_train_model_trainer(
        model, train_dataset, eval_dataset=eval_dataset
    )

    prediction = multilabel_predict_trainer(trained_model, test_dataset, batch_size=8)

    labels_test = test_dataset["labels"]

    accuracy = compute_accuracy(prediction, labels_test, "test", MODEL_NAME)
    recall = compute_recall(prediction, labels_test, "test", MODEL_NAME)
    precision = compute_precision(prediction, labels_test, "test", MODEL_NAME)
    f1 = compute_f1(prediction, labels_test, "test", MODEL_NAME)
    classification_report = compute_classification_report(
        prediction, labels_test, "test", MODEL_NAME
    )

    """ATTENTION"""

    # tokenizer, model = load_model_and_tokenizer_with_attention(MODEL_PATH)

    # train_dataset, eval_dataset, test_dataset = prepare_datasets(tokenizer)
    # labels_test = test_dataset["labels"]

    # trained_model = train_model_trainer(model, train_dataset, eval_dataset=eval_dataset)

    # prediction, attention = predict_trainer(
    #     trained_model, test_dataset, batch_size=32, output_attention=True
    # )

    # document_index = 0
    # input_tokens = tokenizer.convert_ids_to_tokens(
    #     test_dataset["input_ids"][document_index]
    # )

    # # Convert tokens back to the original text
    # original_text = tokenizer.decode(test_dataset["input_ids"][document_index])

    # # Create a directory to save the attention plots
    # # for layer in range(len(attention)):
    # layer = 0
    # for idx in range(len(input_tokens)):
    #     plot_all_attention_weights(
    #         attention,
    #         input_tokens,
    #         token_idx=idx,
    #         saving_path=f"{SAVING_PATH}/Attention-{original_text.replace(" ", "-")}/Layer_{layer}",
    #         layer=layer
    #         )

    """ HYPERPARAMETERS """
    # #delete_CSV(SAVING_PATH)
    # tokenizer, model = load_model_and_tokenizer(MODEL_PATH)

    # train_dataset, eval_dataset, test_dataset = prepare_datasets(tokenizer)

    # # train_dataset = over_and_undersample_dataset(train_dataset)

    # # plot_distribution_of_datasets(
    # #     train_dataset, eval_dataset, test_dataset, saving_path=SAVING_PATH
    # # )

    # batch_sizes = [8, 16, 32, 64]
    # epochs = [0.5, 1, 2, 4]
    # learning_rates = [1e-5, 3e-5, 5e-5, 9e-5]

    # results = train_evaluate_hyperparams(
    #     model,
    #     tokenizer,
    #     train_dataset,
    #     eval_dataset,
    #     test_dataset,
    #     batch_sizes,
    #     epochs,
    #     learning_rates,
    #     MODEL_PATH,
    #     SAVING_PATH,
    # )

    # # Use the same output directory as the Trainer

    # # # Path to the results.csv file
    # # results_file_path = (
    # #     f"{SAVING_PATH}/hyperparameter_results.csv"  # Replace with the actual path
    # # )

    # # # Read the CSV file into a DataFrame
    # # results = pd.read_csv(results_file_path)
    # # print(results.columns)

    # # # Plot Train vs Validation Accuracy for different hyperparameter pairs
    # # plot_train_vs_validation_accuracy(
    # #     results, param_x="Learning Rate", param_y="Batch Size", output_dir=SAVING_PATH
    # # )

    # # plot_train_vs_validation_accuracy(
    # #     results, param_x="Batch Size", param_y="Epochs", output_dir=SAVING_PATH
    # # )

    # # plot_train_vs_validation_accuracy(
    # #     results, param_x="Epochs", param_y="Learning Rate", output_dir=SAVING_PATH
    # # )

    # # untrainded_model_prediction = predict_trainer(model, test_dataset, batch_size=16)

    # # trained_model = train_model_trainer(model, train_dataset, eval_dataset=eval_dataset)

    # # prediction = predict_trainer(trained_model, test_dataset, batch_size=32)

    # # prediction_train = predict_trainer(trained_model, train_dataset, batch_size=32)

    # # labels_test = test_dataset["labels"]
    # # labels_train = train_dataset["labels"]

    # # compute_accuracy(prediction, labels_test, "test",MODEL_NAME )
    # # compute_accuracy(prediction_train, labels_train, "train",MODEL_NAME)
    # # compute_accuracy(untrainded_model_prediction, labels_test, "untrained",MODEL_NAME)
    # #
    # # plot_confusion_matrix(prediction, labels_test, saving_path=SAVING_PATH)
