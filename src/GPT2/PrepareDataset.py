import os
import sys

# Add the path to the parent directory to augment search for module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from Utils import (
    load_model_and_tokenizer,
    prepare_datasets,
)
from LLM import (
    train_model_trainer,
    predict_trainer,
)

from PlotHelper import plot_confusion_matrix, plot_distribution_of_datasets

from MetricsHelper import (
    compute_accuracy,
    compute_recall,
    compute_f1,
    compute_precision,
    compute_classification_report,
)

# GLOBAL VARIABLES
SAVING_PATH = "../Results-GPT2"
MODEL_PATH = "gpt2"
MODEL_NAME = "GPT2"


if __name__ == "__main__":

    tokenizer, model = load_model_and_tokenizer(MODEL_PATH)

    train_dataset, eval_dataset, test_dataset = prepare_datasets(tokenizer)

    plot_distribution_of_datasets(
        train_dataset, eval_dataset, test_dataset, saving_path=SAVING_PATH
    )

    # untrainded_model_prediction =  predict_trainer(model, test_dataset, batch_size=16)

    trained_model = train_model_trainer(model, train_dataset, eval_dataset=eval_dataset)

    prediction = predict_trainer(trained_model, test_dataset, batch_size=32)

    # prediction_train = predict_trainer(trained_model, train_dataset, batch_size=32)

    labels_test = test_dataset["labels"]
    # labels_train = train_dataset["labels"]

    compute_accuracy(prediction, labels_test, "test", MODEL_NAME)
    compute_recall(prediction, labels_test, "test", MODEL_NAME)
    compute_precision(prediction, labels_test, "test", MODEL_NAME)
    compute_classification_report(prediction, labels_test, "test", MODEL_NAME)

    # compute_accuracy(prediction_train, labels_train, "train",MODEL_NAME)
    # compute_accuracy(untrainded_model_prediction, labels_test, "untrained", MODEL_NAME)

    plot_confusion_matrix(prediction, labels_test, saving_path=SAVING_PATH)