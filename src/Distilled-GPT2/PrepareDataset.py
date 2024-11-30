# import numpy as np  # type: ignore
# import torch  # type: ignore
# import os
# import sys
# from datasets import Dataset  # type: ignore
# from sklearn.metrics import accuracy_score  # type: ignore
# import string
#
#
# # Add the path to the parent directory to augment search for module
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
#
# from LLM import train_evaluate_hyperparams
# from PlotHelper import plot_train_vs_validation_accuracy
#
# from Utils import (
#     load_model_and_tokenizer,
#     get_single_label_dataset,
#     tokenize_dataset,
#     oversample_dataset,
#     undersample_features,
#     remove_label,
#     format_datasets_for_pytorch,
# )
#
# from LLM import (
#     train_model_trainer,
#     predict_trainer,
# )
#
# from PlotHelper import plot_confusion_matrix, plot_distribution_of_datasets
#
# # GLOBAL VARIABLES
# SAVING_PATH = "../../Results-Distilled-GPT2"
# MODEL_PATH = "/opt/models/distilgpt2"
#
#
# def prepare_datasets(tokenizer):
#     """
#     Prepare the datasets for training and evaluation.
#
#     Returns:
#         tuple: A tuple containing the training, evaluation, and test datasets.
#     """
#     ds_train, ds_validation, ds_test = get_single_label_dataset()
#
#     tokenized_train = tokenize_dataset(ds_train, tokenizer)
#     tokenized_validation = tokenize_dataset(ds_validation, tokenizer)
#     tokenized_test = tokenize_dataset(ds_test, tokenizer)
#
#     train_dataset, eval_dataset, test_dataset = format_datasets_for_pytorch(
#         tokenized_train, tokenized_validation, tokenized_test
#     )
#     return train_dataset, eval_dataset, test_dataset
#
#
# def compute_accuracy(prediction, labels, model_name):
#     accuracy = accuracy_score(labels, prediction)
#     print(f"Accuracy {model_name}: {accuracy}")
#     return accuracy


# if __name__ == "__main__":
#     tokenizer, model = load_model_and_tokenizer(MODEL_PATH)
#
#     train_dataset, eval_dataset, test_dataset = prepare_datasets(tokenizer)
#
#     # train_dataset = undersample_features(train_dataset)
#     # train_dataset = oversample_dataset(train_dataset)
#
#     print(len(test_dataset["labels"]))
#
#     # plot_distribution_of_datasets(
#     #     train_dataset, eval_dataset, test_dataset, saving_path=SAVING_PATH
#     # )
#
#     # untrainded_model_prediction = predict_trainer(model, test_dataset, batch_size=16)
#     #
#     # trained_model = train_model_trainer(model, train_dataset, eval_dataset=eval_dataset)
#     #
#     # prediction = predict_trainer(trained_model, test_dataset, batch_size=32)
#     #
#     # prediction_train = predict_trainer(trained_model, train_dataset, batch_size=32)
#     #
#     # labels_test = test_dataset["labels"]
#     # labels_train = train_dataset["labels"]
#     #
#     # compute_accuracy(prediction, labels_test, "test")
#     # compute_accuracy(prediction_train, labels_train, "train")
#     # compute_accuracy(untrainded_model_prediction, labels_test, "untrained")
#     #
#     # plot_confusion_matrix(prediction, labels_test, saving_path=SAVING_PATH)
#
#     batch_sizes = [8, 16, 32, 64, 128]
#     epochs = [3, 5, 8, 10]
#     learning_rates = [1e-5, 2e-5, 4e-5, 5e-5, 9e-5]
#
#     results = train_evaluate_hyperparams(
#         model,
#         tokenizer,
#         train_dataset,
#         eval_dataset,
#         test_dataset,
#         batch_sizes,
#         epochs,
#         learning_rates,
#     )
#
#     # Use the same output directory as the Trainer
#     output_dir = "./output"
#
#     # Plot Train vs Validation Accuracy for different hyperparameter pairs
#     plot_train_vs_validation_accuracy(
#         results, param_x="learning_rate", param_y="batch_size", output_dir=output_dir
#     )
#
#     plot_train_vs_validation_accuracy(
#         results, param_x="batch_size", param_y="epochs", output_dir=output_dir
#     )
#
#     plot_train_vs_validation_accuracy(
#         results, param_x="epochs", param_y="learning_rate", output_dir=output_dir
#     )


import numpy as np
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import classification_report  # type: ignore
from wikipedia2vec import Wikipedia2Vec  # type: ignore
from nltk.tokenize import word_tokenize
from src.Utils import get_single_label_dataset


def text_to_embedding(text, model):
    """
    Convert text into an embedding using Wikipedia2Vec.

    Args:
        text (str): The input text.
        model (Wikipedia2Vec): The Wikipedia2Vec model.

    Returns:
        np.ndarray: The average word embedding for the text.
    """
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    embeddings = [
        model.get_word_vector(word) for word in tokens if model.get_word_vector(word) is not None
    ]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.vector_size)


def extract_single_label(example):
    """
    Extract a single label from the dataset.
    Assumes each example has exactly one label.

    Args:
        example (dict): A single dataset example.

    Returns:
        dict: Modified example with a single label.
    """
    example["label"] = example["labels"][0]  # Extract the single label
    return example


def embed_text(example, wiki2vec):
    """
    Embed the text of an example using Wikipedia2Vec.

    Args:
        example (dict): A single dataset example.
        wiki2vec (Wikipedia2Vec): Pretrained Wikipedia2Vec model.

    Returns:
        dict: Modified example with the embedded text.
    """
    embedding = text_to_embedding(example["text"], wiki2vec)
    example["embedding"] = embedding.tolist()  # Convert numpy array to list for compatibility
    return example


def main():
    # Load the pretrained Wikipedia2Vec model
    wiki2vec = Wikipedia2Vec.load("path/to/wikipedia2vec/model/file")

    # Load the single-label GoEmotions dataset
    ds_train, ds_validation, ds_test = get_single_label_dataset()

    # Process the datasets
    ds_train = ds_train.map(extract_single_label)
    ds_validation = ds_validation.map(extract_single_label)
    ds_test = ds_test.map(extract_single_label)

    ds_train = ds_train.map(lambda example: embed_text(example, wiki2vec))
    ds_validation = ds_validation.map(lambda example: embed_text(example, wiki2vec))
    ds_test = ds_test.map(lambda example: embed_text(example, wiki2vec))

    # Convert to NumPy arrays for ML model training
    X_train = np.array(ds_train["embedding"])
    y_train = np.array(ds_train["label"])

    X_validation = np.array(ds_validation["embedding"])
    y_validation = np.array(ds_validation["label"])

    X_test = np.array(ds_test["embedding"])
    y_test = np.array(ds_test["label"])

    # Train a Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Evaluate on the validation set
    y_pred = rf_classifier.predict(X_validation)
    print("Validation Classification Report:")
    print(classification_report(y_validation, y_pred))

    # Test the model on the test set
    y_pred_test = rf_classifier.predict(X_test)
    print("Test Classification Report:")
    print(classification_report(y_test, y_pred_test))


if __name__ == "__main__":
    main()
