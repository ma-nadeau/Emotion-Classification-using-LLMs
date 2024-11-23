from transformers import AutoTokenizer, AutoModel  # type: ignore
from datasets import load_dataset  # type: ignore
import torch  # type: ignore


def get_dataset():
    """
    Load the GoEmotions dataset.

    Returns:
        tuple: A tuple containing the training, validation, and test datasets.
    """
    ds_train = load_dataset(
        "google-research-datasets/go_emotions", "simplified", split="train"
    )
    ds_validation = load_dataset(
        "google-research-datasets/go_emotions", "simplified", split="validation"
    )
    ds_test = load_dataset(
        "google-research-datasets/go_emotions", "simplified", split="test"
    )

    return ds_train, ds_validation, ds_test


def load_model_and_tokenizer(model_path: str) -> tuple:
    """
    Load the tokenizer and model from a given path.

    Args:
        model_path (str): The path to the model directory.

    Returns:
        tuple: A tuple containing the tokenizer and model objects.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, clean_up_tokenization_spaces=True
    )
    model = AutoModel.from_pretrained(model_path)
    return tokenizer, model

def get_columns_from_dataset(dataset):
    """
    Get the columns from the dataset.

    Args:
        dataset (Dataset): The dataset.

    Returns:
        list: A list of columns in the dataset.
    """
    column_text = dataset["text"]
    column_labels = dataset["labels"]
    columns_id = dataset["id"]
    return column_text, column_labels, columns_id


# TODO: write this function
def preprocess_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset