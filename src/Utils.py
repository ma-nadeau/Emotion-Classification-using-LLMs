# from transformers import Trainer, TrainingArguments  # type: ignore
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AdamW  # type: ignore
from datasets import load_dataset, concatenate_datasets  # type: ignore
import torch  # type: ignore
from tqdm import tqdm  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from transformers import get_scheduler  # type: ignore


def get_go_emotions_dataset():
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


def get_single_label_dataset():
    """
    Load the GoEmotions dataset and filter to only include examples with a single label.

    Returns:
        tuple: A tuple containing the filtered training, validation, and test datasets.
    """
    # Filter the dataset to only include examples with a single label
    filter_single_label = lambda example: len(example["labels"]) == 1

    # Load the dataset
    ds_train, ds_validation, ds_test = get_go_emotions_dataset()

    # Filter the dataset
    ds_train = ds_train.filter(filter_single_label)
    ds_validation = ds_validation.filter(filter_single_label)
    ds_test = ds_test.filter(filter_single_label)

    #ds_train = undersample_features(ds_train)
    ds_train = oversample_dataset(ds_train)
    return ds_train, ds_validation, ds_test



def undersample_features(dataset, num_samples=2000, label=27):
    """
    Undersample the dataset to include a maximum number of samples for a specific label.

    Args:
        dataset (Dataset): The dataset to undersample.
        num_samples (int): The maximum number of samples to include for the specified label.
        label (int): The label to undersample.

    Returns:
        Dataset: The undersampled dataset.
    """
    # Filter the dataset to include only the specified label
    label_dataset = dataset.filter(lambda example: label in example["labels"])

    # Randomly select the specified number of samples
    label_dataset = label_dataset.shuffle(seed=42).select(
        range(min(num_samples, len(label_dataset)))
    )

    # Filter the dataset to exclude the specified label
    non_label_dataset = dataset.filter(lambda example: label not in example["labels"])

    # Concatenate the undersampled label dataset with the non-label dataset
    undersampled_dataset = concatenate_datasets([non_label_dataset, label_dataset])

    return undersampled_dataset


def load_model_and_tokenizer(model_path: str) -> tuple:
    """
    Load the tokenizer and model from a given path.

    Args:
        model_path (str): The path to the model directory.

    Returns:
        tuple: A tuple containing the tokenizer and model objects.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, clean_up_tokenization_spaces=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=28, pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer, model


def tokenize_dataset(dataset, tokenizer):
    """
    Tokenize the dataset using the provided tokenizer.

    Args:
        dataset (Dataset): The dataset to tokenize.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.

    Returns:
        Dataset: The tokenized dataset.
    """

    def tokenize_function(batch):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenized = tokenizer(
            batch["text"],
            padding="max_length",  # Pad shorter sequences
            truncation=True,  # Truncate longer sequences
            max_length=128,  # Maximum sequence length
            return_tensors="pt",  # Return PyTorch tensors
        )
        tokenized["labels"] = batch["labels"]  # Directly assign the batch of labels
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


def format_dataset_for_pytorch(dataset):
    """
    Format the dataset for PyTorch.

    Args:
        dataset (Dataset): The dataset to format.

    Returns:
        Dataset: The formatted dataset.
    """

    format_function = lambda batch: {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
    }

    formatted_dataset = dataset.map(format_function, batched=True)
    formatted_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    return formatted_dataset


def format_datasets_for_pytorch(tokenized_train, tokenized_validation, tokenized_test):
    """
    Format the tokenized datasets for PyTorch.

    Args:
        tokenized_train (Dataset): The tokenized training dataset.
        tokenized_validation (Dataset): The tokenized validation dataset.
        tokenized_test (Dataset): The tokenized test dataset.

    Returns:
        tuple: A tuple containing the formatted training, validation, and test datasets.
    """
    train_dataset = format_dataset_for_pytorch(tokenized_train)
    eval_dataset = format_dataset_for_pytorch(tokenized_validation)
    test_dataset = format_dataset_for_pytorch(tokenized_test)
    return train_dataset, eval_dataset, test_dataset