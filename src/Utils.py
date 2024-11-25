from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, TrainingArguments, AdamW  # type: ignore
from datasets import load_dataset  # type: ignore
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


def define_training_arguments(
    output_dir="../Results-Distilled-GPT2",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
):
    """
    Define the training arguments.

    Args:
        output_dir (str): The output directory for the results.
        num_train_epochs (int): The number of training epochs.
        per_device_train_batch_size (int): The batch size per device during training.
        learning_rate (float): The learning rate.

    Returns:
        dict: A dictionary containing the training arguments.
    """
    return {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "learning_rate": learning_rate,
    }


def initialize_trainer(model, training_args, train_dataset, eval_dataset):
    """
    Initialize the training process.

    Args:
        model (PreTrainedModel): The model to train.
        training_args (dict): The training arguments.
        train_dataset (Dataset): The training dataset.
        eval_dataset (Dataset): The evaluation dataset.

    Returns:
        PreTrainedModel: The trained model.
    """
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=training_args["per_device_train_batch_size"],
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=training_args["per_device_train_batch_size"]
    )

    optimizer = AdamW(model.parameters(), lr=training_args["learning_rate"])
    num_training_steps = training_args["num_train_epochs"] * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for _ in range(training_args["num_train_epochs"]):
        for batch in train_dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    model.eval()
    eval_loss = 0
    for batch in eval_dataloader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        eval_loss += outputs.loss.item()

    avg_eval_loss = eval_loss / len(eval_dataloader)
    print(f"Average evaluation loss: {avg_eval_loss}")

    return model
