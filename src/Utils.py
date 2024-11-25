from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification,  Trainer, TrainingArguments  # type: ignore
from datasets import load_dataset  # type: ignore
import torch  # type: ignore


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

    # model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=28, pad_token_id=tokenizer.pad_token_id)
    model = AutoModel.from_pretrained(model_path)
    return tokenizer, model


def load_model_and_tokenizer(
    model_path="/opt/models/distilgpt2", num_labels=28
) -> tuple:
    """
    Load the GPT-2 model for sequence classification and its tokenizer.

    Returns:
        tuple: A tuple containing the tokenizer and model objects.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
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
    train_dataset = format_dataset_for_pytorch(tokenized_train)
    eval_dataset = format_dataset_for_pytorch(tokenized_validation)
    test_dataset = format_dataset_for_pytorch(tokenized_test)
    return train_dataset, eval_dataset, test_dataset

def prepare_datasets_for_training(tokenized_ds):
    """
    Prepare the datasets for PyTorch and split into training, evaluation, and test datasets.

    Args:
        tokenized_ds (DatasetDict): The tokenized dataset.

    Returns:
        tuple: A tuple containing the training, evaluation, and test datasets.
    """
    tokenized_ds = tokenized_ds.map(format_dataset_for_pytorch, batched=True)
    train_dataset = tokenized_ds["train"]
    eval_dataset = tokenized_ds["validation"]
    test_dataset = tokenized_ds["test"]
    return train_dataset, eval_dataset, test_dataset


def define_training_arguments(output_dir="../Results-Distilled-GPT2", num_train_epochs=3, per_device_train_batch_size=8, learning_rate=5e-5):
    """
    Define the training arguments.

    Args:
        output_dir (str): The output directory for the results.
        num_train_epochs (int): The number of training epochs.
        per_device_train_batch_size (int): The batch size per device during training.
        learning_rate (float): The learning rate.

    Returns:
        TrainingArguments: The training arguments.
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
    )


def initialize_trainer(model, training_args, train_dataset, eval_dataset, tokenizer):
    """
    Initialize the Trainer.

    Args:
        model (PreTrainedModel): The model to train.
        training_args (TrainingArguments): The training arguments.
        train_dataset (Dataset): The training dataset.
        eval_dataset (Dataset): The evaluation dataset.
        tokenizer (PreTrainedTokenizer): The tokenizer used for tokenization.

    Returns:
        Trainer: The initialized Trainer.
    """
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )


def train_and_evaluate_model(trainer, test_dataset):
    """
    Train the model and evaluate it on the test dataset.

    Args:
        trainer (Trainer): The Trainer object.
        test_dataset (Dataset): The test dataset.

    Returns:
        Trainer: The trained model.
    """
    trainer.train()
    test_predictions = trainer.predict(test_dataset)
    print(test_predictions)
    print(trainer.evaluate())
    return trainer


def prepare_and_train_model(tokenized_ds, model, tokenizer):
    """
    Prepare the datasets for PyTorch, define training arguments, and train the model.

    Args:
        tokenized_ds (DatasetDict): The tokenized dataset.
        model (PreTrainedModel): The model to train.
        tokenizer (PreTrainedTokenizer): The tokenizer used for tokenization.

    Returns:
        Trainer: The trained model.
    """
    train_dataset, eval_dataset, test_dataset = prepare_datasets_for_training(
        tokenized_ds
    )
    training_args = define_training_arguments()
    trainer = initialize_trainer(
        model, training_args, train_dataset, eval_dataset, tokenizer
    )
    return train_and_evaluate_model(trainer, test_dataset)