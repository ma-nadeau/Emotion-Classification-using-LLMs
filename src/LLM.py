import csv

from transformers import Trainer, TrainingArguments  # type: ignore
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AdamW  # type: ignore
from datasets import load_dataset, concatenate_datasets  # type: ignore
import torch  # type: ignore
from tqdm import tqdm  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from transformers import get_scheduler  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from itertools import product  # type: ignore
import json  # type: ignore


def freeze_model_except_last_layer(model):
    """
    Freeze all layers of the model except the last layer.

    Args:
        model (PreTrainedModel): The model to freeze.

    Returns:
        PreTrainedModel: The model with all layers frozen except the last layer.
    """
    for param in model.base_model.parameters():
        param.requires_grad = False
    return model


def train_model_trainer(
        model,
        train_dataset,
        eval_dataset,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=5e-5,
):
    """
    Train the model with the given dataset and training arguments using the Trainer API.

        train_dataset (Dataset): The training dataset.
        eval_dataset (Dataset): The evaluation dataset.
        output_dir (str): The directory to save the model and training outputs.
        model (PreTrainedModel): The model to train.
        train_dataset (Dataset): The training dataset.
        output_dir (str): The directory to save the model and training outputs.
        num_train_epochs (int): The number of training epochs.
        per_device_train_batch_size (int): The batch size per device during training.
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        PreTrainedModel: The trained model.
    """

    training_args = TrainingArguments(
        eval_strategy="epoch",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        output_dir="./output",
        save_strategy="no",  # Disable saving checkpoints
        logging_dir=None,  # Disable logging

    )

    # model = freeze_model_except_last_layer(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    return model


def predict_trainer(
        model, dataset, batch_size=16, output_dir="./output"
):
    """
    Make predictions using the model on the given dataset using the Trainer API.

    Args:
        model (PreTrainedModel): The trained model.
        dataset (Dataset): The dataset to make predictions on.
        batch_size (int): The batch size for the DataLoader.

    Returns:
        list: A list of predictions.
    """

    training_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="no",  # Disable saving checkpoints
        per_device_eval_batch_size=batch_size,
        logging_dir=None,
    )

    model = model.eval()

    trainer = Trainer(
        model=model,
        args=training_args,
    )

    predictions = trainer.predict(dataset)
    print(predictions)
    return predictions.predictions.argmax(axis=-1)


def train_evaluate_hyperparams(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        test_dataset,
        batch_sizes,
        epochs,
        learning_rates,
):
    """
    Train and evaluate the model for different hyperparameters.

    Args:
        model: The model to train.
        tokenizer: The tokenizer.
        train_dataset: The training dataset.
        eval_dataset: The evaluation dataset.
        test_dataset: The test dataset.
        batch_sizes: List of batch sizes to test.
        epochs: List of epoch counts to test.
        learning_rates: List of learning rates to test.

    Returns:
        list: A list of dictionaries containing train and validation accuracies and hyperparameters.
    """
    # Initialize the results list
    results = []

    with open("tt.csv", "a", newline="") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(["Batch Size", "Epochs", "Learning Rate", "Train Accuracy", "Val Accuracy"])

        for batch_size, epoch, lr in product(batch_sizes, epochs, learning_rates):
            print(f"Training with Batch Size: {batch_size}, Epochs: {epoch}, LR: {lr}")
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForSequenceClassification.from_pretrained("/opt/models/distilgpt2",
                                                                       num_labels=27, pad_token_id=tokenizer.pad_token_id)

            training_args = TrainingArguments(
                eval_strategy="epoch",
                num_train_epochs=epoch,
                per_device_train_batch_size=batch_size,
                learning_rate=lr,
                output_dir="./output",
                save_strategy="no",
                logging_dir=None,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )

            trainer.train()

            train_predictions = trainer.predict(train_dataset)
            test_predictions = trainer.predict(test_dataset)

            train_accuracy = accuracy_score(
                train_dataset["labels"], train_predictions.predictions.argmax(axis=-1)
            )
            test_accuracy = accuracy_score(
                test_dataset["labels"], test_predictions.predictions.argmax(axis=-1)
            )
            print(test_accuracy)

            # Write result as a row in CSV
            writer.writerow([batch_size, epoch, lr, train_accuracy, test_accuracy])
            f.flush()

    return results
