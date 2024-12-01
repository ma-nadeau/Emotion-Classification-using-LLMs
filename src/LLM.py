import csv

from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification # type: ignore
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
import numpy as np  # type: ignore
from CustomTrainer import CustomTrainerForMultilabelClassification
from tqdm import tqdm  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from transformers import get_scheduler  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from itertools import product  # type: ignore
import json  # type: ignore



def freeze_model_except_last_layer(model):
    # Freeze all layers except for the last one
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
    fineTuneLastLayerOnly=False,
):
    """
    Train the model with the given dataset and training arguments.

        model (PreTrainedModel): The model to train.
        train_dataset (Dataset): The training dataset.
        eval_dataset (Dataset): The evaluation dataset.
        output_dir (str): The directory to save the model and training outputs.
        num_train_epochs (int): The number of training epochs.
        per_device_train_batch_size (int): The batch size per device during training.
        learning_rate (float): The learning rate for the optimizer.
        fineTuneLastLayerOnly (bool): Whether to fine-tune only the last layer.

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

    if fineTuneLastLayerOnly:
        model = freeze_model_except_last_layer(model)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    return model


def predict_trainer(
    model, dataset, batch_size=16, output_dir="./output", output_attention=False
):
    """
    Make predictions using the model on the given dataset

    Args:
        model (PreTrainedModel): The trained model.
        dataset (Dataset): The dataset to make predictions on.
        batch_size (int): The batch size for the DataLoader.

    Returns:
        list: A list of predictions.
    """

    training_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="no",
        per_device_eval_batch_size=batch_size,
        logging_dir=None,
    )

    model = model.eval()

    trainer = Trainer(
        model=model,
        args=training_args,
    )

    predictions = trainer.predict(dataset)
    
    return (
        (
            predictions.predictions[0].argmax(axis=-1),
            predictions.predictions[-1],
        )
        if output_attention
        else predictions.predictions.argmax(axis=-1)
    )


### Multilabel Classification ###

def multilabel_train_model_trainer(
    model,
    train_dataset,
    eval_dataset,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    learning_rate=9e-5,
    fineTuneLastLayerOnly=False,
):
    """
    Train the model with the given dataset and training arguments

        model (PreTrainedModel): The model to train.
        train_dataset (Dataset): The training dataset.
        eval_dataset (Dataset): The evaluation dataset.
        output_dir (str): The directory to save the model and training outputs.
        num_train_epochs (int): The number of training epochs.
        per_device_train_batch_size (int): The batch size per device during training.
        learning_rate (float): The learning rate for the optimizer.
        fineTuneLastLayerOnly (bool): Whether to fine-tune only the last layer.

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

    if fineTuneLastLayerOnly:
        model = freeze_model_except_last_layer(model)

    trainer = CustomTrainerForMultilabelClassification(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    return model


def multilabel_predict_trainer(model, dataset, batch_size=16, output_dir="./output"):
    """
    Make predictions using the model on the given dataset

    Args:
        model (PreTrainedModel): The trained model.
        dataset (Dataset): The dataset to make predictions on.
        batch_size (int): The batch size for the DataLoader.

    Returns:
        list: A list of predictions.
    """

    training_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="no",
        per_device_eval_batch_size=batch_size,
        logging_dir=None,
    )

    model = model.eval()

    trainer = CustomTrainerForMultilabelClassification(
        model=model,
        args=training_args,
    )

    predictions = trainer.predict(dataset)
    thresholded_predictions = (
        torch.sigmoid(torch.tensor(predictions.predictions)) > 0.5
    ).int()
    return thresholded_predictions


  
### Hyperparameters ###

def train_evaluate_hyperparams(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        test_dataset,
        batch_sizes,
        epochs,
        learning_rates,
        model_path,
        output_folder,
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

    with open(f"{output_folder}/hyperparam_results.csv", "a", newline="") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(["Batch Size", "Epochs", "Learning Rate", "Train Accuracy", "Val Accuracy"])

        for batch_size, epoch, lr in product(batch_sizes, epochs, learning_rates):
            print(f"Training with Batch Size: {batch_size}, Epochs: {epoch}, LR: {lr}")
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                       num_labels=28, pad_token_id=tokenizer.pad_token_id)

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
            eval_predictions = trainer.predict(eval_dataset)

            train_accuracy = accuracy_score(
                train_dataset["labels"], train_predictions.predictions.argmax(axis=-1)
            )
            val_accuracy = accuracy_score(
                eval_dataset["labels"], eval_predictions.predictions.argmax(axis=-1)
            )

            # Write result as a row in CSV
            writer.writerow([batch_size, epoch, lr, train_accuracy, val_accuracy])
            f.flush()

    return results
