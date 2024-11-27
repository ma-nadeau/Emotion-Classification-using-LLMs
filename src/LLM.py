# from transformers import Trainer, TrainingArguments  # type: ignore
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AdamW  # type: ignore
from datasets import load_dataset, concatenate_datasets  # type: ignore
import torch  # type: ignore
from tqdm import tqdm  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from transformers import get_scheduler  # type: ignore


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


def train_model(
    model,
    train_dataset,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    learning_rate=5e-5,
):
    """
    Train the model with the given dataset and training arguments.

    Args:
        model (PreTrainedModel): The model to train.
        train_dataset (Dataset): The training dataset.
        stop_threshold (float): The threshold for early stopping based on training loss.

    Returns:
        PreTrainedModel: The trained model.
    """
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=per_device_train_batch_size,
    )

    model = freeze_model_except_last_layer(model)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    num_training_steps = num_train_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,  # Ensure this is an integer
        num_training_steps=num_training_steps,
    )
    
    # Calculate the total number of training steps
    progress_bar = tqdm(range(num_training_steps))
    
    # Set the model to training mode
    model.train()
    # Move the model to the device
    for epoch in range(num_train_epochs):
        total_loss = 0
        # Iterate over the training dataloader
        for batch in train_dataloader:

            # Move the batch to the device
            batch = {key: value.to(model.device) for key, value in batch.items()}

            ### Forward pass, backward pass, and optimization ###
            outputs = model(**batch)  # Forward pass
            loss = outputs.loss  # Compute the loss
            loss.backward()  # Backward pass

            optimizer.step()  # Update the model parameters
            lr_scheduler.step()  # Update the learning rate
            optimizer.zero_grad()  # Reset the gradients

            ### Update the progress bar ###
            total_loss += loss.item()  # Accumulate the total loss
            progress_bar.set_postfix(
                {
                    "loss": loss.item(),
                    "epoch": epoch + 1,
                }
            )
            progress_bar.update(1)  # Update the progress bar

    return model


def predict(model, dataset, batch_size=32):
    """
    Make predictions using the model on the given dataset.

    Args:
        model (PreTrainedModel): The trained model.
        dataset (Dataset): The dataset to make predictions on.
        batch_size (int): The batch size for the DataLoader.

    Returns:
        list: A list of predictions.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model.eval()  # Set the model to evaluation mode
    predictions = []  # Initialize an empty list to store the predictions

    progress_bar = tqdm(
        total=len(dataloader), desc="Predicting"
    )  # Initialize a progress bar

    ### Iterate over the dataloader ###
    for batch in dataloader:
        # Move the batch to the device
        batch = {key: value.to(model.device) for key, value in batch.items()}

        with torch.no_grad():  # Disable gradient tracking
            outputs = model(**batch)  # Forward pass
        logits = outputs.logits  # Get the logits
        # Compute the predicted labels
        batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        predictions.extend(batch_predictions)
        progress_bar.update(1)
    progress_bar.close()
    return predictions


# class CustomTrainingArguments(TrainingArguments):
#     @property
#     def _setup_devices(self):
#         # Override the device setup to avoid using accelerate
#         if torch.cuda.is_available():
#             return torch.device("cuda")
#         else:
#             return torch.device("cpu")


# def train_model_trainer(
#     model,
#     train_dataset,
#     eval_dataset,
#     output_dir="../Results-Distilled-GPT2",
#     num_train_epochs=1,
#     per_device_train_batch_size=32,
#     learning_rate=5e-5,
# ):
#     """
#     Train the model with the given dataset and training arguments using the Trainer API.

#         train_dataset (Dataset): The training dataset.
#         eval_dataset (Dataset): The evaluation dataset.
#         output_dir (str): The directory to save the model and training outputs.
#         model (PreTrainedModel): The model to train.
#         train_dataset (Dataset): The training dataset.
#         output_dir (str): The directory to save the model and training outputs.
#         num_train_epochs (int): The number of training epochs.
#         per_device_train_batch_size (int): The batch size per device during training.
#         learning_rate (float): The learning rate for the optimizer.

#     Returns:
#         PreTrainedModel: The trained model.
#     """

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     training_args = CustomTrainingArguments(
#         output_dir=output_dir,
#         eval_strategy="epoch",
#         num_train_epochs=num_train_epochs,
#         per_device_train_batch_size=per_device_train_batch_size,
#         learning_rate=learning_rate,
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#     )

#     trainer.train()

#     return model


# def predict_trainer(
#     model, dataset, batch_size=32, output_dir="../Results-Distilled-GPT2"
# ):
#     """
#     Make predictions using the model on the given dataset using the Trainer API.

#     Args:
#         model (PreTrainedModel): The trained model.
#         dataset (Dataset): The dataset to make predictions on.
#         batch_size (int): The batch size for the DataLoader.

#     Returns:
#         list: A list of predictions.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     training_args = CustomTrainingArguments(
#         output_dir=output_dir,
#         per_device_eval_batch_size=batch_size,
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#     )

#     predictions = trainer.predict(dataset)
#     return predictions.predictions.argmax(axis=-1)
