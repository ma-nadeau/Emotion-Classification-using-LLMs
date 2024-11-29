from transformers import Trainer, TrainingArguments  # type: ignore
import torch # type: ignore
import torch.nn.functional as F # type: ignore

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
    per_device_train_batch_size=16,
    learning_rate=1e-5,
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


def predict_trainer(model, dataset, batch_size=16, output_dir="./output"):
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
    return predictions.predictions.argmax(axis=-1)

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs):
     
        outputs = model(**inputs)
        print("Outputs:", outputs)
        
        logits = outputs.logits
        print("Logits", logits)
        
        print("Labels:", inputs["labels"])
        labels = inputs["labels"].float()
        print("Labels:", labels)

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss

def multilabel_train_model_trainer(
    model,
    train_dataset,
    eval_dataset,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=1e-5,
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

    trainer = CustomTrainer(
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

    trainer = Trainer(
        model=model,
        args=training_args,
    )

    predictions = trainer.predict(dataset)
    return (predictions.predictions > 0.5).astype(int)