from datasets import load_dataset, DatasetDict  # type: ignore
import torch  # type: ignore
from transformers import (AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification)  # type: ignore

if __name__ == "__main__":
    # Load dataset
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")

    # Filter dataset to include only single-label examples
    def filter_single_label(example):
        return len(example["labels"]) == 1


    filtered_ds = DatasetDict({
        split: ds[split].filter(filter_single_label)
        for split in ds if split in ds
    })

    # Initialize GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load GPT-2 model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        "gpt2",
        num_labels=28  # Number of labels in your dataset
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # Tokenize dataset
    def tokenize_function(batch):
        tokenized = tokenizer(
            batch["text"],
            padding="max_length",  # Pad shorter sequences
            truncation=True,  # Truncate longer sequences
            max_length=128,  # Maximum sequence length
            return_tensors="pt",  # Return PyTorch tensors
        )
        tokenized["labels"] = batch["labels"]  # Directly assign the batch of labels
        return tokenized


    tokenized_ds = filtered_ds.map(tokenize_function, batched=True)

    # Define a function to prepare the dataset for PyTorch
    def format_dataset(batch):
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }


    # Prepare the datasets for PyTorch
    tokenized_ds = tokenized_ds.map(format_dataset, batched=True)

    # Set format for PyTorch
    tokenized_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # Split into training and evaluation datasets
    train_dataset = tokenized_ds["train"]
    eval_dataset = tokenized_ds["validation"]
    test_dataset = tokenized_ds["test"]

    print(train_dataset)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./Results-Distilled-GPT2",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=5e-5,
    )
    trainer = Trainer(
        model=model,  # Model to train
        args=training_args,  # Training arguments
        train_dataset=train_dataset,  # Training dataset
        eval_dataset=eval_dataset,  # Evaluation dataset
        tokenizer=tokenizer,  # Tokenizer (optional, for padding and truncation)
    )

    # Train the model
    trainer.train()

    # Make predictions
    test_predictions = trainer.predict(test_dataset)
    print(test_predictions)
    print(trainer.evaluate())
