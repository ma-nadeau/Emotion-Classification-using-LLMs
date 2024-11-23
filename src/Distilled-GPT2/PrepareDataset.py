import numpy as np  # type: ignore
import torch  # type: ignore
import os
import sys
from datasets import Dataset # type: ignore

# Add the path to the parent directory to augment search for module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from Utils import load_model_and_tokenizer, get_dataset, get_columns_from_dataset, preprocess_dataset



# GLOBAL VARIABLES
SAVING_PATH = "../../Results-Distilled-GPT2"
MODEL_PATH = "/opt/models/distilgpt2"


if __name__ == "__main__":

    # Load the model and tokenizer
    tokenizer, model = load_model_and_tokenizer(MODEL_PATH)

    # Load the dataset
    ds_train, ds_validation, ds_test = get_dataset()

    # Get the columns from the dataset
    ds_train_text, ds_train_labels, ds_train_id = get_columns_from_dataset(ds_train)
    ds_validation_text, ds_validation_labels, ds_validation_id = get_columns_from_dataset(ds_validation)
    ds_test_text, ds_test_labels, ds_test_id = get_columns_from_dataset(ds_test)

