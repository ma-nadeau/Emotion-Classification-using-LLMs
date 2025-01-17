import os
import sys
import pandas as pd  # type: ignore

# Add the path to the parent directory to augment search for module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from Utils import (
    load_model_and_tokenizer,
    prepare_datasets,
    load_model_and_tokenizer_with_attention,
    delete_CSV,
    load_model_and_tokenizer_multilabel,
    prepare_multilabel_datasets,
)
from LLM import (
    train_model_trainer,
    predict_trainer,
    train_evaluate_hyperparams,
    multilabel_train_model_trainer,
    multilabel_predict_trainer,
)

from PlotHelper import (
    plot_confusion_matrix,
    plot_distribution_of_datasets,
    plot_all_attention_weights,
    plot_train_vs_validation_accuracy,
)

from MetricsHelper import (
    compute_accuracy,
    compute_recall,
    compute_f1,
    compute_precision,
    compute_classification_report,
)

# GLOBAL VARIABLES
SAVING_PATH = "../Results-BERT"
MODEL_PATH = "/opt/models/bert-base-uncased"
MODEL_NAME = "BERT"

if __name__ == "__main__":
    
    """Single Label Classification"""

    tokenizer, model = load_model_and_tokenizer(MODEL_PATH)

    train_dataset, eval_dataset, test_dataset = prepare_datasets(tokenizer)

    plot_distribution_of_datasets(
        train_dataset, eval_dataset, test_dataset, saving_path=SAVING_PATH
    )

    # untrainded_model_prediction =  predict_trainer(model, test_dataset, batch_size=16)

    trained_model = train_model_trainer(model, train_dataset, eval_dataset=eval_dataset)

    prediction = predict_trainer(trained_model, test_dataset, batch_size=32)

    # prediction_train = predict_trainer(trained_model, train_dataset, batch_size=32)

    labels_test = test_dataset["labels"]
    # labels_train = train_dataset["labels"]

    compute_accuracy(prediction, labels_test, "test", MODEL_NAME)
    compute_recall(prediction, labels_test, "test", MODEL_NAME)
    compute_precision(prediction, labels_test, "test", MODEL_NAME)
    compute_classification_report(prediction, labels_test, "test", MODEL_NAME)
    # compute_accuracy(prediction_train, labels_train, "train")
    # compute_accuracy(untrainded_model_prediction, labels_test, "untrained")

    plot_confusion_matrix(prediction, labels_test, saving_path=SAVING_PATH)
     
    """ATTENTION"""

    tokenizer, model = load_model_and_tokenizer_with_attention(MODEL_PATH)

    train_dataset, eval_dataset, test_dataset = prepare_datasets(tokenizer)
    labels_test = test_dataset["labels"]

    trained_model = train_model_trainer(model, train_dataset, eval_dataset=eval_dataset)

    prediction, attention = predict_trainer(
        trained_model, test_dataset, batch_size=32, output_attention=True
    )

    # document_index = 0

    # correct 
    correct_indices = [
        i for i in range(len(prediction)) if prediction[i] == labels_test[i]
    ]
    incorrect_indices = [
        i for i in range(len(prediction)) if prediction[i] != labels_test[i]
    ]
    # correct 
    layers = [0,5,10]
    for layer in layers:
        for document_index in correct_indices[:1]:
            # Convert tokens back to the original text
            original_text = tokenizer.decode(test_dataset["input_ids"][document_index])
            input_tokens = tokenizer.convert_ids_to_tokens(test_dataset["input_ids"][document_index] )
            # # Create a directory to save the attention plots
            # for layer in range(len(attention)):
            for idx in range(len(input_tokens)):
                plot_all_attention_weights(
                    attention,
                    input_tokens,
                    token_idx=idx,
                    saving_path=f"{SAVING_PATH}/Attention/Correct/{original_text.replace(' ', '-')}/Layer_{layer}",
                    layer=layer
                    )
    for layer in layers:
        for document_index in incorrect_indices[0:1]:
            # Convert tokens back to the original text
            original_text = tokenizer.decode(test_dataset["input_ids"][document_index])
            input_tokens = tokenizer.convert_ids_to_tokens(test_dataset["input_ids"][document_index] )
            # # Create a directory to save the attention plots
            # for layer in range(len(attention)):
            for idx in range(len(input_tokens)):
                plot_all_attention_weights(
                    attention,
                    input_tokens,
                    token_idx=idx,
                    saving_path=f"{SAVING_PATH}/Attention/Incorrect/{original_text.replace(' ', '-')}/Layer_{layer}",
                    layer=layer
                    )


    """ Multilabel Classification """

    tokenizer, model = load_model_and_tokenizer_multilabel(MODEL_PATH)
    train_dataset, eval_dataset, test_dataset = prepare_multilabel_datasets(tokenizer)

    # plot_distribution_of_datasets_binary_vector_labels(
    #     train_dataset, eval_dataset, test_dataset, saving_path=SAVING_PATH
    # )

    trained_model = multilabel_train_model_trainer(
        model, train_dataset, eval_dataset=eval_dataset
    )

    prediction = multilabel_predict_trainer(trained_model, test_dataset, batch_size=8)

    labels_test = test_dataset["labels"]

    accuracy = compute_accuracy(prediction, labels_test, "test", MODEL_NAME)
    recall = compute_recall(prediction, labels_test, "test", MODEL_NAME)
    precision = compute_precision(prediction, labels_test, "test", MODEL_NAME)
    f1 = compute_f1(prediction, labels_test, "test", MODEL_NAME)
    classification_report = compute_classification_report(
        prediction, labels_test, "test", MODEL_NAME
    )
    
    """HYPERPARAMETERS"""
    
    # delete_CSV(SAVING_PATH)
    tokenizer, model = load_model_and_tokenizer(MODEL_PATH)

    train_dataset, eval_dataset, test_dataset = prepare_datasets(tokenizer)

    # train_dataset = over_and_undersample_dataset(train_dataset)

    # plot_distribution_of_datasets(
    #     train_dataset, eval_dataset, test_dataset, saving_path=SAVING_PATH
    # )

    batch_sizes = [8, 16, 32, 64]
    epochs = [0.5, 1, 2, 4]
    learning_rates = [1e-5, 3e-5, 5e-5, 9e-5]

    results = train_evaluate_hyperparams(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        test_dataset,
        batch_sizes,
        epochs,
        learning_rates,
        MODEL_PATH,
        SAVING_PATH,
    )

    results_file_path = (
        f"{SAVING_PATH}/hyperparam_results.csv"  # Replace with the actual path
    )

    # Read the CSV file into a DataFrame
    results = pd.read_csv(results_file_path)
    print(results.columns)

    # Plot Train vs Validation Accuracy for different hyperparameter pairs
    plot_train_vs_validation_accuracy(
        results, param_x="Learning Rate", param_y="Batch Size", output_dir=SAVING_PATH
    )

    plot_train_vs_validation_accuracy(
        results, param_x="Batch Size", param_y="Epochs", output_dir=SAVING_PATH
    )

    plot_train_vs_validation_accuracy(
        results, param_x="Epochs", param_y="Learning Rate", output_dir=SAVING_PATH
    )