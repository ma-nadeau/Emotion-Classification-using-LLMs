import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.metrics import confusion_matrix  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import os

def plot_distribution_of_datasets(train_dataset, eval_dataset, test_dataset, saving_path="../Results-Distilled-GPT2"):
    """
    Plot the distribution of the datasets.

    Args:
        train_dataset (Dataset): The training dataset.
        eval_dataset (Dataset): The evaluation dataset.
        test_dataset (Dataset): The test dataset.
        saving_path (str): The path where the plot will be saved.
    """
    plt.figure(figsize=(12, 8))
    train_dataset_labels = train_dataset["labels"]
    eval_dataset_labels = eval_dataset["labels"]
    test_dataset_labels = test_dataset["labels"]
    
    plt.hist(train_dataset_labels, bins=28, alpha=0.3, label="Train", color="skyblue")
    plt.hist(eval_dataset_labels, bins=28, alpha=0.3, label="Eval", color="lightgreen")
    plt.hist(test_dataset_labels, bins=28, alpha=0.3, label="Test", color="lightcoral")
    

    plt.xticks(range(28))
    plt.xlabel("Classes", fontsize=12, weight="bold")
    plt.ylabel("Frequency", fontsize=12, weight="bold")
    plt.title("Dataset Distribution", fontsize=14, weight="bold")
    plt.legend(title="Dataset", title_fontsize='13', fontsize='10')
    os.makedirs(saving_path, exist_ok=True)
    plt.savefig(f"{saving_path}/dataset_distribution.png", bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(predictions, labels, saving_path="../Results-Distilled-GPT2"):
    """
    Plot the confusion matrix given the predictions and true labels.

    Args:
        predictions (list or np.array): The predicted labels.
        labels (list or np.array): The true labels.
    """
    sns.set_theme()
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(24, 20))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="coolwarm", 
        cbar=True,
        annot_kws={"size": 10},
        linewidths=0.5,
    )
    plt.xlabel("Predicted Labels", fontsize=12, weight="bold")
    plt.ylabel("True Labels", fontsize=12, weight="bold")
    plt.title("Confusion Matrix", fontsize=14, weight="bold")
    plt.xticks(fontsize=10, weight="bold")
    plt.yticks(fontsize=10, weight="bold")

    os.makedirs(saving_path, exist_ok=True)
    plt.savefig(f"{saving_path}/confusion_matrix.png", bbox_inches="tight")
    plt.close()

def plot_train_vs_validation_accuracy(results, param_x, param_y, output_dir="./output"):
    """
    Save train and validation accuracy plots against two hyperparameters to the Trainer's output directory.

    Args:
        results: List of dictionaries containing train and validation accuracies.
        param_x: The x-axis hyperparameter (e.g., batch_size).
        param_y: The y-axis hyperparameter (e.g., learning_rate).
        output_dir: The Trainer's output directory to save the plots.
    """
    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Create pivot tables for train and validation accuracy
    train_pivot = df.pivot(index=param_x, columns=param_y, values="train_accuracy")
    val_pivot = df.pivot(index=param_x, columns=param_y, values="val_accuracy")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save Train Accuracy Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(train_pivot, cmap="Blues", aspect="auto", origin="lower")
    plt.colorbar(label="Train Accuracy")
    plt.xticks(range(len(train_pivot.columns)), train_pivot.columns, rotation=45)
    plt.yticks(range(len(train_pivot.index)), train_pivot.index)
    plt.xlabel(param_y)
    plt.ylabel(param_x)
    plt.title(f"Train Accuracy by {param_x} and {param_y}")
    train_plot_path = os.path.join(output_dir, f"train_accuracy_{param_x}_vs_{param_y}.png")
    plt.savefig(train_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Save Validation Accuracy Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(val_pivot, cmap="Oranges", aspect="auto", origin="lower")
    plt.colorbar(label="Validation Accuracy")
    plt.xticks(range(len(val_pivot.columns)), val_pivot.columns, rotation=45)
    plt.yticks(range(len(val_pivot.index)), val_pivot.index)
    plt.xlabel(param_y)
    plt.ylabel(param_x)
    plt.title(f"Validation Accuracy by {param_x} and {param_y}")
    val_plot_path = os.path.join(output_dir, f"validation_accuracy_{param_x}_vs_{param_y}.png")
    plt.savefig(val_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plots saved to {output_dir}:")
    print(f"  - {train_plot_path}")
    print(f"  - {val_plot_path}")