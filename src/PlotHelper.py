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
    plt.xlabel("Predicted Labels", fontsize=22, weight="bold")
    plt.ylabel("True Labels", fontsize=22, weight="bold")
    plt.title("Confusion Matrix", fontsize=24, weight="bold")
    plt.xticks(fontsize=10, weight="bold")
    plt.yticks(fontsize=10, weight="bold")


    os.makedirs(saving_path, exist_ok=True)
    plt.savefig(f"{saving_path}/confusion_matrix.png", bbox_inches="tight")
    plt.close()

