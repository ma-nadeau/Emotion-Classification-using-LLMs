import seaborn as sns  # type: ignore
from sklearn.metrics import confusion_matrix  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import os
import math

def plot_distribution_of_datasets(
    train_dataset, eval_dataset, test_dataset, saving_path="../Results-Distilled-GPT2"
):
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

    plt.hist(train_dataset_labels, bins=28, alpha=0.3, label="Train", color="blue")
    plt.hist(eval_dataset_labels, bins=28, alpha=0.3, label="Eval", color="green")
    plt.hist(test_dataset_labels, bins=28, alpha=0.3, label="Test", color="red")

    plt.xticks(range(28))
    plt.xlabel("Classes", fontsize=12, weight="bold")
    plt.ylabel("Frequency", fontsize=12, weight="bold")
    plt.title("Dataset Distribution", fontsize=14, weight="bold")
    plt.legend(title="Dataset", title_fontsize="13", fontsize="10")

    os.makedirs(saving_path, exist_ok=True)
    plt.savefig(f"{saving_path}/dataset_distribution.png", bbox_inches="tight")
    plt.close()


def plot_distribution_of_datasets_binary_vector_labels(
    train_dataset, eval_dataset, test_dataset, saving_path="../Results-Distilled-GPT2"
):
    """
    Plot the distribution of the datasets with binary vector labels.

    Args:
        train_dataset (Dataset): The training dataset.
        eval_dataset (Dataset): The evaluation dataset.
        test_dataset (Dataset): The test dataset.
        saving_path (str): The path where the plot will be saved.
    """
    plt.figure(figsize=(12, 8))
    train_dataset_labels = train_dataset["labels"].sum(axis=0)
    eval_dataset_labels = eval_dataset["labels"].sum(axis=0)
    test_dataset_labels = test_dataset["labels"].sum(axis=0)

    labels = range(28)
    plt.bar(labels, train_dataset_labels, alpha=0.3, label="Train", color="blue")
    plt.bar(
        labels,
        eval_dataset_labels,
        alpha=0.3,
        label="Eval",
        color="green",
        bottom=train_dataset_labels,
    )
    plt.bar(
        labels,
        test_dataset_labels,
        alpha=0.3,
        label="Test",
        color="red",
        bottom=train_dataset_labels + eval_dataset_labels,
    )

    plt.xticks(labels)
    plt.xlabel("Labels", fontsize=12, weight="bold")
    plt.ylabel("Frequency", fontsize=12, weight="bold")
    plt.title("Dataset Distribution (Binary Vector Labels)", fontsize=14, weight="bold")
    plt.legend(title="Dataset", title_fontsize="13", fontsize="10")

    os.makedirs(saving_path, exist_ok=True)
    plt.savefig(
        f"{saving_path}/dataset_distribution_binary_vector_labels.png",
        bbox_inches="tight",
    )
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


def plot_attention_weights(
    attentions, input_tokens, saving_path="../Results-Distilled-GPT2", head=0, layer=0
):
    attention_matrix = attentions[layer][head][0]

    print(attention_matrix)

    plt.figure(figsize=(10, 10))
    sns.heatmap(
        attention_matrix,
        xticklabels=input_tokens,
        yticklabels=input_tokens,
        cmap="coolwarm",
    )
    plt.xlabel("Tokens")
    plt.ylabel("Tokens")
    plt.title(f"Attention Weights - Layer {layer}, Head {head}")
    plt.savefig(f"{saving_path}/attention.png", bbox_inches="tight")
    plt.close()


def plot_all_attention_weights(
    attentions, input_tokens, saving_path="../Results-Distilled-GPT2", token_idx=0, layer=0
):
    num_heads = len(attentions[layer][token_idx])
    cols = 4
    rows = math.ceil(num_heads/cols)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    axes = axes.flatten()

    for idx, attention_matrix in enumerate(attentions[layer][token_idx]):
        sns.heatmap(
            attention_matrix,
            xticklabels=input_tokens,
            yticklabels=input_tokens,
            cmap="coolwarm",
            ax=axes[idx]
        )
        axes[idx].set_title(f"Attention Head {idx}")
        axes[idx].set_xlabel("Tokens")
        axes[idx].set_ylabel("Tokens")

    for idx in range(num_heads, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(f"All Attention Heads for Token {input_tokens[token_idx]} at layer {layer}", fontsize=20)
    plt.tight_layout()
    os.makedirs(saving_path, exist_ok=True)
    plt.savefig(f"{saving_path}/all_attention_heads_for_token_{input_tokens[token_idx]}.png", bbox_inches="tight")
    plt.close()
