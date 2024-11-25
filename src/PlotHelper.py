import seaborn as sns  # type: ignore
from sklearn.metrics import confusion_matrix  # type: ignore
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, TrainingArguments, AdamW  # type: ignore
from datasets import load_dataset  # type: ignore
import torch  # type: ignore
from tqdm import tqdm  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from transformers import get_scheduler  # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os

def plot_confusion_matrix(model, test_dataset, batch_size=32, saving_path="../Results-Distilled-GPT2"):
    """
    Plot the confusion matrix of the model on the test set.

    Args:
        model (PreTrainedModel): The trained model.
        test_dataset (Dataset): The test dataset.
        batch_size (int): The batch size for the DataLoader.
    """
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    model.eval()
    all_predictions = []
    all_labels = []
    for batch in test_dataloader:
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
        
    plt.savefig(f"{saving_path}/confusion_matrix.png")
    