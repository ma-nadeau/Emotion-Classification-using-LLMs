from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report  # type: ignore


def compute_accuracy(prediction, labels, dataset, model_name):
    accuracy = accuracy_score(labels, prediction)
    print(f"Accuracy {model_name} on {dataset}: {accuracy}")
    return accuracy


def compute_recall(prediction, labels, dataset, model_name):
    recall = recall_score(labels, prediction, average="weighted")
    print(f"Recall {model_name} on {dataset}: {recall}")
    return recall


def compute_f1(prediction, labels, dataset, model_name):
    f1 = f1_score(labels, prediction, average="weighted")
    print(f"F1 {model_name} on {dataset}: {f1}")
    return f1


def compute_precision(prediction, labels, dataset, model_name):
    precision = precision_score(labels, prediction, average="weighted")
    print(f"Precision {model_name} on {dataset}: {precision}")
    return precision


def compute_classification_report(prediction, labels, dataset, model_name):
    report = classification_report(labels, prediction)
    print(f"Classification Report {model_name} on {dataset}:\n{report}")
    return report
