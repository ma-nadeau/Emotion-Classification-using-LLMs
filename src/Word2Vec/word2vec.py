import ssl

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import classification_report  # type: ignore
from sklearn.model_selection import GridSearchCV
from wikipedia2vec import Wikipedia2Vec
from nltk.tokenize import word_tokenize
from xgboost import XGBClassifier

from src.Utils import get_single_label_dataset, oversample_dataset
import nltk


def text_to_embedding(text, model):
    """
    Convert text into an embedding using Wikipedia2Vec.

    Args:
        text (str): The input text.
        model (Wikipedia2Vec): The Wikipedia2Vec model.

    Returns:
        np.ndarray: The average word embedding for the text.
    """
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    embeddings = []
    for word in tokens:
        try:
            embedding = model.get_word_vector(word)
            embeddings.append(embedding)
        except KeyError:
            # Skip words not in the vocabulary
            continue

    # Get vector size dynamically from a known word
    if embeddings:
        vector_size = len(embeddings[0])  # Use the size of any embedding in the list
    else:
        # If no embeddings are found, assume the size of the model's embeddings
        try:
            vector_size = len(model.get_word_vector('example'))  # Replace 'example' with a common word
        except KeyError:
            raise ValueError("Unable to determine vector size. No embeddings found in text or model.")

    return np.mean(embeddings, axis=0) if embeddings else np.zeros(vector_size)


def extract_single_label(example):
    """
    Extract a single label from the dataset.
    Assumes each example has exactly one label.

    Args:
        example (dict): A single dataset example.

    Returns:
        dict: Modified example with a single label.
    """
    example["label"] = example["labels"][0]  # Extract the single label
    return example


def embed_text(example, wiki2vec):
    """
    Embed the text of an example using Wikipedia2Vec.

    Args:
        example (dict): A single dataset example.
        wiki2vec (Wikipedia2Vec): Pretrained Wikipedia2Vec model.

    Returns:
        dict: Modified example with the embedded text.
    """
    embedding = text_to_embedding(example["text"], wiki2vec)
    example["embedding"] = embedding.tolist()  # Convert numpy array to list for compatibility
    return example


def main():
    ssl._create_default_https_context = ssl._create_unverified_context

    # nltk.download('punkt_tab')

    # MODEL FILE IN DOWNLOADED FOLDERS.
    wiki2vec = wiki2vec = Wikipedia2Vec.load('enwiki_20180420_100d.pkl')

    # Load the single-label GoEmotions dataset
    ds_train, ds_validation, ds_test = get_single_label_dataset()

    # Process the datasets
    ds_train = ds_train.map(extract_single_label)
    ds_validation = ds_validation.map(extract_single_label)
    ds_test = ds_test.map(extract_single_label)

    ds_train = ds_train.map(lambda example: embed_text(example, wiki2vec))
    ds_validation = ds_validation.map(lambda example: embed_text(example, wiki2vec))
    ds_test = ds_test.map(lambda example: embed_text(example, wiki2vec))

    # Convert to NumPy arrays for ML model training
    X_train = np.array(ds_train["embedding"])
    y_train = np.array(ds_train["label"])

    X_validation = np.array(ds_validation["embedding"])
    y_validation = np.array(ds_validation["label"])

    X_test = np.array(ds_test["embedding"])
    y_test = np.array(ds_test["label"])

    # Apply SMOTE to balance the training set for Random Forest
    # smote = SMOTE(random_state=42)
    # X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train a Random Forest classifier
    # rf_classifier = RandomForestClassifier(random_state=42)
    # rf_classifier.fit(X_train_resampled, y_train_resampled)

    # Train an XGBoost classifier
    xgb_classifier = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_classifier.fit(X_train, y_train)

    # Define the parameter grid
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "scale_pos_weight": [1, 10, 20],
    }

    # Initialize XGBoost classifier
    # Grid Search
    grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=3, scoring="f1_weighted", verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best parameters
    print("Best Parameters:", grid_search.best_params_)

    # Evaluate on the validation set
    y_pred = xgb_classifier.predict(X_validation)
    print("Validation Classification Report:")
    print(classification_report(y_validation, y_pred))

    # Test the model on the test set
    y_pred_test = xgb_classifier.predict(X_test)
    print("Test Classification Report:")
    print(classification_report(y_test, y_pred_test))


if __name__ == "__main__":
    main()
