# import ssl
#
# import numpy as np
# import pandas as pd
# from imblearn.over_sampling import SMOTE
# from sklearn.ensemble import RandomForestClassifier  # type: ignore
# from sklearn.metrics import classification_report, accuracy_score  # type: ignore
# from sklearn.model_selection import GridSearchCV
# from wikipedia2vec import Wikipedia2Vec
# from nltk.tokenize import word_tokenize
# from xgboost import XGBClassifier
#
# from src.Utils import get_single_label_dataset, oversample_dataset
# import nltk
#
#
# def text_to_embedding(text, model):
#     """
#     Convert text into an embedding using Wikipedia2Vec.
#
#     Args:
#         text (str): The input text.
#         model (Wikipedia2Vec): The Wikipedia2Vec model.
#
#     Returns:
#         np.ndarray: The average word embedding for the text.
#     """
#     tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
#     embeddings = []
#     for word in tokens:
#         try:
#             embedding = model.get_word_vector(word)
#             embeddings.append(embedding)
#         except KeyError:
#             # Skip words not in the vocabulary
#             continue
#
#     # Get vector size dynamically from a known word
#     if embeddings:
#         vector_size = len(embeddings[0])  # Use the size of any embedding in the list
#     else:
#         # If no embeddings are found, assume the size of the model's embeddings
#         try:
#             vector_size = len(model.get_word_vector('example'))  # Replace 'example' with a common word
#         except KeyError:
#             raise ValueError("Unable to determine vector size. No embeddings found in text or model.")
#
#     return np.mean(embeddings, axis=0) if embeddings else np.zeros(vector_size)
#
#
# def extract_single_label(example):
#     """
#     Extract a single label from the dataset.
#     Assumes each example has exactly one label.
#
#     Args:
#         example (dict): A single dataset example.
#
#     Returns:
#         dict: Modified example with a single label.
#     """
#     example["label"] = example["labels"][0]  # Extract the single label
#     return example
#
#
# def embed_text(example, wiki2vec):
#     """
#     Embed the text of an example using Wikipedia2Vec.
#
#     Args:
#         example (dict): A single dataset example.
#         wiki2vec (Wikipedia2Vec): Pretrained Wikipedia2Vec model.
#
#     Returns:
#         dict: Modified example with the embedded text.
#     """
#     embedding = text_to_embedding(example["text"], wiki2vec)
#     example["embedding"] = embedding.tolist()  # Convert numpy array to list for compatibility
#     return example
#
#
# def main():
#     ssl._create_default_https_context = ssl._create_unverified_context
#
#     # nltk.download('punkt_tab')
#
#     # MODEL FILE IN DOWNLOADED FOLDERS.
#     wiki2vec = wiki2vec = Wikipedia2Vec.load('enwiki_20180420_100d.pkl')
#
#     # Load the single-label GoEmotions dataset
#     ds_train, ds_validation, ds_test = get_single_label_dataset()
#
#     # Process the datasets
#     ds_train = ds_train.map(extract_single_label)
#     ds_validation = ds_validation.map(extract_single_label)
#     ds_test = ds_test.map(extract_single_label)
#
#     ds_train = ds_train.map(lambda example: embed_text(example, wiki2vec))
#     ds_validation = ds_validation.map(lambda example: embed_text(example, wiki2vec))
#     ds_test = ds_test.map(lambda example: embed_text(example, wiki2vec))
#
#     # Convert to NumPy arrays for ML model training
#     X_train = np.array(ds_train["embedding"])
#     y_train = np.array(ds_train["label"])
#
#     X_validation = np.array(ds_validation["embedding"])
#     y_validation = np.array(ds_validation["label"])
#
#     X_test = np.array(ds_test["embedding"])
#     y_test = np.array(ds_test["label"])
#
#     # Apply SMOTE to balance the training set for Random Forest
#     # smote = SMOTE(random_state=42)
#     # X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
#
#     # Train Random Forest
#     rf_classifier = RandomForestClassifier(max_depth=15, random_state=42)
#
#     rf_classifier.fit(X_train, y_train)
#     y_pred_rf = rf_classifier.predict(X_test)
#     rf_test_acc = accuracy_score(y_test, y_pred_rf)
#
#     # Train XGBoost
#     xgb_classifier = XGBClassifier(reg_lambda=20, random_state=42, eval_metric='logloss')
#     xgb_classifier.fit(X_train, y_train)
#     y_pred_xgb = xgb_classifier.predict(X_test)
#     xgb_test_acc = accuracy_score(y_test, y_pred_xgb)
#
#
#     # Populate results table
#     results_table = pd.DataFrame({
#         "Classifier": ["Random Forest", "XGBoost"],
#         "Training Accuracy (%)": [accuracy_score(y_train, rf_classifier.predict(X_train)) * 100,
#                                    accuracy_score(y_train, xgb_classifier.predict(X_train)) * 100],
#         "Test Accuracy (%)": [rf_test_acc * 100, xgb_test_acc * 100],
#     })
#
#     # Save table to CSV and print it
#     results_table.to_csv("results_table.csv", index=False)
#     print(results_table)
#
#
#
# if __name__ == "__main__":
#     main()
import ssl
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from wikipedia2vec import Wikipedia2Vec
from nltk.tokenize import word_tokenize
from src.Utils import get_single_label_dataset
import nltk


def text_to_embedding(text, model):
    tokens = word_tokenize(text.lower())
    embeddings = []
    for word in tokens:
        try:
            embedding = model.get_word_vector(word)
            embeddings.append(embedding)
        except KeyError:
            continue
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(len(model.get_word_vector('example')))


def extract_single_label(example):
    example["label"] = example["labels"][0]
    return example


def embed_text(example, wiki2vec):
    embedding = text_to_embedding(example["text"], wiki2vec)
    example["embedding"] = embedding.tolist()
    return example


def softmax_regression(X_train, y_train, X_test, y_test):
    """
    Regularized Softmax Regression using L2 regularization.
    """
    softmax_model = LogisticRegression(
        # penalty="l2",  # L2 regularization
        # C=1.0,  # Regularization strength (smaller = stronger regularization)
        solver="lbfgs",  # Solver for optimization (supports multi-class)
        multi_class="multinomial",  # Enable softmax regression
        max_iter=5000,  # Increase iterations for convergence
        random_state=42
    )

    # Train the model
    softmax_model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred_test = softmax_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)

    return softmax_model, test_acc


def main():
    ssl._create_default_https_context = ssl._create_unverified_context

    # Load the pre-trained Wikipedia2Vec model
    wiki2vec = Wikipedia2Vec.load('enwiki_20180420_100d.pkl')

    # Load the single-label GoEmotions dataset
    ds_train, ds_validation, ds_test = get_single_label_dataset()

    # Process the datasets
    ds_train = ds_train.map(extract_single_label)
    ds_test = ds_test.map(extract_single_label)

    ds_train = ds_train.map(lambda example: embed_text(example, wiki2vec))
    ds_test = ds_test.map(lambda example: embed_text(example, wiki2vec))

    # Convert to NumPy arrays for ML model training
    X_train = np.array(ds_train["embedding"])
    y_train = np.array(ds_train["label"])
    X_test = np.array(ds_test["embedding"])
    y_test = np.array(ds_test["label"])

    # Train Random Forest
    rf_classifier = RandomForestClassifier(max_depth=11, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred_rf = rf_classifier.predict(X_test)
    rf_test_acc = accuracy_score(y_test, y_pred_rf)

    # Train XGBoost
    xgb_classifier = XGBClassifier(reg_lambda=350, random_state=42, eval_metric='logloss')
    xgb_classifier.fit(X_train, y_train)
    y_pred_xgb = xgb_classifier.predict(X_test)
    xgb_test_acc = accuracy_score(y_test, y_pred_xgb)

    # Train Softmax Regression
    softmax_model,softmax_test_acc = softmax_regression(
        X_train, y_train, X_test, y_test
    )

    # Populate results table
    results_table = pd.DataFrame({
        "Classifier": ["Random Forest", "XGBoost", "Softmax Regression"],
        "Training Accuracy (%)": [
            accuracy_score(y_train, rf_classifier.predict(X_train)) * 100,
            accuracy_score(y_train, xgb_classifier.predict(X_train)) * 100,
            accuracy_score(y_train, softmax_model.predict(X_train)) * 100
        ],

        "Test Accuracy (%)": [
            rf_test_acc * 100,
            xgb_test_acc * 100,
            softmax_test_acc * 100
        ]
    })

    # Save table to CSV and print it
    results_table.to_csv("results_table.csv", index=False)
    print(results_table)


if __name__ == "__main__":
    main()
