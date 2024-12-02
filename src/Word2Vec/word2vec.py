import ssl
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import  accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils.tests.test_pprint import CountVectorizer
from xgboost import XGBClassifier
from wikipedia2vec import Wikipedia2Vec
from nltk.tokenize import word_tokenize
from src.Utils import get_single_label_dataset


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


def vectorize_text(ds_train, ds_test, method="tfidf"):
    """
    Vectorize text data using CountVectorizer or TfidfVectorizer.
    """
    if method == "tfidf":
        vectorizer = TfidfVectorizer(max_features=5000)  # Limit features for performance
    else:
        vectorizer = CountVectorizer(max_features=5000)

    # Fit on training text and transform both train and test sets
    X_train = vectorizer.fit_transform(ds_train["text"]).toarray()
    X_test = vectorizer.transform(ds_test["text"]).toarray()

    return X_train, X_test, vectorizer


def softmax_regression(X_train, y_train, X_test, y_test):
    """
    Regularized Softmax Regression using L2 regularization.
    """
    softmax_model = LogisticRegression(
        penalty="l2",  # L2 regularization
        C=0.5,  # Regularization strength (smaller = stronger regularization)
        solver="lbfgs",  # Solver for optimization (supports multi-class)
        multi_class="multinomial",  # Enable softmax regression
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

    # Vectorize the text data
    X_train, X_test, vectorizer = vectorize_text(ds_train, ds_test, method="tfidf")
    y_train = np.array(ds_train["label"])
    y_test = np.array(ds_test["label"])

    # Train Random Forest
    rf_classifier = RandomForestClassifier(max_depth=30,random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred_rf = rf_classifier.predict(X_test)
    rf_test_acc = accuracy_score(y_test, y_pred_rf)

    # Train XGBoost
    xgb_classifier = XGBClassifier(reg_lambda=100,random_state=42, eval_metric='logloss')
    xgb_classifier.fit(X_train, y_train)
    y_pred_xgb = xgb_classifier.predict(X_test)
    xgb_test_acc = accuracy_score(y_test, y_pred_xgb)

    # Train Softmax Regression
    softmax_model, softmax_test_acc = softmax_regression(
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
    results_table.to_csv("regularized_results_table.csv", index=False)
    print(results_table)


if __name__ == "__main__":
    main()
