import pandas as pd
import numpy as np

# Data Loading Paths
splits = {
    'train': 'simplified/train-00000-of-00001.parquet',
    'validation': 'simplified/validation-00000-of-00001.parquet',
    'test': 'simplified/test-00000-of-00001.parquet'
}

df_train = pd.read_parquet("hf://datasets/google-research-datasets/go_emotions/" + splits["train"])
df_val = pd.read_parquet("hf://datasets/google-research-datasets/go_emotions/" + splits["validation"])
df_test = pd.read_parquet("hf://datasets/google-research-datasets/go_emotions/" + splits["test"])

# Drop data points with more than one label
df_train_simplified = df_train[df_train["labels"].apply(lambda x: len(x) == 1)]
df_val_simplified = df_val[df_val["labels"].apply(lambda x: len(x) == 1)]
df_test_simplified = df_test[df_test["labels"].apply(lambda x: len(x) == 1)]

# Downsample Function
def downsample(data):
    majority = data[data['labels'] == 27]
    minority = data[data['labels'] != 27]

    # Downsample majority class to match minority class size
    majority_downsampled = majority.sample(n=int(len(majority)/6), random_state=42)

    # Combine downsampled majority class with minority class
    balanced_data = pd.concat([majority_downsampled, minority])

    return balanced_data

df_train_simplified = downsample(df_train_simplified)
df_val_simplified = downsample(df_val_simplified)
df_test_simplified = downsample(df_test_simplified)

# Preprocess Data Function (Moved from XGBOOST.py)
def preprocess_data(df_train, df_val, df_test, text_column, label_column, vectorizer_type="tfidf"):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    # Choose vectorizer
    if vectorizer_type == "count":
        vectorizer = CountVectorizer(stop_words="english", max_features=700)
    elif vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(stop_words="english")
    else:
        raise ValueError("Invalid vectorizer type. Choose 'count' or 'tfidf'.")

    # Fit on the training text data and transform the splits
    X_train = vectorizer.fit_transform(df_train[text_column])  # Learn vocabulary and transform train
    X_val = vectorizer.transform(df_val[text_column])          # Transform validation data
    X_test = vectorizer.transform(df_test[text_column])        # Transform test data

    # Convert labels to a simple format (if needed)
    y_train = df_train[label_column].apply(lambda x: x[0])  # Single label per row
    y_val = df_val[label_column].apply(lambda x: x[0])
    y_test = df_test[label_column].apply(lambda x: x[0])

    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer