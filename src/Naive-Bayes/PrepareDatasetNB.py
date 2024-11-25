import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

splits = {'train': 'simplified/train-00000-of-00001.parquet',
          'validation': 'simplified/validation-00000-of-00001.parquet',
          'test': 'simplified/test-00000-of-00001.parquet'}
df_train = pd.read_parquet("hf://datasets/google-research-datasets/go_emotions/" + splits["train"])
df_val = pd.read_parquet("hf://datasets/google-research-datasets/go_emotions/" + splits["validation"])
df_test = pd.read_parquet("hf://datasets/google-research-datasets/go_emotions/" + splits["test"])

# Drop data points with more than one label
df_train_simplified = df_train[df_train["labels"].apply(lambda x: len(x) == 1)]
df_val_simplified = df_val[df_val["labels"].apply(lambda x: len(x) == 1)]
df_test_simplified = df_test[df_test["labels"].apply(lambda x: len(x) == 1)]


'''
df_train_simplified.reset_index(drop=True, inplace=True)
df_val_simplified.reset_index(drop=True, inplace=True)
df_test_simplified.reset_index(drop=True, inplace=True)
'''



# Example preprocessing pipeline using CountVectorizer
def preprocess_data(df_train, df_val, df_test, text_column="text", label_column="labels"):
    """
    Preprocess text data using Bag of Words representation with CountVectorizer.
    """
    # Initialize CountVectorizer
    vectorizer = CountVectorizer()

    # Fit on the training text data and transform the splits
    X_train = vectorizer.fit_transform(df_train[text_column])  # Learn vocabulary and transform train
    X_val = vectorizer.transform(df_val[text_column])          # Transform validation data
    X_test = vectorizer.transform(df_test[text_column])        # Transform test data

    # Convert labels to a simple format (if needed)
    y_train = df_train[label_column].apply(lambda x: x[0])  # Single label per row
    y_val = df_val[label_column].apply(lambda x: x[0])
    y_test = df_test[label_column].apply(lambda x: x[0])

    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer

# Example usage
X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = preprocess_data(
    df_train_simplified, df_val_simplified, df_test_simplified, text_column="text", label_column="labels"
)

# Check the shape of the processed data
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Test data shape: {X_test.shape}")
