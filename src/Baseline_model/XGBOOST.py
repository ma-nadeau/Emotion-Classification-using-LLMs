import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report


def clean_text(text):
    """
    Clean the input text by applying several preprocessing steps.
    """
    # Lowercase
    text = text.lower()
    # Remove non-emotional punctuation (keep !, ?)
    text = re.sub(r"[^\w\s!?':;.,]", '', text)
    return text

def preprocess_data(df_train, df_val, df_test, text_column="text", label_column="labels", vectorizer_type="tfidf"):
    """
    Preprocess text data using a vectorizer (CountVectorizer or TfidfVectorizer).
    """
    # Apply text cleaning to each dataset
    df_train[text_column] = df_train[text_column].apply(clean_text)
    df_val[text_column] = df_val[text_column].apply(clean_text)
    df_test[text_column] = df_test[text_column].apply(clean_text)

    # Initialize the vectorizer
    if vectorizer_type == "count":
        vectorizer = CountVectorizer(stop_words="english", max_features=700, min_df=4)
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

# Train XGBoost Model
def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train an XGBoost classifier and evaluate it on validation data.
    """
    model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    model.fit(X_train, y_train)

    # Predict on validation set
    y_val_pred = model.predict(X_val)

    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))

    return model

# Example Workflow
def run_xgboost_pipeline(df_train, df_val, df_test):
    """
    Complete pipeline for training and evaluating an XGBoost model.
    """
    # Preprocess data using TF-IDF
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = preprocess_data(
        df_train, df_val, df_test, text_column="text", label_column="labels", vectorizer_type="tfidf"
    )

    # Train and evaluate XGBoost
    model = train_xgboost(X_train, y_train, X_val, y_val)

    # Predict on test set
    y_test_pred = model.predict(X_test)

    print("Test Classification Report:")
    print(classification_report(y_test, y_test_pred))

    return model, vectorizer