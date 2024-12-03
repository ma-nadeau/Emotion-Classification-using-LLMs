from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from PrepareDataset import preprocess_data, df_train_simplified, df_val_simplified, df_test_simplified

# Train XGBoost Model
def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train an XGBoost classifier and evaluate it on validation data.
    """
    model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=123)
    model.fit(X_train, y_train)

    # Predict on validation set
    y_val_pred = model.predict(X_val)
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))

    return model

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

    y_train_pred = model.predict(X_train)
    print("Train Classification Report:")
    print(classification_report(y_train, y_train_pred))

    return model, vectorizer

# Example Run
run_xgboost_pipeline(df_train_simplified, df_val_simplified, df_test_simplified)