import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from PrepareDataset import preprocess_data, df_train_simplified, df_val_simplified, df_test_simplified
import os
def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 50]
    }

    # Perform Grid Search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy',
                               return_train_score=True)
    grid_search.fit(X_train, y_train)

    # Capture performance of the top 10 models
    results = grid_search.cv_results_
    scores_and_params = list(zip(results['mean_test_score'], results['params']))
    top_10_models = sorted(scores_and_params, key=lambda x: x[0], reverse=True)[:10]  # Sort by score

    # Print top 10 models
    print("\nTop 10 Models during Grid Search:")
    for i, (score, params) in enumerate(top_10_models, start=1):
        print(f"{i}. Accuracy: {score:.4f}, Parameters: {params}")

    return grid_search.best_estimator_, top_10_models


def plot_model_performance(top_10_models):
    # Extract accuracy scores and parameter strings
    accuracies = [score for score, _ in top_10_models]
    params = [str(params) for _, params in top_10_models]

    # Create a bar plot
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(accuracies)), accuracies, color='skyblue')
    plt.xlabel("Accuracy", fontsize=14)
    plt.ylabel("Model Rank", fontsize=14)
    plt.yticks(range(len(params)), labels=params, fontsize=10)
    plt.title("Performance of Top 10 Models During Hyperparameter Tuning", fontsize=16)
    plt.gca().invert_yaxis()  # Highest score on top
    plt.tight_layout()
    result_folder = "../../Results-RandomF"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Correcting the filename
    result_file = os.path.join(result_folder, "Compare_through_RFs")
    plt.savefig(result_file)
    plt.close()


def evaluate_best_model_on_test(best_model, X_test, y_test, class_names):
    # Predict on the test set
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)

    # Compute Accuracy
    accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {accuracy:.4f}\n")


    # Plot ROC Curve (works for binary or multiclass classification)
    plt.figure(figsize=(10, 7))
    if len(class_names) == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_test, y_test_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    else:
        # Multiclass classification (one-vs-rest approach)
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_test == i, y_test_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {class_name} (area = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label="Chance level")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve on Test Set")
    plt.legend(loc="lower right")
    plt.grid()
    result_folder = "../../Results-RandomF"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Correcting the filename
    result_file = os.path.join(result_folder, "ROC_best_model")
    plt.savefig(result_file)
    plt.close()

def run_random_forest_pipeline(df_train, df_val, df_test):
    # Preprocess data using TF-IDF
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = preprocess_data(
        df_train, df_val, df_test, text_column="text", label_column="labels", vectorizer_type="tfidf"
    )

    # Train Random Forest with hyperparameter tuning
    best_model, top_10_models = train_random_forest(X_train, y_train)

    # Predict on validation set
    y_val_pred = best_model.predict(X_val)
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))

    # Predict on test set
    y_test_pred = best_model.predict(X_test)
    print("Test Classification Report:")
    print(classification_report(y_test, y_test_pred))

    # Plot performance of top 10 models
    plot_model_performance(top_10_models)

    class_names = np.unique(df_train['labels']) # Assuming these are the class names
    evaluate_best_model_on_test(best_model, X_test, y_test, class_names)

    return best_model, vectorizer


# Example Run
best_model, vectorizer = run_random_forest_pipeline(df_train_simplified, df_val_simplified, df_test_simplified)

