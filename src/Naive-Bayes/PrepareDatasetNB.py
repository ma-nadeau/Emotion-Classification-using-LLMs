import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from NaiveBayes import *
import os
import re
from sklearn.metrics import classification_report

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


def downsample(data):
    majority = data[data['labels'] == 27]
    minority = data[data['labels'] != 27]

    # Downsample majority class to match minority class size
    majority_downsampled = majority.sample(n=int(len(majority)/6), random_state=42)

    # Combine downsampled majority class with minority class
    balanced_data = pd.concat([majority_downsampled, minority])

    return balanced_data

def clean_text(text):
    """
    Clean the input text by applying several preprocessing steps.
    """
    # Lowercase
    text = text.lower()
    # Remove non-emotional punctuation (keep !, ?)
    text = re.sub(r"[^\w\s!?':;.,]", '', text)
    return text

# Example preprocessing pipeline using CountVectorizer
def preprocess_data(df_train, df_val, df_test, text_column="text", label_column="labels"):
    """
    Preprocess text data using Bag of Words representation with CountVectorizer.
    """


    # Do the text cleaning
    df_train[text_column] = df_train[text_column].apply(clean_text)
    df_val[text_column] = df_val[text_column].apply(clean_text)
    df_test[text_column] = df_test[text_column].apply(clean_text)


    # Initialize CountVectorizer
    vectorizer = CountVectorizer(stop_words="english",ngram_range=(1,2),max_features=700,min_df = 2)

    # Fit on the training text data and transform the splits
    X_train = vectorizer.fit_transform(df_train[text_column])  # Learn vocabulary and transform train
    X_val = vectorizer.transform(df_val[text_column])          # Transform validation data
    X_test = vectorizer.transform(df_test[text_column])        # Transform test data

    # Convert labels to a simple format
    y_train = df_train[label_column].apply(lambda x: x[0])  # Single label per row
    y_val = df_val[label_column].apply(lambda x: x[0])
    y_test = df_test[label_column].apply(lambda x: x[0])

    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer

# Example usage
X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = preprocess_data(
    df_train_simplified, df_val_simplified, df_test_simplified, text_column="text", label_column="labels"
)


vocab = vectorizer.get_feature_names_out()
word_counts = np.asarray(X_train.sum(axis=0)).flatten()

# Create a DataFrame to display word frequencies
word_freq = pd.DataFrame({"word": vocab, "count": word_counts})
word_freq = word_freq.sort_values(by="count", ascending=False)

def evaluate_naive_bayes_pipeline(df_train, df_val, df_test, downsample_flag=False):
    """
    Evaluate Naive Bayes performance with and without downsampling.
    """
    if downsample_flag:
        print("Using downsampled data...")
        df_train = downsample(df_train)
        df_val = downsample(df_val)
        df_test = downsample(df_test)
    else:
        print("Using original data without downsampling...")

    # Preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = preprocess_data(
        df_train, df_val, df_test, text_column="text", label_column="labels"
    )

    # Fit the Naive Bayes model
    model = NaiveBayes()
    model.fit(X_train.toarray(), y_train)

    # Evaluate on train data
    y_train_pred = model.predict(X_train.toarray())
    print("\n--- Classification Report (Train Data) ---")
    print(classification_report(y_train, y_train_pred))

    # Evaluate on test data
    y_test_pred = model.predict(X_test.toarray())
    print("\n--- Classification Report (Test Data) ---")
    print(classification_report(y_test, y_test_pred))

    # Compute overall accuracy and per-class metrics
    accuracy = model.evaluate_acc(y_test, y_test_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    metrics = model.evaluate_precision_recall(y_test, y_test_pred)
    print("\nPer-class Precision and Recall:")
    for cls, values in metrics.items():
        print(f"Class {cls}: Precision = {values['precision']:.4f}, Recall = {values['recall']:.4f}")

    # Plot Precision and Recall
    plot_precision_recall(metrics, downsample_flag)

def plot_precision_recall(metrics, downsample_flag):
    """
    Plot precision and recall values for each class.
    """
    classes = sorted(metrics.keys())
    precision_values = [metrics[cls]['precision'] for cls in classes]
    recall_values = [metrics[cls]['recall'] for cls in classes]

    # Plotting
    x = np.arange(len(classes))
    bar_width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - bar_width/2, precision_values, bar_width, label='Precision', color='skyblue')
    plt.bar(x + bar_width/2, recall_values, bar_width, label='Recall', color='lightgreen')

    # Add labels and title
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title(f'Precision and Recall for Each Class {"(Downsampled)" if downsample_flag else "(Original)"}')
    plt.xticks(x, classes)
    plt.ylim(0, 1.1)
    plt.legend()

    # Save plot
    result_folder = "../../Results-Naive-Bayes"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    result_file = os.path.join(result_folder, f"Precision_Recall_{'downsampled' if downsample_flag else 'original'}.png")
    plt.tight_layout()
    plt.savefig(result_file)
    plt.close()

# Evaluate with original data
evaluate_naive_bayes_pipeline(df_train_simplified, df_val_simplified, df_test_simplified, downsample_flag=False)

# Evaluate with downsampled data
evaluate_naive_bayes_pipeline(df_train_simplified, df_val_simplified, df_test_simplified, downsample_flag=True)