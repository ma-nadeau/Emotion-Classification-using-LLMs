import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from NaiveBayes import *
import os
import re

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


'''
df_test_simplified = downsample(df_test_simplified)
df_train_simplified = downsample(df_train_simplified)
df_val_simplified = downsample(df_val_simplified)
'''


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
    vectorizer = CountVectorizer(stop_words="english",max_features=700, min_df=4)

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

print(y_train.shape)

vocab = vectorizer.get_feature_names_out()
word_counts = np.asarray(X_train.sum(axis=0)).flatten()

# Create a DataFrame to display word frequencies
word_freq = pd.DataFrame({"word": vocab, "count": word_counts})
word_freq = word_freq.sort_values(by="count", ascending=False)
print(word_freq)

model = NaiveBayes()
model.fit(X_train.toarray(), y_train)
y_pred = model.predict(X_val.toarray())
accuracy = model.evaluate_acc(y_val, y_pred)
metrics = model.evaluate_precision_recall(y_val,y_pred)
print(accuracy)

def plot_precision_recall(metrics):
    classes = sorted(metrics.keys())
    precision_values = [metrics[cls]['precision'] for cls in classes]
    recall_values = [metrics[cls]['recall'] for cls in classes]

    # Plotting
    x = np.arange(len(classes))  # class labels on the x-axis

    plt.figure(figsize=(10, 6))
    bar_width = 0.35

    # Precision bars
    plt.bar(x - bar_width/2, precision_values, bar_width, label='Precision', color='skyblue')

    # Recall bars
    plt.bar(x + bar_width/2, recall_values, bar_width, label='Recall', color='lightgreen')

    # Add labels and title
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title('Precision and Recall for Each Class')
    plt.xticks(x, classes)  # Set class labels
    plt.ylim(0, 1.1)  # Set y-axis limit
    plt.legend()

    # Display plot
    plt.tight_layout()

    result_folder = "../../Results-Naive-Bayes"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Correcting the filename
    result_file = os.path.join(result_folder, "Precision & Recall along classes")
    plt.savefig(result_file)
    plt.close()

plot_precision_recall(metrics)