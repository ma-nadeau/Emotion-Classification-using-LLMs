import numpy as np
from gensim.models import KeyedVectors  # type: ignore
from datasets import Dataset
from nltk import word_tokenize
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import classification_report # type: ignore

from src.Utils import get_single_label_dataset

from wikipedia2vec import Wikipedia2Vec  # type: ignore

# Load the pretrained Wikipedia2Vec model
wiki2vec = Wikipedia2Vec.load("path/to/wikipedia2vec/model/file")


# Load the single-label GoEmotions dataset
ds_train, ds_validation, ds_test = get_single_label_dataset()


# Convert labels to a single integer (since they are lists with one element)
def extract_single_label(example):
    example["label"] = example["labels"][0]  # Assume only one label exists per filtered dataset
    return example


def text_to_embedding(text, model):
    """
    Convert text into an embedding using Wikipedia2Vec.

    Args:
        text (str): The input text.
        model (Wikipedia2Vec): The Wikipedia2Vec model.

    Returns:
        np.ndarray: The average word embedding for the text.
    """
    from nltk.tokenize import word_tokenize
    import numpy as np

    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    embeddings = [
        model.get_word_vector(word) for word in tokens if model.get_word_vector(word) is not None
    ]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.vector_size)


ds_train = ds_train.map(extract_single_label)
ds_validation = ds_validation.map(extract_single_label)
ds_test = ds_test.map(extract_single_label)


# Apply text_to_embedding to each example in the dataset
def embed_text(example):
    embedding = text_to_embedding(example["text"], wiki2vec)
    example["embedding"] = embedding.tolist()  # Convert numpy array to list for compatibility
    return example

ds_train = ds_train.map(embed_text)
ds_validation = ds_validation.map(embed_text)
ds_test = ds_test.map(embed_text)

# Convert to NumPy arrays for ML model training
X_train = np.array(ds_train["embedding"])
y_train = np.array(ds_train["label"])

X_validation = np.array(ds_validation["embedding"])
y_validation = np.array(ds_validation["label"])

X_test = np.array(ds_test["embedding"])
y_test = np.array(ds_test["label"])

# Train a Random Forest classifier as an example
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluate on the validation set
y_pred = rf_classifier.predict(X_validation)
print("Validation Classification Report:")
print(classification_report(y_validation, y_pred))

# Test the model on the test set
y_pred_test = rf_classifier.predict(X_test)
print("Test Classification Report:")
print(classification_report(y_test, y_pred_test))
