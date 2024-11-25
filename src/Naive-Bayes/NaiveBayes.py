import numpy as np

class NaiveBayes:
    def __init__(self):
        """
        Constructor to initialize model parameters.
        """
        self.classes = None  # To store unique class labels
        self.class_priors = {}  # P(Y): Prior probabilities of each class
        self.feature_likelihoods = {}  # P(X|Y): Likelihoods of features given each class
        self.feature_counts = {}  # Count of features for each class for smoothing
        self.num_features = None  # Number of features in the input data

    def fit(self, X, y):
        """
        Fit the Naive Bayes model using training data (X, y).
        Parameters:
        - X: np.array, shape (n_samples, n_features), feature matrix
        - y: np.array, shape (n_samples,), labels
        """
        # Get unique class labels and feature count
        self.classes = np.unique(y)
        self.num_features = X.shape[1]

        # Compute prior probabilities P(Y) and likelihoods P(X|Y)
        for cls in self.classes:
            # Get all samples belonging to the current class
            X_class = X[y == cls]
            self.class_priors[cls] = len(X_class) / len(y)  # P(Y)

            # Calculate likelihoods P(X|Y) with Laplace smoothing
            # Add 1 for smoothing to avoid zero probabilities
            feature_count = np.sum(X_class, axis=0)
            self.feature_counts[cls] = feature_count
            self.feature_likelihoods[cls] = (feature_count + 1) / (np.sum(feature_count) + self.num_features)

    def predict(self, X):
        """
        Predict the class for each input sample in X.
        Parameters:
        - X: np.array, shape (n_samples, n_features), feature matrix
        Returns:
        - predictions: np.array, shape (n_samples,), predicted labels
        """
        predictions = []
        for sample in X:
            # Compute posterior probabilities P(Y|X) for each class
            posteriors = {}
            for cls in self.classes:
                # Logarithmic probabilities to avoid underflow
                log_prior = np.log(self.class_priors[cls])
                log_likelihood = np.sum(np.log(self.feature_likelihoods[cls]) * sample)
                posteriors[cls] = log_prior + log_likelihood

            # Choose the class with the highest posterior probability
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)

    def evaluate_acc(self, y_true, y_pred):
        """
        Evaluate the accuracy of the model.
        Parameters:
        - y_true: np.array, shape (n_samples,), true labels
        - y_pred: np.array, shape (n_samples,), predicted labels
        Returns:
        - accuracy: float, accuracy score
        """
        return np.mean(y_true == y_pred)