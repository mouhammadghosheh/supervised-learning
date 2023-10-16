import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


class DecisionStump():
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None


class AdaBoost():
    def __init__(self, S=10):
        self.S = S

    def fit(self, X, y):
        print(self.S)
        m, n = X.shape
        W = np.full(m, 1 / m)
        self.clfs = []

        for _ in range(self.S):
            clf = DecisionStump()
            min_error = float('inf')

            for feature in range(n):
                feature_vals = np.sort(np.unique(X[:, feature]))
                thresholds = (feature_vals[:-1] + feature_vals[1:]) / 2
                for threshold in thresholds:
                    for polarity in [1, -1]:
                        yhat = np.ones(len(y))
                        yhat[polarity * X[:, feature] < polarity * threshold] = -1
                        error = W[(yhat != y)].sum()

                        if error < min_error:
                            clf.polarity = polarity
                            clf.threshold = threshold
                            clf.feature_index = feature
                            min_error = error

            eps = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + eps) / (min_error + eps))
            W = W * np.exp(-clf.alpha * y * yhat)
            W = W / sum(W)

            self.clfs.append(clf)

    def predict(self, X):
        m, _ = X.shape
        yhat = np.zeros(m)
        for clf in self.clfs:
            pred = np.ones(m)
            pred[clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold] = -1
            yhat += clf.alpha * pred
        return np.sign(yhat)


def preprocess_data(data):
    # Separate features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Fill missing values
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna(X[col].mode().iloc[0])
        else:
            X[col] = X[col].fillna(X[col].median())

    # Convert categorical variables into dummy/indicator variables (i.e. one-hot encoding).
    one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_encoded = one_hot_encoder.fit_transform(X.select_dtypes(include=['object']))
    X_encoded_df = pd.DataFrame(X_encoded, columns=one_hot_encoder.get_feature_names_out(
        X.select_dtypes(include=['object']).columns))

    # Concatenate the one-hot encoded columns to the non-categorical part of X
    X = pd.concat([X.select_dtypes(exclude=['object']).reset_index(drop=True), X_encoded_df.reset_index(drop=True)],
                  axis=1)

    # Convert the labels to {-1, 1}
    y = np.where(y == 0, -1, 1)

    return X.values, y


def plot_cm(cm, title):
    sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title(title, size=15)


from sklearn.ensemble import AdaBoostClassifier


def main():
    # Load dataset
    data = pd.read_csv('heart.csv')

    # Preprocess data
    X, y = preprocess_data(data)

    # Shuffle data
    random_indices = np.random.permutation(X.shape[0])
    X = X[random_indices]
    y = y[random_indices]

    # Split the dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create Logistic Regression classifier
    clf_lr = LogisticRegression(max_iter=1000)

    # Fit the classifier
    clf_lr.fit(X_train, y_train)

    # Make predictions
    predictions_lr = clf_lr.predict(X_test)

    # Calculate accuracy
    accuracy_lr = accuracy_score(y_test, predictions_lr)
    print("Logistic Regression Accuracy:", accuracy_lr)

    # Generate the confusion matrix and classification report
    cm_lr = confusion_matrix(y_test, predictions_lr)
    print("Logistic Regression Confusion Matrix:")
    print(cm_lr)

    cr_lr = classification_report(y_test, predictions_lr)
    print("Logistic Regression Classification Report:")
    print(cr_lr)

    # Create confusion matrix table for Logistic Regression
    tn, fp, fn, tp = cm_lr.ravel()
    confusion_matrix_lr_table = pd.DataFrame({
        "Predicted: Negative Class": [tn, fn],
        "Predicted: Positive Class": [fp, tp]
    }, index=["Actual: Negative Class", "Actual: Positive Class"])
    print("\nLogistic Regression Confusion Matrix Table:")
    print(confusion_matrix_lr_table)

    # Create AdaBoost classifier using sklearn
    clf_ada = AdaBoostClassifier(n_estimators=20, random_state=42)

    # Fit the classifier
    clf_ada.fit(X_train, y_train)

    # Make predictions
    predictions_ada = clf_ada.predict(X_test)

    # Calculate accuracy
    accuracy_ada = accuracy_score(y_test, predictions_ada)
    print("\nAdaBoost Accuracy:", accuracy_ada)

    # Generate the confusion matrix and classification report
    cm_ada = confusion_matrix(y_test, predictions_ada)
    print("AdaBoost Confusion Matrix:")
    print(cm_ada)

    cr_ada = classification_report(y_test, predictions_ada)
    print("AdaBoost Classification Report:")
    print(cr_ada)

    # Create confusion matrix table for AdaBoost
    tn, fp, fn, tp = cm_ada.ravel()
    confusion_matrix_ada_table = pd.DataFrame({
        "Predicted: Negative Class": [tn, fn],
        "Predicted: Positive Class": [fp, tp]
    }, index=["Actual: Negative Class", "Actual: Positive Class"])
    print("\nAdaBoost Confusion Matrix Table:")
    print(confusion_matrix_ada_table)

    # Plot confusion matrix for Logistic Regression
    plot_cm(cm_lr, 'Confusion matrix for Logistic Regression')

    # Plot confusion matrix for AdaBoost
    plot_cm(cm_ada, 'Confusion matrix for AdaBoost')


if __name__ == "__main__":
    main()
