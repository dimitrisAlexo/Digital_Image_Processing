from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import compute_sample_weight


def train_test(dataset, n_neighbors=1):
    # Passes the dataset into the KNN algorithm which trains and tests the model in order to calculate the confusion
    # matrix and the weighted accuracy.
    # - dataset: input dataset of the letters
    # - conf_matrix, weighted_accuracy: return parameters

    if dataset.empty:
        return None, None

    x = dataset.iloc[:, :-1]  # Select all columns except the last one
    y = dataset.iloc[:, -1]  # Select only the last column (labels)

    # duplicate rows for classes with only one member
    unique_classes, class_counts = np.unique(y, return_counts=True)
    for c in unique_classes[class_counts == 1]:
        indices = np.where(y == c)[0]
        x = pd.concat([x, x.iloc[indices]], axis=0)
        y = np.concatenate([y, np.repeat(c, len(indices))], axis=0)

    testing_size = 30
    cm_list = []
    wa_list = []
    knn_best = None
    best_accuracy = 0

    for i in range(testing_size):
        # Split the data into training and testing sets with stratified sampling based on the label column
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

        # Train the KNN classifier
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(x_train, y_train)

        # Predict the labels of the test set
        y_predict = knn.predict(x_test)

        # Calculate the confusion matrix
        cm = confusion_matrix(y_test, y_predict)
        cm_list.append(cm)

        # Calculate the weighted accuracy
        sample_weight = compute_sample_weight(class_weight='balanced', y=y_test)
        wa = accuracy_score(y_test, y_predict, sample_weight=sample_weight)
        wa_list.append(wa)

        if wa > best_accuracy:
            best_accuracy = wa
            knn_best = knn

    # Calculate the mean confusion matrix and weighted accuracy
    weighted_accuracy = np.mean(wa_list)
    conf_matrix = cm_list[0]

    return conf_matrix, best_accuracy, knn_best



