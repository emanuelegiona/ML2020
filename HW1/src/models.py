"""
Model definitions.
"""

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from typing import Any


def get_SVM_model(C: float = 1.0, random_state: Any = None, verbose: int = 0) -> LinearSVC:
    """
    Builds a Support Vector Machines model with linear kernel for classification tasks with the given parameters.
    :param C: error function coefficient (the higher, the less important the regularization term is)
    :param random_state: seed for the pseudo-random number generator
    :param verbose: verbosity level to be used during training
    :return: SVM model for classification using SciKit Learn library
    """

    return LinearSVC(C=C,
                     random_state=random_state,
                     verbose=verbose)


def get_KNN_model(k: int = 5, weighted: bool = False) -> KNeighborsClassifier:
    """
    Builds a K-Nearest Neighbors model for classification tasks with the given parameters.
    :param k: number of neighbors to be used during prediction
    :param weighted: if True, closer neighbors will have a higher impact than distant ones (default: False)
    :return: KNN model for classification using SciKit Learn library
    """

    return KNeighborsClassifier(n_neighbors=k,
                                weights="distance" if weighted else "uniform",
                                algorithm="auto")


if __name__ == "__main__":
    pass
