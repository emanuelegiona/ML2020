"""
Model definitions.
"""

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from typing import Any


def get_SVM_model(C: float = 1.0, kernel: str = "rbf", gamma: Any = "auto", random_state: Any = None) -> SVC:
    """
    Builds a Support Vector Machines model for classification tasks with the given parameters.
    :param C: error function coefficient (the higher, the less important the regularization term is)
    :param kernel: kernel to be used in SVM computation
    :param gamma: kernel coefficient in cases of RBF, polynomial or sigmoid kernels
    :param random_state: seed for the pseudo-random number generator
    :return: SVM model for classification using SciKit Learn library
    """

    return SVC(C=C,
               kernel=kernel,
               gamma=gamma,
               random_state=random_state,
               cache_size=1024)


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
