"""
Model training and parameter tuning.
"""

from src.preprocess import Preprocessor, load_dataset
from src.models import get_SVM_model, get_KNN_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


if __name__ == "__main__":
    train_path = "../data/train_dataset.jsonl"
    X, Y = load_dataset(path=train_path,
                        input_dict_path="../data/2grams.txt",
                        compiler_dict_path="../data/compilers.txt",
                        optimization_dict_path="../data/optimizations.txt",
                        ngrams_size=2,
                        count_addresses=False,
                        count_brackets=False,
                        train=True)

    X = X[:15_000]
    Y = Y[:15_000]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y[:, 0], test_size=0.2)

    model = get_KNN_model()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    print(confusion_matrix(Y_test, Y_pred))
