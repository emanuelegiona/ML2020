"""
Prediction and creation of output files in the given format.
"""

from src.preprocess import Preprocessor, load_dataset
from src.train import log_message
from joblib import load
import random


def predict(test_path: str, compiler_model_path: str, optimization_model_path: str,
            input_dict_path: str, ngrams_size: int,
            compiler_dict_path: str, optimization_dict_path: str,
            output_path: str) -> None:

    log_message(file_handle=None,
                message="Loading dataset...")
    inputs, _ = load_dataset(path=test_path,
                             input_dict_path=input_dict_path,
                             compiler_dict_path=compiler_dict_path,
                             optimization_dict_path=optimization_dict_path,
                             ngrams_size=ngrams_size)

    helper = Preprocessor()
    dictionaries = helper.import_dictionaries(compilers_path=compiler_dict_path,
                                              optimizations_path=optimization_dict_path)

    rev_compilers = [k[0] for k in sorted(dictionaries.compilers.items(), key=lambda x: x[1])]
    rev_optimizations = [k[0] for k in sorted(dictionaries.optimizations.items(), key=lambda x: x[1])]
    del dictionaries

    log_message(file_handle=None,
                message="Predicting compilers...")
    model = load(compiler_model_path)
    compiler_preds = model.predict(inputs).tolist()
    compiler_preds = [int(p) for p in compiler_preds]
    del model

    log_message(file_handle=None,
                message="Predicting optimization levels...")
    model = load(optimization_model_path)
    optimization_preds = model.predict(inputs).tolist()
    optimization_preds = [int(p) for p in optimization_preds]
    del model, inputs

    log_message(file_handle=None,
                message="Writing output file...")
    with open(output_path, "w"):
        pass

    with open(output_path, "a") as preds_file:
        for compiler, optimization in zip(compiler_preds, optimization_preds):
            # in case the prediction is unknown compiler, fallback to random selection
            compiler = rev_compilers[compiler] if compiler != 0 else random.choice(rev_compilers)

            # in case the prediction is unknown optimization level, fallback to random selection
            optimization = rev_optimizations[optimization] if optimization != 0 else random.choice(rev_optimizations)

            log_message(file_handle=preds_file,
                        message="{c},{o}".format(c=compiler,
                                                 o=optimization),
                        with_time=False,
                        to_stdout=False)

    log_message(file_handle=None,
                message="Done.")


if __name__ == "__main__":
    predict(test_path="../data/test_dataset_blind.jsonl",
            compiler_model_path="../misc/svm_compiler.model",
            optimization_model_path="../misc/svm_optimization.model",
            input_dict_path="../data/2grams.txt",
            ngrams_size=2,
            compiler_dict_path="../data/compilers.txt",
            optimization_dict_path="../data/optimizations.txt",
            output_path="../misc/predictions.csv")
