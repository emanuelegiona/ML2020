# HW1: Compiler provenance

Given the binary code of a function as a list of instructions,
the goal is to detect which compiler has produced it, and at which
optimization level.

For this homework, the compilers are the following:

- gcc;
- icc;
- clang;

while the optimization levels are two:

- high;
- low.

[Read the report][report]

## Feature engineering
Each function in the dataset can either be represented with a Bag-of-Words
model built from the different mnemonics found in the dataset, or
using an ngram model still built from the different mnemonics.
In both the representations, the input vector contains the number of
occurrences of the corresponding feature (be it the single mnemonic or
the ngram one).

Additionally, the input vector also contains the number of instructions
contained in the given function as last dimension.

## Models used
The models compared when solving this homework are two:

- SVM with linear kernel
- K-Nearest Neighbors

both implementations are from the [scikit-learn][sklearn] library.

[Read the report][report]

## Directory structure

```
- data                          # general data directory, containing training and test sets
    |__ 2grams.txt              # input dictionary of 2grams over mnemonics
    |__ compilers.txt           # input/output dictionary of compilers
    |__ mnemonics.txt           # input dictionary of mnemonics
    |__ optimizations.txt       # input/output dictionary of optimizations
- src                           # source file directory
    |__ models.py               # model definitions
    |__ predict.py              # prediction script
    |__ preprocess.py           # pre-processing utilities
    |__ train.py                # model training, parameter tuning, and model evaluation
- README.md                     # this file
- anonymous_report.pdf          # PDF report for this homework
```

[sklearn]: https://scikit-learn.org/
[report]: ./anonymous_report.pdf
