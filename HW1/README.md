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

## Directory structure

```
- data                          # general data directory, containing training and test sets
    |__ 2grams.txt              # input dictionary of 2grams over mnemonics
    |__ compilers.txt           # input/output dictionary of compilers
    |__ mnemonics.txt           # input dictionary of mnemonics
    |__ optimizations.txt       # input/output dictionary of optimizations
- src                           # source file directory
    |__ models.py               # model definitions
    |__ preprocess.py           # pre-processing utilities
    |__ test.py                 # model evaluation and output printing
    |__ train.py                # model training and parameter tuning
- main.py                       # main Python script to run the homework
- README.md                     # this file
```
