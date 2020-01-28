# ML2020
Projects developed as homework for the [Machine Learning][ml2019] course (A.Y. 2019/2020).

## [Homework 1][hw1]

**Focus:**
compiler provenance problem

**Task Description:**
given the binary code of a function, determine the compiler that produced it, and also
the optimization level used

**ML models used**:
linear-kernel SVM, K-nearest-neighbors; both models have been separately trained using
two different feature engineering processes:

- simple Bag-of-Words model

    &forall; f &isin; &#x2115;<sup>D</sup> s.t.
    &forall; i &isin; \[0, D] f<sub>i</sub> = |{ m<sub>i</sub> in f | m<sub>i</sub> &isin; V}|
    
    with f<sub>i</sub> representing the occurrences of mnemonic m<sub>i</sub> in function f, V representing
    the set of unique mnemonics present in the whole dataset s.t. |V| = D

- ngram Bag-of-Words model

    &forall; f &isin; &#x2115;<sup>D</sup> s.t.
    &forall; i &isin; \[0, D] f<sub>i</sub> = |{ ng<sub>i</sub> in f | m<sub>i</sub> &isin; G}|
    
    with f<sub>i</sub> representing the occurrences of ngram ng<sub>i</sub> in function f, G representing
    the set of unique ngrams present in the whole dataset s.t. |G| = D
    
    _Ngram of size 2 example:_ `mov;call` encodes the occurrence of `mov` and `call` in this exact order in a function.

**[Click here for the report][hw1_report]**

## [Homework 2][hw2]

**Focus:**
weather classification

**Task Description:**
given an image, determine which is the weather condition depicted in it

**ML models used**:
custom CNN model exploiting two separate stacks of convolutional and deconvolutional operations, later joined to
leverage fully-connected layers; pre-trained CNN models used as feature extraction modules, then feeding their 
outputs into fully-connected layers.

Both the models present a highly tunable interface, fully compatible with the Keras framework.

**[Click here for the report][hw2_report]**


[ml2019]: https://sites.google.com/a/diag.uniroma1.it/ml2019/
[hw1]: ./HW1
[hw2]: ./HW2
[hw1_report]: ./HW1/anonymous_report.pdf
[hw2_report]: ./HW2/anonymous_report.pdf
