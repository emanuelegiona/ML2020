# HW2: Multi-class weather classification

Given an image (with no specific constraints), the goal is to
assign the weather condition affecting it.

For this homework, the weather conditions are the following:

- haze;
- rainy;
- snowy;
- sunny.

The dataset being used is the [MWI Dataset][mwi_dataset], used in whole
or parts of it during the training and/or validation phases.

## Directory structure

```
- data                          # general data directory, containing training and test sets
- src                           # source file directory
    |__ models.py               # model definitions
    |__ predict.py              # prediction script
    |__ preprocess.py           # pre-processing utilities
    |__ train.py                # model training, parameter tuning, and model evaluation
- README.md                     # this file
```

[mwi_dataset]: https://mwidataset.weebly.com
[report]: ./anonymous_report.pdf
