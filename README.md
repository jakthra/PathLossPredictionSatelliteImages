This repository contains the code from the work detailed in the paper submitted to IEEE Access

```
@article{Thrane2020,
    author       = {Jakob Thrane, Darko Zibar, Henrik L. Christiansen},
    title        = {{Model-aided Deep Learning Method for Path Loss Prediction in Mobile Communication Systems at 2.6 GHz}},
    month        = Jan,
    year         = 2020,
    publisher    = {IEEE},
    journal      = {IEEE Access}
}
```

Previous work is detailed in:

```
@article{Thrane2018,
    author      = {Thrane, Jakob and Artuso, Matteo and Zibar, Darko and Christiansen, Henrik L},
    journal     = {VTC 2018 Fall},
    publisher   = {IEEE}
    title       = {{Drive test minimization using Deep Learning with Bayesian approximation}},
    year        = {2018}
}

```

# Instructions for the code

1. Download the dataset from 
https://ieee-dataport.org/open-access/mobile-communication-system-measurements-and-satellite-images

2. Put the raw data into the `raw_data` folder
    * such that the data is located in:
    * `raw_data\feature_matrix.csv`
    * `raw_data\output_matrix.csv`
    * `raw_data\mapbox_api\*.png`
3. Generate the test and training set using `generate_training_test.py`
4. Run the training of the model by `train.py`, see the script for commandline arguments
