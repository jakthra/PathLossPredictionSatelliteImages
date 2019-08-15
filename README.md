This repository contains the code from the work detailed in the paper submitted to Transactions on Wireless Communications

```
@article{thrane_pathloss_prediction_2019,
    author       = {Jakob Thrane, Darko Zibar, Henrik L. Christiansen},
    title        = {{Path Loss Prediction using Deep Learning utilizing Satellite Images for 5G Mobile Communication Systems}},
    month        = Aug,
    year         = 2019,
    publisher    = {IEEE},
    journal      = {IEEE Transactions on Wireless Communications}
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