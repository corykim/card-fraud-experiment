# Card Fraud Detection Experiment

This is an experiment to build a card fraud detection model using Spark MLlib.

## Data file
- The data file can be found [here](https://www.kaggle.com/datasets/kelvinkelue/credit-card-fraud-prediction)
- The data file should be installed locally at the path `./data/card-fraud/fraud-test.csv`

## Spark
This repo assumes that you have a Spark cluster available. We use a standalone cluster in Docker for dev and test purposes.

## Scripts
- `training.py`: trains the model and saves the MLlib pipeline to local file system
- `inference.py`: performs inference, using part of the training data
- `data.py`: a support module to load the data file into Spark
- `features.py`: a support module to add new features to the dataset, for the model to train on
