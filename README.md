# NCTracer

## Overview

We use sbt to compile our source code.

## How to use makefile

Use `make alone` to train on small data set locally.

Make sure that you copy training data under input/training folder and validation data under input/validation folder. For example, we took 1/10 of image1, image2, image3, image6 as training data, and use 1/10 of image4 as validation data.

Use `make alone-prediction` to predict on small data set locally.

Make sure that you first copy your input file into input/test folder, and run `make alone` to get models created in models folder. The prediction result is stored in output/prediction_result folder.

In summary, we have three subfolders under input folder.

1. input/test, used as input for prediction.
2. input/training, used as input for training.
3. input/validation, used as input for training.

The output folder of training is input/models, which is used as input for prediction, and output folder of prediction is output/prediction_result