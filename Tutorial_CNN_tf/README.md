# Tutorial_CNN_tf

CNN trained on the tensorflow flowers dataset

-main.py is for searching a decent architecture, it creates a folder at each try with all the train logs (model.summay, hyperparameters used, train history etc)

-hyper_parameters_finetuning.py allows to train with a random search in order to find the best hyoerparameters once decided the architecture

-test.py is for testing, it gives loss, accuracy and plots confusion matrix and multiclass roc curve

-transfer_learning.py build a classifier based on a pretrained net from keras.applications and comprare the result with the custom naive implementation

## Custom CNN

A standard architecture for a CNN, as one can see from the images it reaches around 85% accuracy in validation, decreasing to a not so great 77% at test time

## Transfer learning

Using the pretrained EfficientNetB0 (https://arxiv.org/abs/1905.11946), the model has firstly been trained with all the layers but the last frozen, than it has been fine tuned unfreezing all the layers and retraining averything starting from the previous weights and with a very low learning rate for stability reasons

## ToDo



