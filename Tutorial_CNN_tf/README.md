# Tutorial_CNN_tf

CNN trained on the tensorflow flowers tutorial

-main.py is for searching a decent architecture, it creates a folder at each try with all the train logs (model.summay, hyperparameters used, train history etc)
-hyper_parameters_finetuning.py allows to train with a random search in order to find the best hyoerparameters once decided the architecture
-test.py is for testing, it gives loss, accuracy and plots confusion matrix and multiclass roc curve
