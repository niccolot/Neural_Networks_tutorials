import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

targets = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

targets_dict = {'0': 'daisy',
                '1': 'dandelion',
                '2': 'roses',
                '3': 'sunflowers',
                '4': 'tulips'}


#returns ROCAUC score of the model and plots the ROC curve
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):

    plt.figure(figsize=(6, 6))
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    for (idx, c_label) in enumerate(targets):
        fpr, tpr, thresholds = roc_curve(y_test[:, idx].astype(int), y_pred[:, idx])
        plt.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
    plt.plot(fpr, fpr, 'b-', label='Random Guessing')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    return roc_auc_score(y_test, y_pred, average=average)


def get_errors(x_test, test_probs, pred_probs, num_errors=4):

    test_labels = np.argmax(test_probs, axis=1)
    pred_labels = np.argmax(pred_probs, axis=1)

    error_labels = (pred_labels - test_labels != 0)

    pred_probs_errors = pred_probs[error_labels]
    test_labels_errors = test_labels[error_labels]
    pred_labels_errors = pred_labels[error_labels]
    x_test_errors = x_test[error_labels]

    # Probabilities of the wrong predictions
    y_pred_errors_prob = np.max(pred_probs_errors, axis=1)

    # Predicted probabilities of the true values in the error set
    true_prob_errors = np.diagonal(np.take(pred_probs_errors, test_labels_errors, axis=1))

    # Difference between the probability of the predicted label and the true label
    delta_pred_true_errors = y_pred_errors_prob - true_prob_errors

    # Sorted list of the delta prob errors
    sorted_delta_errors = np.argsort(delta_pred_true_errors)

    # Top errors
    most_important_errors = sorted_delta_errors[-num_errors:]

    return most_important_errors, pred_probs_errors, test_labels_errors, x_test_errors, pred_labels_errors


def display_errors(errors_index, img_errors, pred_errors, obs_errors, target_dict):
    n = 0
    nrows = 2
    ncols = 2
    fig, ax = plt.subplots(nrows, ncols)
    fig.subplots_adjust(hspace=.5, wspace=.5)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row, col].imshow(img_errors[error])
            ax[row, col].set_title("Predicted label: {}\nTrue label: {}".format(target_dict[str(pred_errors[error])],
                                                                                target_dict[str(obs_errors[error])]))
            n += 1


project_dir = os.getcwd()
test_data_dir = os.path.join(project_dir, 'flower_photos_test')

img_height = 180
img_width = 180

test_ds = keras.utils.image_dataset_from_directory(
    test_data_dir,
    label_mode='categorical',
    shuffle=False,#important to switch off shuffling otherwise the labels will be get mixed resulting in a random confusion matrix and roc curve
    validation_split=None,
    subset=None,
    color_mode='rgb',
    image_size=(img_height, img_width)
)

model_file_path = """insert here the path to the .h5 file to test"""

model = keras.models.load_model(model_file_path)

results = model.evaluate(test_ds)
print('\ntest_loss: %.2f' % results[0])
print('test_accuracy: %.2f%%\n' % (results[1]*100))

predictions = model.predict(test_ds)

test_labels = np.concatenate([y for x, y in test_ds], axis=0)

ConfusionMatrixDisplay.from_predictions(np.argmax(test_labels, axis=1),
                                        np.argmax(predictions, axis=1),
                                        normalize='true',
                                        display_labels=targets,
                                        cmap='Blues')

print('ROC AUC score:', multiclass_roc_auc_score(np.argmax(test_labels, axis=1),
                                                 np.argmax(predictions, axis=1)))


x_test = np.concatenate([x for x, y in test_ds], axis=0)
x_test = x_test.astype('float32') / 255.



most_important_errors, \
pred_probs_errors, \
test_labels_errors, \
x_test_errors, \
pred_labels_errors = get_errors(x_test, predictions, test_labels)

display_errors(most_important_errors, x_test_errors, test_labels_errors, pred_labels_errors, targets_dict)
plt.show()
