import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score

targets = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


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


physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

project_dir = r'C:\Users\nico_\PycharmProjects\Tutorial_CNN_tf'
test_data_dir = os.path.join(project_dir, 'flower_photos_test')

img_height = 180
img_width = 180

test_ds = keras.utils.image_dataset_from_directory(
    test_data_dir,
    label_mode='categorical',
    shuffle=True,
    seed=25,
    validation_split=None,
    subset=None,
    color_mode='rgb',
    image_size=(img_height, img_width)
)

model_file_path = os.path.join(project_dir, 'model.h5')

model = keras.models.load_model(model_file_path)

results = model.evaluate(test_ds)
print('\ntest_loss: %.2f' % results[0])
print('test_accuracy: %.2f%%\n' % (results[1]*100))

predictions = model.predict(test_ds)

test_labels = np.concatenate([y for x, y in test_ds], axis=0)

ConfusionMatrixDisplay.from_predictions(np.argmax(test_labels, axis=1),
                                        np.argmax(predictions, axis=1),
                                        normalize='true',
                                        display_labels=targets)

print('ROC AUC score:', multiclass_roc_auc_score(np.argmax(test_labels, axis=1),
                                                 np.argmax(predictions, axis=1)))


plt.show()



