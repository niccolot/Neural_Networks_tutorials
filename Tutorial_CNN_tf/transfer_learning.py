"""
CNN wrote using prebuilt model from keras.applications library

first the net is trained keeping all the layers but the last frozen, then is finetuned
retraining the whole model with a smaller learning rate
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from glob import glob
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras import regularizers


#project_dir = r'C:\Users\nico_\PycharmProjects\Tutorial_CNN_tf'
project_dir = os.getcwd()

try_num = 127
tl_try_dir = os.path.join(project_dir, 'tl_try_{try_num}'.format(try_num=try_num))
if os.path.exists(tl_try_dir):
    raise Exception("Directory name already used, update try_num")
else:
    os.mkdir(tl_try_dir)


img_height = 180
img_width = 180
validation_split = 0.2
batch_size = 32
learning_rate = 1e-4
lr_fine_tuning = 7e-7
drop_rate = 0.65
drop_rate_conv = 0.4
epochs = 150
epochs_finetuning = 200
l2_penalty = 1e-1


#this builds the model once fed with the base model taken from keras.applications
def build_model(base_model):

    data_augmentation = Sequential([
        layers.RandomFlip("horizontal_and_vertical", input_shape=(img_height, img_width, 3)),
        layers.RandomRotation(0.25, seed=25),
    ])

    inputs = keras.Input(shape=(img_height, img_width, 3))

    #only custom part in order to regularize the final result

    x = data_augmentation(inputs)
    #x = layers.Rescaling(scale=1 / 125.5, offset=-1)(x)
    x = keras.applications.efficientnet.preprocess_input(x)

    """
    here important to always use training=False in order
    to use the base net in inference mode, otherwise layers like
    batchnorm could destabilize the training updating batch statistics
    when retraining later
    """

    x = base_model(x, training=False)
    #x = layers.SpatialDropout2D(drop_rate_conv)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(drop_rate)(x)
    outputs = layers.Dense(len(class_names),
                           activation='softmax',
                           activity_regularizer=regularizers.L2(l2_penalty))(x)

    model = keras.Model(inputs, outputs)
    return model


physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

"""
DATASET LOADING
"""


data_dir = os.path.join(project_dir, 'flower_photos')
image_count = len(list(glob(os.path.join(data_dir, '*/*.jpg'))))

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=validation_split,
    seed=25,
    label_mode='categorical',
    subset="training",
    color_mode='rgb',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=validation_split,
    seed=25,
    label_mode='categorical',
    subset="validation",
    color_mode='rgb',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names

hyper_parameters_file_path = os.path.join(tl_try_dir, 'hyper_parameters.txt')

with open(hyper_parameters_file_path, 'a') as f:
    f.write('batch_size: %d\n' % batch_size)
    f.write('learning_rate: %e\n' % learning_rate)
    f.write('lr_fine_tuning: %e\n' % lr_fine_tuning)
    f.write('validation_split: %.2f\n' % validation_split)
    f.write('img_height: %d\n' % img_height)
    f.write('img_width: %d\n' % img_width)
    f.write('drop_rate: %.2f\n' % drop_rate)
    f.write('drop_rate_conv: %.2f\n' % drop_rate_conv)
    f.write('base model: EfficientNetB0\n')
    f.write('l2_penalty: %f\n' % l2_penalty)


train_ds = train_ds.cache().shuffle(image_count).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

"""
BASE MODEL SELECTION AND TRANSFER LEARNING
"""
"""
base_model = tf.keras.applications.InceptionResNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(img_height, img_width, 3),
)
"""

"""
base_model = keras.applications.EfficientNetV2S(
    include_top=False,
    weights="imagenet",
    input_shape=(img_height, img_width, 3),
    pooling='avg',
    include_preprocessing=True,
)
"""

base_model = keras.applications.EfficientNetB0(
    include_top = False,
    weights='imagenet',
    input_shape=(img_height, img_width, 3)
)

base_model.trainable = False


model = build_model(base_model=base_model)

model_summary_file_path = os.path.join(tl_try_dir, 'model_summary.txt')
with open(model_summary_file_path, 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))


loss = keras.losses.CategoricalCrossentropy()

model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              loss=loss,
              metrics=['accuracy'])

model_file_path = os.path.join(tl_try_dir, 'model.h5')

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=25,
                                     monitor="val_loss"),
    tf.keras.callbacks.ModelCheckpoint(filepath=model_file_path,
                                       save_best_only=True)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks,
)

"""
FINETUNING

after having trained trained the model with all but the last layer frozen one can 
retrain the whole net in order to achieve better result, important to use small
learning rate for stability issues
"""

#the model is always called with training = False for inference mode only, the
#below model.trainable = True allows to train all the other layers
base_model.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(lr_fine_tuning),
    loss=loss,
    metrics=['accuracy']
)

finetuned_model_file_path = os.path.join(tl_try_dir, 'fine_tuned_model.h5')

callbacks_fine_tuned = [
    tf.keras.callbacks.EarlyStopping(patience=10,
                                     monitor="val_loss"),
    tf.keras.callbacks.ModelCheckpoint(filepath=finetuned_model_file_path,
                                       save_best_only=True)
]

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs_finetuning,
    callbacks=callbacks_fine_tuned,
)

hist_csv_file_path = os.path.join(tl_try_dir, 'history.csv')

hist_df = pd.DataFrame(history.history)
hist_csv_file = hist_csv_file_path
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(loss))

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

hist1_csv_file_path = os.path.join(tl_try_dir, 'history1.csv')

hist1_df = pd.DataFrame(history1.history)
hist1_csv_file = hist1_csv_file_path
with open(hist1_csv_file, mode='w') as f:
    hist1_df.to_csv(f)

acc1 = history1.history['accuracy']
val_acc1 = history1.history['val_accuracy']

loss1 = history1.history['loss']
val_loss1 = history1.history['val_loss']

epochs_range1 = range(len(loss1))

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.plot(epochs_range1, acc1, label='Training Accuracy')
plt.plot(epochs_range1, val_acc1, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy finetuning')

plt.subplot(1, 2, 2)
plt.plot(epochs_range1, loss1, label='Training Loss')
plt.plot(epochs_range1, val_loss1, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss finetuning')

plt.show()
