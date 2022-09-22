"""
hyper_parameters_tuning with random search
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras_tuner
from tensorflow import keras
from keras import layers
from keras import regularizers
from glob import glob


physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

batch_size = 32
img_height = 180
img_width = 180
epochs = 600
validation_split = 0.2

project_dir = os.getcwd()
data_dir = os.path.join(project_dir, 'flower_photos')

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
image_count = len(list(glob(os.path.join(data_dir, '*/*.jpg'))))


train_ds = train_ds.cache().shuffle(image_count).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.25, seed=25),
    layers.RandomCrop(img_height, img_width, seed=25),
    layers.GaussianNoise(1.0, seed=25)
])


def define_model(drop_rate_fc, drop_rate_conv, l2_penalty, learning_rate):
    model = keras.Sequential()
    model.add(data_augmentation)
    model.add(layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)))
    model.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
    model.add(layers.SpatialDropout2D(drop_rate_conv))
    model.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
    model.add(layers.SpatialDropout2D(drop_rate_conv))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add(layers.SpatialDropout2D(drop_rate_conv))
    model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add(layers.SpatialDropout2D(drop_rate_conv))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(layers.SpatialDropout2D(drop_rate_conv))
    model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(layers.SpatialDropout2D(drop_rate_conv))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(128, 3, padding='same', activation='relu'))
    model.add(layers.SpatialDropout2D(drop_rate_conv))
    model.add(layers.Conv2D(128, 3, padding='same', activation='relu'))
    model.add(layers.SpatialDropout2D(drop_rate_conv))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(256, 3, padding='same', activation='relu'))
    model.add(layers.SpatialDropout2D(drop_rate_conv))
    model.add(layers.Conv2D(256, 3, padding='same', activation='relu'))
    model.add(layers.SpatialDropout2D(drop_rate_conv))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu', activity_regularizer=regularizers.L2(l2_penalty)))
    model.add(layers.Dropout(drop_rate_fc))
    model.add(layers.Dense(len(class_names), activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
    )

    return model


def build_model(hp):
    drop_rate_fc = hp.Float('drop_rate_fc', min_value=0.4, max_value=0.8, step=0.1)
    drop_rate_conv = hp.Float('drop_rate_conv', min_value=0.05, max_value=0.25, step=0.05)
    l2_penalty = hp.Float('l2_penalty', min_value=1e-4, max_value=5e-3, sampling='log')
    learning_rate = hp.Float('learning_rate', min_value=5e-5, max_value=1e-3, sampling='log')

    model = define_model(drop_rate_fc=drop_rate_fc,
                         drop_rate_conv=drop_rate_conv,
                         l2_penalty=l2_penalty,
                         learning_rate=learning_rate)

    return model


build_model(keras_tuner.HyperParameters())

fine_tuning_dir = os.path.join(project_dir, 'fine_tuning')

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=30,
    executions_per_trial=1,
    overwrite=True,
    directory=fine_tuning_dir,
    seed=42
)

callbacks = [
    keras.callbacks.EarlyStopping(patience=60, monitor='val_loss'),
    keras.callbacks.TensorBoard()
]

tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks
)
