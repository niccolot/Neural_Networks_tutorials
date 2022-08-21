"""
CNN for tensorflow flowers dataset classification
dataset url = http://download.tensorflow.org/example_images/flower_photos.tgz
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from glob import glob
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential

#it allocates VRAM gradually and not all at once
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#write here your path to this project
project_dir = r'C:\Users\nico_\PycharmProjects\Tutorial_CNN_tf'

data_dir = os.path.join(project_dir, 'flower_photos')

#total number of train/valid images, glob returns an iterable over all the .jpg files
image_count = len(list(glob(os.path.join(data_dir, '*/*.jpg'))))

#for every run the program will create a subfolder with all the data(model summary, graph,
# hyperparameters etc, update manually try num at every num)
try_num = 47

dir = os.path.join(project_dir, 'try_{try_num}'.format(try_num=try_num))
if os.path.exists(dir):
    raise Exception("Directory name already used, update try_num")
else:
    os.mkdir(dir)

#hyper_parameters
batch_size = 32
img_height = 224
img_width = 224
drop_rate_fc = 0.6
learning_rate = 1e-5
epochs = 550
validation_split = 0.2
l2_penalty = 1e-2
augment_param = 0.4

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

#in order to check some images
'''
for im_batch, labels_batch in train_ds:
    print(im_batch.shape)
    print(labels_batch.shape)
    break

plt.figure(figsize=(5, 5))
for images, labels in train_ds.take(1):
  for i in range(len(train_ds.class_names)):
    ax = plt.subplot(2, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()
'''

#this speed up the loading optimizing training time
# .cache() caches the dataset in order to have to open the file only for the first epoch
# .shuffle() shuffle the data, the argument is how many images to shuffle every epoch and so
# it must be greater or equal to the train set, here is set equal to the entirety of the dataset
# .prefetch() is to load the next batch while the current is training
train_ds = train_ds.cache().shuffle(image_count).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(augment_param, seed=25),
    layers.RandomZoom(augment_param, seed=25)
])

model = Sequential([
    data_augmentation,
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(32, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(256, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(256, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(512, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(512, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.L2(l2_penalty)),
    layers.Dropout(drop_rate_fc),
    layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.L2(l2_penalty)),
    layers.Dropout(drop_rate_fc),
    layers.Dense(5, activation='softmax')
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model_file_path = os.path.join(project_dir, 'try_{try_num}'.format(try_num=try_num), 'model.h5')

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=15,
                                     monitor="val_loss"),
    tf.keras.callbacks.ModelCheckpoint(filepath=model_file_path,
                                       save_best_only=True)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks
)

hist_csv_file_path = os.path.join(project_dir, 'try_{try_num}'.format(try_num=try_num), 'history_csv')

hist_df = pd.DataFrame(history.history)
hist_csv_file = hist_csv_file_path
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(loss))

plt.figure(figsize=(5, 5))
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
plt.show()

model_summary_file_path = os.path.join(project_dir, 'try_{try_num}'.format(try_num=try_num), 'model_summary.txt')
hyper_parameters_file_path = os.path.join(project_dir, 'try_{try_num}'.format(try_num=try_num), 'hyper_parameters.txt')

with open(model_summary_file_path, 'a') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

with open(hyper_parameters_file_path, 'a') as f:
    f.write('batch_size: %d\n' % batch_size)
    f.write('drop_rate_fc: %.2f\n' % drop_rate_fc)
    f.write('learning_rate: %f\n' % learning_rate)
    f.write('epochs: %d\n' % epochs)
    f.write('validation_split: %.2f\n' % validation_split)
    f.write('img_height: %d\n' % img_height)
    f.write('img_width: %d\n' % img_width)
    f.write('l2_penalty: %f\n' % l2_penalty)
    f.write('augment_param: %f\n' % augment_param)
    f.write('used kernel_reg\n')
    f.write('------------\n')
