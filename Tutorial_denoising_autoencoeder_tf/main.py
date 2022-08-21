"""
Convolutional autoencoder used to clean artificially 'dirty' documents

the trainset is a collection of clean images (ground truth) with the correspondant artificially ruined ones
dataset url https://www.kaggle.com/competitions/denoising-dirty-documents/data
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import matplotlib.image
import tensorflow as tf
import numpy as np
import cv2
import keras
from keras import layers
from glob import glob
from sklearn.model_selection import train_test_split


batch_size = 4
img_width = 512
img_height = 512
validation_split = 0.25
epochs = 300
learning_rate = 1e-3
input_img = layers.Input(shape=(img_height,img_width, 1))


def read_rescale_images(data):
    images = []
    for i in range(len(data)):
        img = matplotlib.image.imread(data[i])
        img = cv2.resize(img, (img_height, img_width))
        images.append(img)
    return images


def prep_images(x):

    x = np.asarray(x)
    x = x.astype('float32')
    x = x/np.max(x)
    x = x.reshape(-1, img_height, img_width, 1)
    return x


project_dir = r'C:\Users\nico_\PycharmProjects\Tutorial_denoising_autoencoder_tf'

#for every run the program will create a subfolder with all the data(model summary, graph,
# hyperparameters etc, update manually try num at every num)
try_num = 52

try_dir = os.path.join(project_dir, 'try_{try_num}'.format(try_num=try_num))
if os.path.exists(try_dir):
    raise Exception("Directory name already used, update try_num")
else:
    os.mkdir(try_dir)


train_noisy_dir = os.path.join(project_dir, 'train')
train_cleaned_dir = os.path.join(project_dir, 'train_cleaned')

#glob returns an iterable of all the .png files in order to feed it to the data preparation functions
train_noisy_data = glob(os.path.join(train_noisy_dir, '*.png'))
train_cleaned_data = glob(os.path.join(train_cleaned_dir, '*.png'))

train_noisy_images = read_rescale_images(train_noisy_data)
train_cleaned_images = read_rescale_images(train_cleaned_data)

train_noisy_arr = prep_images(train_noisy_images)
train_cleaned_arr = prep_images(train_cleaned_images)

train_noisy, val_noisy, train_cleaned, val_cleaned = train_test_split(train_noisy_arr,
                                                                      train_cleaned_arr,
                                                                      test_size=validation_split,
                                                                      random_state=25)

#in order to visualize some images
"""
for i in range(5):
  plt.subplot(3, 5, i+1)
  curr_img = train_noisy[i]
  plt.imshow(curr_img, cmap='gray')
  plt.axis("off")

for i in range(5):
  plt.subplot(3, 5, i+6)
  curr_img = train_cleaned[i]
  plt.imshow(curr_img, cmap='gray')
  plt.axis("off")

plt.show()
"""

def autoencoder(input_img):
    #encoder
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    pool4 = layers.MaxPooling2D(pool_size=(2,2))(conv4)
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    pool5 = layers.MaxPooling2D(pool_size=(2,2))(conv5)
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool5)


    #decoder
    conv7 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    up1 = layers.UpSampling2D((2,2))(conv7)
    conv8 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(up1)
    up2 = layers.UpSampling2D((2,2))(conv8)
    conv9 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up2)
    up3 = layers.UpSampling2D((2, 2))(conv9)
    conv10 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up3)
    up4 = layers.UpSampling2D((2, 2))(conv10)
    conv11 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up4)
    up5 = layers.UpSampling2D((2, 2))(conv11)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up5)

    return decoded

#for autoencoders binary_cross entropy is still a suitable loss function, tho not minimized exactly by 0 being the
#inputs not binary but in the range [0, 1]
autoencoder = keras.models.Model(input_img, autoencoder(input_img))
autoencoder.compile(loss='binary_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    metrics=['mae'])

autoencoder.summary()

model_file_path = os.path.join(try_dir, 'model.h5')
callbacks = [tf.keras.callbacks.EarlyStopping(patience=30, monitor="val_loss"),
             tf.keras.callbacks.ModelCheckpoint(filepath=model_file_path,
                save_best_only=True)
]

autoencoder_train = autoencoder.fit(train_noisy,train_cleaned,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(val_noisy, val_cleaned),
                            callbacks = callbacks)

loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
mae = autoencoder_train.history['mae']
epochs_range = range(len(loss))

plt.figure(figsize=(5, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, mae, label='MAE')
plt.legend(loc='upper right')
plt.title('MAE')

plt.show()

hyper_parameters_file_path = os.path.join(try_dir, 'hyper_parameters.txt')
model_summary_file_path = os.path.join(try_dir, 'model_summary.txt')

with open(hyper_parameters_file_path, 'a') as f:
    f.write('batch_size: %d\n' % batch_size)
    f.write('learning_rate: %f\n' % learning_rate)
    f.write('epochs: %d\n' % epochs)
    f.write('validation_split: %.2f\n' % validation_split)
    f.write('img_height: %d\n' % img_height)
    f.write('img_width: %d\n' % img_width)

with open(model_summary_file_path, 'a') as f:
    autoencoder.summary(print_fn=lambda x: f.write(x + '\n'))
