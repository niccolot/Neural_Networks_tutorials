import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from focal_loss import SparseCategoricalFocalLoss
import pandas as pd
import numpy as np
import time

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

project_dir = r'/'

try_num = 17
try_dir = os.path.join(project_dir, 'dlv3plus_resnet50_try_{try_num}'.format(try_num=try_num))
if os.path.exists(try_dir):
    raise Exception("Directory name already used, update try_num")
else:
    os.mkdir(try_dir)

image_size = 512
batch_size = 2
seed = 25
drop_rate_conv = 0.2
learning_rate = 1e-5
gamma_loss = 2.0
epochs = 60
num_train_images = 500
val_split = 0.2
num_val_images = int(num_train_images*val_split)
num_test_images = 20

hyper_parameters_file_path = os.path.join(try_dir, 'hyper_parameters.txt')

with open(hyper_parameters_file_path, 'w') as f:
    f.write('batch_size: %d\n' % batch_size)
    f.write('image_size: %d\n' % image_size)
    f.write('drop_rate_conv: %.2f\n' % drop_rate_conv)
    f.write('gamma: %2f\n' % gamma_loss)

"""
#IMAGE MANIPULATION FUNCTIONS
"""


def resize(input_image, input_mask):
    input_image = tf.image.resize(input_image, (image_size, image_size), method="nearest")
    input_mask = tf.image.resize(input_mask, (image_size, image_size), method="nearest")
    return input_image, input_mask


def augment(input_image, input_mask):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_up_down(input_image)
        input_mask = tf.image.flip_up_down(input_mask)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.rot90(input_image)
        input_mask = tf.image.rot90(input_mask)

    return input_image, input_mask


def normalize(input_image, input_mask):
   input_image = tf.cast(input_image, tf.float32) / 255.0
   input_mask -= 1
   return input_image, input_mask


def load_image_train(datapoint):
    input_image = datapoint["image"]
    input_mask = datapoint["segmentation_mask"]
    input_image, input_mask = resize(input_image, input_mask)
    input_image, input_mask = augment(input_image, input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


def load_image_test(datapoint):
    input_image = datapoint["image"]
    input_mask = datapoint["segmentation_mask"]
    input_image, input_mask = resize(input_image, input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ["Input Image", "True Mask", "Predicted Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(model, sample_image, sample_mask):
    display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])


"""
#MODEL BUILDING FUNCTIONS
"""

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
    )(block_input)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = layers.SpatialDropout2D(drop_rate_conv)(x)
    return x


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    x = layers.SpatialDropout2D(drop_rate_conv)(x)
    output = convolution_block(x, kernel_size=1)
    return output


def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = layers.SpatialDropout2D(drop_rate_conv)(x)
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    x = layers.SpatialDropout2D(drop_rate_conv)(x)
    model_output = layers.Conv2D(num_classes, 1, padding="same", activation="softmax")(x)
    return keras.Model(inputs=model_input, outputs=model_output)


"""
#TRAINING
"""

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

train_dataset = dataset["train"].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = dataset["test"].map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)

#take all the train images (circa 3000) and 100 images for validation from the test dataset, then 200 test images
train_batches = train_dataset.cache().shuffle(1000).batch(batch_size)
train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_batches = test_dataset.take(1000).batch(batch_size)
test_batches = test_dataset.skip(1000).take(200).batch(batch_size)

model = DeeplabV3Plus(image_size, 3)

loss = SparseCategoricalFocalLoss(gamma=gamma_loss)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=loss,
              metrics="accuracy")

model_summary_file_path = os.path.join(try_dir, 'model_summary.txt')
with open(model_summary_file_path, 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

model_file_path = os.path.join(try_dir, 'model.h5')

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=6,
                                     monitor="val_loss"),
    tf.keras.callbacks.ModelCheckpoint(filepath=model_file_path,
                                       save_best_only=True)
]

history = model.fit(train_batches,
                    epochs=epochs,
                    validation_data=validation_batches,
                    callbacks=callbacks,
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

hist_csv_file_path = os.path.join(try_dir, 'history.csv')
hist_df = pd.DataFrame(history.history)
hist_csv_file = hist_csv_file_path
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

"""
#PLOTTING TRAIN CURVE AND SEGMENTATION SAMPLES
"""

epochs_range = range(len(loss))


plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training acc')
plt.plot(epochs_range, val_acc, label='Validation acc')
plt.legend(loc='lower right')
plt.title('Training and Validation Metric')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training FocalLoss')
plt.plot(epochs_range, val_loss, label='Validation FocalLoss')
plt.legend(loc='upper right')
plt.title('Training and Validation FocalLoss')


sample_batch = next(iter(test_batches))
np.random.seed(seed=int(time.time()))
random_index = np.random.choice(sample_batch[0].shape[0])
sample_image, sample_mask = sample_batch[0][random_index], sample_batch[1][random_index]

show_predictions(model=model, sample_image=sample_image, sample_mask=sample_mask)

plt.show()
