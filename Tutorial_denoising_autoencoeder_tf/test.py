"""
This code checks the output of the autoencoder
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import cv2
import keras
from glob import glob


img_width = 512
img_height = 512
validation_split = 0.25

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


test_dir = os.path.join(project_dir, 'test')
test_data = glob(os.path.join(test_dir, '*.png'))
test_images = read_rescale_images(test_data)
test_arr = prep_images(test_images)


#put here best .h5 model
autoencoder = keras.models.load_model()

#extract one random dirty imge from the test set
np.random.seed(int(time.time()))
x = np.random.randint(0, len(test_arr)-1)
fig_to_clean = np.expand_dims(test_arr[x], axis=0)
pred = autoencoder.predict(fig_to_clean)

plt.figure(figsize=(15, 15))

plt.subplot(1, 2, 1)
plt.imshow(test_arr[x], cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(pred[x], cmap='gray')
plt.axis('off')

plt.show()
