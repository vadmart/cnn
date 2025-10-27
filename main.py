import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# for dirpath, dirnames, filenames in os.walk("./dataset"):
#     print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

base_dir = './dataset'
train_dir = os.path.join(base_dir, 'seg_train', 'seg_train')
test_dir = os.path.join(base_dir, 'seg_test', 'seg_test')

# Training Directory Path
train_mountain_dir = os.path.join(train_dir, 'mountain')
train_street_dir = os.path.join(train_dir, 'street')
train_buildings_dir = os.path.join(train_dir, 'buildings')
train_sea_dir = os.path.join(train_dir, 'sea')
train_forest_dir = os.path.join(train_dir, 'forest')
train_glacier_dir = os.path.join(train_dir, 'glacier')

batch_size = 16
img_height = 150
img_width = 150

ncols = 4
nrows = 1
num_images = 4

AUTOTUNE = tf.data.AUTOTUNE

train_ds = (image_dataset_from_directory(train_dir, image_size=(img_height, img_width), batch_size=batch_size,
                                         label_mode='categorical')
            .cache()
            .shuffle(1000)
            .prefetch(tf.data.AUTOTUNE))

validation_ds = (image_dataset_from_directory(test_dir, image_size=(img_height, img_width), batch_size=batch_size,
                                              label_mode='categorical')
                 .cache()
                 .prefetch(tf.data.AUTOTUNE))


def plot_loss_curves(history):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(history.history['loss']))

    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


# def view_random_image(target_dir):
#     fig = plt.gcf()
#     fig.set_size_inches(ncols * 4, nrows * 4)
#
#     random_image = random.sample(os.listdir(target_dir), num_images)
#
#     for i, img_path in enumerate(random_image):
#         ax = plt.subplot(nrows, ncols, i + 1)
#         ax.axis('off')
#
#         img = mpimg.imread(os.path.join(target_dir, img_path))
#         plt.imshow(img)
#         plt.title(os.path.basename(target_dir))
#
#     plt.show()


model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(6, activation='softmax')
])
model.summary()
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss')

history = model.fit(train_ds,
                    validation_data=validation_ds,
                    epochs=10)
plot_loss_curves(history)