import keras
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.data import AUTOTUNE
from config import class_names, img_width, img_height, num_channels
from keras import layers


def configure_for_performance(ds, batch_size):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


# def show_image(image):
#     _ = plt.imshow(image)
#     _ = plt.title("Image")

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(factor=(-0.0277, 0.0277)),
    layers.RandomCrop(height=img_height, width=img_width)
])

rescaling = layers.Rescaling(1. / 255, input_shape=(img_height, img_width, num_channels))


def visualize(original, augmented):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1, 2, 2)
    plt.title('Augmented image')
    plt.imshow(augmented)


def prepare(ds, shuffle=False, augment=False):
    if shuffle:
        ds = ds.shuffle(1000)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    
    return ds.prefetch(buffer_size=AUTOTUNE)


def show_images_from_ds(ds, labels=None):
    plt.figure(figsize=(10, 10))
    for images in ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(labels[i])
            plt.axis("off")
    plt.show()


def show_images_from_ds_with_labels(ds):
    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(12):
            ax = plt.subplot(3, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[np.argmax(labels[i])])
            plt.axis("off")
    plt.show()
