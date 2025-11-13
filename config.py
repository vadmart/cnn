import os
import tensorflow as tf
from keras.preprocessing import image_dataset_from_directory

base_dir = './dataset'
train_dir = os.path.join(base_dir, 'seg_train', 'seg_train')
test_dir = os.path.join(base_dir, 'seg_test', 'seg_test')
pred_dir = os.path.join(base_dir, 'seg_pred', 'seg_pred')

batch_size = 32
img_height = 150
img_width = 150
num_channels = 3

ncols = 4
nrows = 1
num_images = 4

AUTOTUNE = tf.data.AUTOTUNE

train_ds = image_dataset_from_directory(train_dir, image_size=(img_height, img_width), batch_size=batch_size,
                                        label_mode='categorical')

validation_ds = image_dataset_from_directory(test_dir, image_size=(img_height, img_width),
                                             label_mode='categorical')

pred_ds = image_dataset_from_directory(pred_dir,
                                       image_size=(img_height, img_width),
                                       labels=None,
                                       shuffle=False)

class_names = train_ds.class_names
