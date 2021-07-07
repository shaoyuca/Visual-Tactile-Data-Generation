import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import librosa.display
import pandas as pd
import seaborn as sns
import pandas
import datetime


P = "/home/shaoyu/Desktop/ral_code/t2v/generated_images/"
x0 = tf.constant([os.path.join(P, "%d.jpg" % i) for i in range(0, 200)]) # aylic
x1 = tf.constant([os.path.join(P, "%d.jpg" % i) for i in range(200, 400)]) # compress
x2 = tf.constant([os.path.join(P, "%d.jpg" % i) for i in range(400, 600)]) # carboil
x3 = tf.constant([os.path.join(P, "%d.jpg" % i) for i in range(600, 800)]) # carpet
x4 = tf.constant([os.path.join(P, "%d.jpg" % i) for i in range(800, 1000)]) # finefoam
x5 = tf.constant([os.path.join(P, "%d.jpg" % i) for i in range(1000, 1200)]) # marble
x6 = tf.constant([os.path.join(P, "%d.jpg" % i) for i in range(1200, 1400)]) # rubber
x7 = tf.constant([os.path.join(P, "%d.jpg" % i) for i in range(1400, 1600)]) # squared
x8 = tf.constant([os.path.join(P, "%d.jpg" % i) for i in range(1600, 1800)]) # leather


valid_filename = tf.concat([x0, x1, x2, x3, x4, x5, x6, x7, x8], axis=-1)
y0 = tf.fill(x0.shape, 0)
y1 = tf.fill(x1.shape, 1)
y2 = tf.fill(x2.shape, 2)
y3 = tf.fill(x3.shape, 3)
y4 = tf.fill(x4.shape, 4)
y5 = tf.fill(x5.shape, 5)
y6 = tf.fill(x6.shape, 6)
y7 = tf.fill(x7.shape, 7)
y8 = tf.fill(x8.shape, 8)
valid_labels = tf.concat([y0, y1, y2, y3, y4, y5, y6, y7, y8], axis=-1)


def _decode_and_resize(filename, label):
    input_image = tf.io.read_file(filename)
    input_image = tf.image.decode_jpeg(input_image, channels=1)
    input_image = tf.concat([input_image, input_image, input_image], axis=2)
    input_image = tf.cast(input_image, tf.float32)
    image_resized = input_image / 255.0


    return image_resized, label


valid_dataset = tf.data.Dataset.from_tensor_slices((valid_filename, valid_labels))
valid_dataset = valid_dataset.map(map_func=_decode_and_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
valid_dataset = valid_dataset.batch(1)

readable_labels = ["M1", "M2", "M3", "M4",  "M5", "M6", "M7", "M8", "M9"]



shape = (256, 256, 3)
base_model = tf.keras.applications.DenseNet121(input_shape=shape, include_top=False, weights='imagenet')
base_model.trainable = False
flatten = tf.keras.layers.GlobalAveragePooling2D()
dropout = tf.keras.layers.Dropout(0.9)
dense_2 = tf.keras.layers.Dense(128, activation='relu')
prediction_layer = tf.keras.layers.Dense(9)

model = tf.keras.Sequential([
  base_model,
  flatten,
  dropout,
  dense_2,
  prediction_layer])

# compile new model
model.summary()
print(len(model.trainable_variables))

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr = base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

base_model.trainable = False
model.load_weights('./pretrain/densevisual.h5')

# evaluation
loss1, accuracy1 = model.evaluate(valid_dataset)
print(accuracy1)
print("initial loss: {:.2f}".format(loss1))
print("initial accuracy: {:.2f}".format(accuracy1))



