#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import cv2
import glob
import os

import keras
from keras.layers import Dense, Dropout, Activation, Flatten, AveragePooling2D, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential

def read_image(filename):
    im = cv2.imread(filename)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None else None

normalized_h = 128
normalized_w = 64

base = os.getcwd()
train_green = base + "/traffic_light_images/training/green"
train_yellow = base + "/traffic_light_images/training/yellow"
train_red = base + "/traffic_light_images/training/red"

test_green = base + "/traffic_light_images/test/green"
test_yellow = base + "/traffic_light_images/test/yellow"
test_red = base + "/traffic_light_images/test/red"

xs_train = []
ys_train = []

xs_test = []
ys_test = []

for idx, name in enumerate([train_green, train_yellow, train_red]):
    for file_name in glob.glob(name + "/*.jpg"):
        img = read_image(file_name)
        img = cv2.resize(img, (normalized_w, normalized_h))
        xs_train.append(img)
        ys_train.append(idx)
        
for idx, name in enumerate([test_green, test_yellow, test_red]):
    for file_name in glob.glob(name + "/*.jpg"):
        img = read_image(file_name)
        img = cv2.resize(img, (normalized_w, normalized_h))
        xs_test.append(img)
        ys_test.append(idx)

xs_train = np.stack(xs_train, axis=0)
xs_test = np.stack(xs_test, axis=0)


ys_train = to_categorical(ys_train)
ys_test = to_categorical(ys_test)

# Adapted from Keras documentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.05,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.05,
        shear_range=0.05,  # set range for random shear
        zoom_range=0.05,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
    )
     #   validation_split=0.2)


datagen.fit(xs_train)

model = Sequential(name="tl_net")

model.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', input_shape=(normalized_h, normalized_w, 3)))
model.add(Dropout(rate=0.5))
model.add(AveragePooling2D())


model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(rate=0.5))
model.add(AveragePooling2D())

model.add(Flatten())

model.add(Dense(units=120, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=3, activation = 'softmax'))

opt = keras.optimizers.Adam(decay=1e-4)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

print(model.summary())

epochs = 3
batch_size = 16

model.fit_generator(
    generator=datagen.flow(xs_train, ys_train,
                 batch_size=batch_size),
    epochs=epochs,
    validation_data=(xs_test, ys_test),
    workers=4)


model_name = 'tl_net'
model_path = os.path.join(os.getcwd(), model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(xs_test, ys_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


from time import time

xs_test_one_example = xs_test[0, ...]
xs_test_one_example = np.expand_dims(xs_test_one_example, 0)

start = time()
model.predict(xs_test_one_example)
duration = time() - start


# In[50]:


print("Predicting one image took %f ms" % (duration * 1000))


# In[52]:


xs_train.shape


# In[ ]:





