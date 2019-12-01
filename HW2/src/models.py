"""
Model definitions.
"""

import tensorflow as tf
from keras.losses import categorical_crossentropy
import keras
from keras import Model, Input
from keras.layers import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers.local import LocallyConnected2D
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.nasnet import NASNetMobile, NASNetLarge
from typing import List, Tuple


def get_EmaNet_model(input_shape: Tuple[int, int], num_classes: int,
                     convolutions: int = 1, dense_layers: int = 3,
                     dropout_rate: float = 0.3) -> Model:

    input_image = Input(shape=input_shape)

    # convolutional stack
    conv = Conv2D(filters=32,
                  kernel_size=(2, 2),
                  padding="same",
                  activation="relu")(input_image)

    for i in range(1, convolutions):
        conv = Conv2D(filters=16,
                      kernel_size=(1+i, 1+i),
                      padding="same",
                      activation="relu")(conv)

    conv = MaxPooling2D(pool_size=(2, 2),
                        padding="valid")(conv)

    # deconvolutional stack
    deconv = Conv2DTranspose(filters=32,
                             kernel_size=(2, 2),
                             padding="same",
                             activation="relu")(input_image)

    for i in range(1, convolutions):
        deconv = Conv2DTranspose(filters=16,
                                 kernel_size=(1+i, 1+i),
                                 padding="same",
                                 activation="relu")(deconv)

    deconv = MaxPooling2D(pool_size=(2, 2),
                          padding="valid")(deconv)

    # merge the two stacks
    merged = keras.layers.concatenate([conv, deconv])

    merged = Conv2D(filters=16,
                               kernel_size=(3, 3),
                               padding="valid",
                               activation="relu")(merged)
    merged = MaxPooling2D(pool_size=(2, 2),
                          padding="valid")(merged)

    merged = Conv2D(filters=16,
                    kernel_size=(3, 3),
                    padding="valid",
                    activation="relu")(merged)
    merged = MaxPooling2D(pool_size=(2, 2),
                          padding="valid")(merged)

    # final fully-connected layers
    dense = Flatten()(merged)

    for i in range(1, dense_layers+1):
        dense = Dense(units=int(64/i),
                      activation="relu")(dense)
        dense = Dropout(rate=dropout_rate)(dense)

    output_dense = Dense(units=num_classes,
                         activation="relu")(dense)

    model = Model(input=input_image, output=output_dense)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer="adam",
                  metrics=["accuracy"])

    return model


def get_transfer_learning_model():
    pass


def test():
    model_input = Input(shape=shape)
    vgg16 = NASNetLarge(weights="imagenet", include_top=False, input_tensor=model_input)
    vgg16 = vgg16.output
    back_model = Model(input=model_input, output=vgg16)
    back_optimizer = "adam"
    back_model.compile(loss=categorical_crossentropy,
                       optimizer=back_optimizer,
                       metrics=["accuracy"])

    vgg16_layer = "block5_pool"
    back_output = back_model.get_layer(name=vgg16_layer).output
    flatten = Flatten()(back_output)
    dense = Dense(
        units=train_gen.num_classes,
        activation="softmax"
    )(flatten)

    model = Model(input=model_input, output=dense, name="TestNet")
    optimizer = "adam"
    model.compile(loss=categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=["accuracy"])

    model.summary()


if __name__ == "__main__":
    train = "../data/MWI-Dataset-1.1.1_400/"
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_gen = train_datagen.flow_from_directory(directory=train,
                                                  target_size=(480, 640),
                                                  color_mode="rgb",
                                                  batch_size=64,
                                                  class_mode="categorical",
                                                  shuffle=True)

    shape = train_gen.image_shape
    num_classes = train_gen.num_classes
    print("There are {n} samples of shape {s} over {c} classes.".format(n=train_gen.n,
                                                                        s=shape,
                                                                        c=num_classes))

    m = get_EmaNet_model(input_shape=shape,
                         num_classes=num_classes,
                         convolutions=5,
                         dense_layers=5)
    m.summary()
