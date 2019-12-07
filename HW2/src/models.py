"""
Model definitions.
"""

import tensorflow as tf
import keras
from keras import Model, Input
from keras.layers import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.nasnet import NASNetMobile, NASNetLarge
from keras.optimizers import Adam
from typing import Tuple
from enum import Enum, auto
from math import sqrt


class FeatureExtractor(Enum):
    """
    Enumerator class representing the choices for feature extractor models.
    """

    # --- -------- ---
    # --- DenseNet ---
    # --- -------- ---
    Dense121 = auto()
    Dense169 = auto()
    Dense201 = auto()

    # --- ------ ---
    # --- NASNet ---
    # --- ------ ---
    NASNetMobile = auto()
    NASNetLarge = auto()

    @staticmethod
    def get(name: str):
        if name == "Dense121":
            return FeatureExtractor.Dense121
        elif name == "Dense169":
            return FeatureExtractor.Dense169
        elif name == "Dense201":
            return FeatureExtractor.Dense201
        elif name == "NASNetMobile":
            return FeatureExtractor.NASNetMobile
        elif name == "NASNetLarge":
            return FeatureExtractor.NASNetLarge


def EmaNet(input_shape: Tuple[int, int], num_classes: int,
           convolutions: int = 1, dense_layers: int = 3,
           filters: int = 32, dropout_rate: float = 0.3) -> Model:
    """
    Custom CNN model that makes use of two separate stacks of convolution and deconvolution operations.
    The resulting tensors are then concatenated and fed into a new stack of convolution operations, and, finally,
    through a fully-connected NN with multiple layers.
    Dropout and batch normalization are used throughout the network.

    :param input_shape: Shape of the input tensor as a 2-dimensional int tuple
    :param num_classes: Number of classes for the final FC layer
    :param convolutions: Size of the convolution/deconvolution stacks
    :param dense_layers: Number of layers for the FC NN
    :param filters: Number of filters to be used at the first operations in the stacks; subsequent operations
                    will use a number of filters in a proportional way to their 'position' in the stack
    :param dropout_rate: Dropout rate

    :return: a Keras Model
    """

    input_image = Input(shape=input_shape)

    # convolutional stack
    conv = Conv2D(filters=filters,
                  kernel_size=(2, 2),
                  padding="same",
                  activation="relu")(input_image)
    conv = BatchNormalization()(conv)

    for i in range(1, convolutions):
        conv = Conv2D(filters=int(filters * sqrt(i)),
                      kernel_size=(1+i, 1+i),
                      padding="same",
                      activation="relu")(conv)
        conv = BatchNormalization()(conv)

    conv = MaxPooling2D(pool_size=(2, 2),
                        padding="valid")(conv)
    conv = BatchNormalization()(conv)

    # deconvolutional stack
    deconv = Conv2DTranspose(filters=filters,
                             kernel_size=(2, 2),
                             padding="same",
                             activation="relu")(input_image)
    deconv = BatchNormalization()(deconv)

    for i in range(1, convolutions):
        deconv = Conv2DTranspose(filters=int(filters * sqrt(i)),
                                 kernel_size=(1+i, 1+i),
                                 padding="same",
                                 activation="relu")(deconv)
        deconv = BatchNormalization()(deconv)

    deconv = MaxPooling2D(pool_size=(2, 2),
                          padding="valid")(deconv)
    deconv = BatchNormalization()(deconv)

    # merge the two stacks
    merged = keras.layers.concatenate([conv, deconv])

    merged = Conv2D(filters=filters,
                    kernel_size=(2, 2),
                    padding="valid",
                    activation="relu")(merged)
    merged = MaxPooling2D(pool_size=(2, 2),
                          padding="valid")(merged)
    merged = BatchNormalization()(merged)

    merged = Conv2D(filters=filters,
                    kernel_size=(3, 3),
                    padding="valid",
                    activation="relu")(merged)
    merged = MaxPooling2D(pool_size=(2, 2),
                          padding="valid")(merged)
    merged = BatchNormalization()(merged)

    # final fully-connected layers
    dense = Flatten()(merged)
    dense = BatchNormalization()(dense)
    dense = Dropout(rate=dropout_rate)(dense)

    num_units = filters * 5
    for i in range(1, dense_layers+1):
        dense = Dense(units=int(num_units/i),
                      activation="relu")(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(rate=dropout_rate)(dense)

    output_dense = Dense(units=num_classes,
                         activation="softmax")(dense)

    model = Model(input=input_image, output=output_dense, name="EmaNet")
    adam_opt = Adam(lr=0.1)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=adam_opt,
                  metrics=["accuracy"])

    return model


def TransferNet(input_shape: Tuple[int, int], num_classes: int,
                feature_extractor: FeatureExtractor = None,
                dense_layers: int = 3, dropout_rate: float = 0.3) -> Model:
    """
    Exploits a pre-trained model as feature extractor, feeding its output into a fully-connected NN.
    The feature extractor model is NOT fine-tuned for the specific task.
    Dropout and batch normalization are used throughout the trainable portion of the network.

    :param input_shape: Shape of the input tensor as a 2-dimensional int tuple
    :param num_classes: Number of classes for the final FC layer
    :param feature_extractor: FeatureExtractor instance representing which pre-trained model to use as feature extractor
    :param dense_layers: Number of layers for the FC NN
    :param dropout_rate: Dropout rate

    :return: a Keras model
    """

    adam_opt = Adam(lr=0.1)
    model_input = Input(shape=input_shape)

    # load pre-trained model on ImageNet
    if feature_extractor == FeatureExtractor.Dense121:
        fe_model = DenseNet121(weights="imagenet", include_top=False, input_tensor=model_input)
        out_layer = "relu"
    elif feature_extractor == FeatureExtractor.Dense169:
        fe_model = DenseNet169(weights="imagenet", include_top=False, input_tensor=model_input)
        out_layer = "relu"
    elif feature_extractor == FeatureExtractor.Dense201:
        fe_model = DenseNet201(weights="imagenet", include_top=False, input_tensor=model_input)
        out_layer = "relu"
    elif feature_extractor == FeatureExtractor.NASNetLarge:
        fe_model = NASNetLarge(weights="imagenet", include_top=False, input_tensor=model_input)
        out_layer = "activation_260"
    else:
        # default: NASNetMobile
        fe_model = NASNetMobile(weights="imagenet", include_top=False, input_tensor=model_input)
        out_layer = "activation_188"

    fe_model = Model(input=model_input, output=fe_model.output, name="FeatureExtractor")
    fe_model.compile(loss=keras.losses.categorical_crossentropy,
                     optimizer=adam_opt,
                     metrics=["accuracy"])

    # get handles to the model (input, output tensors)
    fe_input = fe_model.get_layer(index=0).input
    fe_output = fe_model.get_layer(index=-1).output

    # freeze layers
    for _, layer in enumerate(fe_model.layers):
        layer.trainable = False

    # final fully-connected layers
    dense = Flatten()(fe_output)
    dense = BatchNormalization()(dense)
    dense = Dropout(rate=dropout_rate)(dense)

    num_units = 128
    for i in range(1, dense_layers + 1):
        dense = Dense(units=int(num_units / i),
                      activation="relu")(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(rate=dropout_rate)(dense)

    output_dense = Dense(units=num_classes,
                         activation="softmax")(dense)

    model = Model(input=fe_input, output=output_dense, name="TransferNet")
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=adam_opt,
                  metrics=["accuracy"])

    return model


if __name__ == "__main__":
    train = "../data/MWI-Dataset-1.1.1_400/"
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_gen = train_datagen.flow_from_directory(directory=train,
                                                  target_size=(48, 64),
                                                  color_mode="rgb",
                                                  batch_size=64,
                                                  class_mode="categorical",
                                                  shuffle=False)

    samples = train_gen.n
    shape = train_gen.image_shape
    num_classes = train_gen.num_classes
    print("There are {n} samples of shape {s} over {c} classes.".format(n=samples,
                                                                        s=shape,
                                                                        c=num_classes))

    #m = EmaNet(input_shape=shape,
    #           num_classes=num_classes,
    #           convolutions=3,
    #           dense_layers=5)

    #m = TransferNet(input_shape=shape,
    #                num_classes=num_classes)

    #m.summary()
