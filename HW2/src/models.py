"""
Model definitions.
"""

import tensorflow as tf
from keras.losses import categorical_crossentropy
from keras import Model, Input
from keras.layers import Flatten, Dense
from keras.applications.vgg16 import VGG16

if __name__ == "__main__":
    train = "../data/MWI-Dataset-1.1.1_400/"
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_gen = train_datagen.flow_from_directory(directory=train,
                                                  target_size=(48, 48),
                                                  color_mode="rgb",
                                                  batch_size=64,
                                                  class_mode="categorical",
                                                  shuffle=True)

    shape = train_gen.image_shape
    print("There are {n} samples of shape {s} over {c} classes.".format(n=train_gen.n,
                                                                        s=shape,
                                                                        c=train_gen.num_classes))

    model_input = Input(shape=shape)
    vgg16 = VGG16(weights="imagenet", include_top=False, input_tensor=model_input)
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
