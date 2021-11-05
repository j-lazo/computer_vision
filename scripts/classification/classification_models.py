import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, \
    GlobalAveragePooling2D, Concatenate, AveragePooling2D


def simple_sequential_3layers(num_classes):

    #nclass = len(train_gen.class_indices)
    model = Sequential()
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    return model


def simple_sequential_3Conv(num_classes):

    #nclass = len(train_gen.class_indices)
    model = Sequential()
    model.add(AveragePooling2D())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))


def residual_conv_3_layer(num_classes):

    def conv_block():
        return 0


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
