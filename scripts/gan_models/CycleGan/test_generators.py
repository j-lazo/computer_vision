import os

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow_addons as tfa
from absl import app, flags
from absl.flags import FLAGS
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Nadam
from tensorflow.keras.models import Model
import skimage.io as iio
import cv2

import skimage.io as iio


def _check(images, dtypes, min_value=-np.inf, max_value=np.inf):
    # check type
    assert isinstance(images, np.ndarray), '`images` should be np.ndarray!'

    # check dtype
    dtypes = dtypes if isinstance(dtypes, (list, tuple)) else [dtypes]
    assert images.dtype in dtypes, 'dtype of `images` shoud be one of %s!' % dtypes

    # check nan and inf
    assert np.all(np.isfinite(images)), '`images` contains NaN or Inf!'

    # check value
    if min_value not in [None, -np.inf]:
        l = '[' + str(min_value)
    else:
        l = '(-inf'
        min_value = -np.inf
    if max_value not in [None, np.inf]:
        r = str(max_value) + ']'
    else:
        r = 'inf)'
        max_value = np.inf
    assert np.min(images) >= min_value and np.max(images) <= max_value, \
        '`images` should be in the range of %s!' % (l + ',' + r)


def to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    """Transform images from [-1.0, 1.0] to [min_value, max_value] of dtype."""
    _check(images, [np.float32, np.float64], -1.0, 1.0)
    dtype = dtype if dtype else images.dtype
    return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)


def im2uint(images):
    """Transform images from [-1.0, 1.0] to uint8."""
    return to_range(images, 0, 255, np.uint8)


def imwrite(image, path, quality=100, **plugin_args):
    """Save a [-1.0, 1.0] image."""
    if path.endswith('.jpg'):
        iio.imsave(path, im2uint(image), quality=quality, **plugin_args)

    else:
        iio.imsave(path, im2uint(image), **plugin_args)

def load_model(directory_model):
    if directory_model.endswith('.h5'):
        model_path = directory_model
    else:
        files_dir = [f for f in os.listdir(directory_model) if f.endswith('.h5')]
        if files_dir:
            model_path = files_dir.pop()
        else:
            files_dir = [f for f in os.listdir(directory_model) if f.endswith('.pb')]
            if files_dir:
                model_path = ''
                print(f'Tensorflow model found at {directory_model}')
            else:
                print(f'No model found in {directory_model}')

    print('MODEL USED:')
    print(model_path)
    #model = tf.keras.models.load_model(model_path, compile=False)
    print(f'Model path: {directory_model + model_path}')
    model = tf.keras.models.load_model(directory_model + model_path)
    model.summary()
    input_size = (model.layers[0].input_shape[:])

    return model, input_size


def imread(path):

    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, 3)

    return img


def _map_fn(img, crop_size=32):  # preprocessing
    img = tf.image.resize(img, [crop_size, crop_size])
    # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(img, crop_size)
    img = tf.clip_by_value(img, 0, 255) / 255.0
    # or img = tl.minmax_norm(img)
    img = img * 2 - 1
    return img


def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return keras.layers.LayerNormalization


def ResnetGenerator(input_shape=(256, 256, 3),
                    output_channels=3,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=9,
                    norm='instance_norm'):
    Norm = _get_norm_layer(norm)

    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)

        return keras.layers.add([x, h])

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(dim, 7, padding='valid', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)

    # 2
    for _ in range(n_downsamplings):
        dim *= 2
        h = keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 3
    for _ in range(n_blocks):
        h = _residual_block(h)

    # 4
    for _ in range(n_downsamplings):
        dim //= 2
        h = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 5
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(output_channels, 7, padding='valid')(h)
    h = tf.tanh(h)

    return keras.Model(inputs=inputs, outputs=h)

class Checkpoint:
    """Enhanced "tf.train.Checkpoint"."""

    def __init__(self,
                 checkpoint_kwargs,  # for "tf.train.Checkpoint"
                 directory,  # for "tf.train.CheckpointManager"
                 max_to_keep=5,
                 keep_checkpoint_every_n_hours=None):
        self.checkpoint = tf.train.Checkpoint(**checkpoint_kwargs)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory, max_to_keep, keep_checkpoint_every_n_hours)

    def restore(self, save_path=None):
        save_path = self.manager.latest_checkpoint if save_path is None else save_path
        return self.checkpoint.restore(save_path)

    def save(self, file_prefix_or_checkpoint_number=None, session=None):
        if isinstance(file_prefix_or_checkpoint_number, str):
            return self.checkpoint.save(file_prefix_or_checkpoint_number, session=session)
        else:
            return self.manager.save(checkpoint_number=file_prefix_or_checkpoint_number)

    def __getattr__(self, attr):
        if hasattr(self.checkpoint, attr):
            return getattr(self.checkpoint, attr)
        elif hasattr(self.manager, attr):
            return getattr(self.manager, attr)
        else:
            self.__getattribute__(attr)  # this will raise an exception


def build_complete_model(G_A2B, G_B2A, target_domain='nbi'):
  input_sizes_models = {'vgg16': (224, 224), 'vgg19': (224, 224), 'inception_v3': (299, 299),
                          'resnet50': (224, 224), 'resnet101': (224, 244), 'mobilenet': (224, 224),
                          'densenet121': (224, 224), 'xception': (299, 299)}
  input_model = Input((256, 256, 3))

  if target_domain == 'nbi':
    c = G_A2B(input_model)
    r = G_B2A(c)
  else:
    c = G_B2A(input_model)
    r = G_A2B(c)

  output_layer = [c, r]
  return Model(inputs=input_model, outputs=output_layer, name='multi_input_output_classification')


def main(_argv):
    base_dir = os.getcwd()

    unique_classes = ['CIS', 'HGC', 'HLT', 'LGC', 'NTL']
    checkpoint_dir = FLAGS.model_dir
    target_test = FLAGS.test_target
    target_domain = FLAGS.target_domain
    directory_model = FLAGS.model_classification
    # models
    G_A2B = ResnetGenerator(input_shape=(256, 256, 3))
    G_B2A = ResnetGenerator(input_shape=(256, 256, 3))

    Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), checkpoint_dir).restore()

    model = build_complete_model(G_A2B, G_B2A, target_domain)
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics='accuracy')

    model_classification, input_size_classification = load_model(directory_model)
    print(input_size_classification)
    if os.path.isdir(target_test):
        list_imgs = os.listdir(target_test)
        for img in list_imgs:
            print(img)
            image_dir = target_test + img
            input_image = imread(image_dir)
            proc_imgs = _map_fn(input_image, crop_size=256)
            proc_imgs = np.expand_dims(proc_imgs, axis=0)

            predict = model.predict(proc_imgs)

            img_b2a = predict[0][0]
            img_b2a2b = predict[1][0]
            imwrite(img_b2a, os.path.join(base_dir, 'temp_folder', 'converted.jpg'))
            imwrite(img_b2a2b, os.path.join(base_dir, 'temp_folder', 'reconverted.jpg'))

            c = im2uint(predict[0][0])
            r = im2uint(predict[1][0])
            print(np.amin(predict[0][0]), np.amax(predict[0][0]))
            print(np.amin(predict[1][0]), np.amax(predict[1][0]))

            input_c = cv2.resize(c, (224, 224))
            input_r = cv2.resize(r, (224, 224))

            input_c = cv2.cvtColor(input_c, cv2.COLOR_RGB2BGR)
            input_r = cv2.cvtColor(input_r, cv2.COLOR_RGB2BGR)

            class_c = model_classification.predict(np.expand_dims(input_c, axis=0))
            class_r = model_classification.predict(np.expand_dims(input_r, axis=0))

            print(np.shape(c), class_c, unique_classes[np.argmax(class_c)])
            print(np.shape(r), class_r, unique_classes[np.argmax(class_r)])

            input_cv = cv2.imread(image_dir)
            plot_original = cv2.resize(input_cv, (256, 256))
            #input_cv = cv2.cvtColor(input_cv, cv2.COLOR_BGR2RGB)
            input_classification = cv2.resize(input_cv, (224, 224))
            input_classification = np.expand_dims(input_classification, axis=0)
            class_O = model_classification.predict(input_classification)
            print('Prediction Original', class_O, unique_classes[np.argmax(class_O)])

            read_c = cv2.imread(os.path.join(base_dir, 'temp_folder', 'converted.jpg'))
            read_r = cv2.imread(os.path.join(base_dir, 'temp_folder', 'reconverted.jpg'))

            input2_c = cv2.resize(read_c, (224, 224))
            input2_r = cv2.resize(read_r, (224, 224))

            #print(np.amin(input2_c), np.amax(input2_c))
            #print(np.amin(input2_r), np.amax(input2_r))

            class_c2 = model_classification.predict(np.expand_dims(input2_c, axis=0))
            class_r2 = model_classification.predict(np.expand_dims(input2_r, axis=0))

            print(np.shape(input2_c), class_c2, unique_classes[np.argmax(class_c2)])
            print(np.shape(input2_r), class_r2, unique_classes[np.argmax(class_r2)])

            img_mix = np.concatenate([plot_original, c, r], axis=1)
            plt.figure()
            plt.imshow(img_mix)
            plt.show()

    elif target_test.endswith('.avi') or target_test.endswith('.mp4') or target_test.endswith('.mpg'):
        # If the input is the camera, pass 0 instead of the video file name
        cap = cv2.VideoCapture(target_test)
        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        # Read until video is completed
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                #image = frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = np.expand_dims(image, axis=0)
                proc_imgs = _map_fn(image, crop_size=256)
                predict = model.predict(proc_imgs)
                original_img = (proc_imgs[0] + 1) * (127)
                c = im2uint(predict[0][0])
                r = im2uint(predict[1][0])
                img_mix = np.concatenate([c, r], axis=1)
                # Display the resulting frame

                #x = image / 255
                #x = np.expand_dims(x, axis=0)
                #y_pred = model_classification.predict(x)


                original_img = (proc_imgs[0] + 1)
                #norm_image = cv2.normalize(img_mix, None, alpha=np.amin(img_mix), beta=np.amax(img_mix),
                #                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                norm_image = cv2.cvtColor(img_mix, cv2.COLOR_RGB2BGR)
                reshaped_frame = cv2.resize(frame, (256, 256))
                cv2.imshow('Original Frames', reshaped_frame)
                cv2.imshow('Converted ', norm_image)

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Break the loop
            else:
                break

        # When everything done, release the video capture object
        cap.release()
        # Closes all the frames
        cv2.destroyAllWindows()


if __name__ == '__main__':
    flags.DEFINE_string('model_dir', 'gan_merge_features', 'name of the model')
    flags.DEFINE_string('test_target', 'gan_merge_features', 'name of the model')
    flags.DEFINE_string('model_classification', '', 'model to perform classification')
    flags.DEFINE_string('target_domain', 'nbi', 'nbi ir wli')
    try:
        app.run(main)
    except SystemExit:
        pass