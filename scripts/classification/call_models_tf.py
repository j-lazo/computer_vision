import copy
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import tensorflow as tf
from absl import app, flags
from absl.flags import FLAGS
from tensorflow.keras import applications
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Nadam
import pandas as pd
import tqdm
import time
from tensorflow import keras
import tensorflow_addons as tfa
import cv2


def generate_experiment_ID(name_model='', learning_rate='na', batch_size='na', backbone_model='',
                           prediction_model='', mode=''):
    """
    Generate a ID name for the experiment considering the name of the model, the learning rate,
    the batch size, and the date of the experiment

    :param name_model: (str)
    :param learning_rate: (float)
    :param batch_size: (int)
    :param backbone_model: (str)
    :return: (str) id name
    """
    if type(learning_rate) == list:
        lr = learning_rate[0]
    else:
        lr = learning_rate

    if prediction_model == '':
        training_date_time = datetime.datetime.now()
        if backbone_model != '':
            name_mod = ''.join([name_model, '+', backbone_model])
        else:
            name_mod = name_model
        id_name = ''.join([name_mod, '_', mode, '_lr_', str(lr),
                                  '_bs_', str(batch_size), '_',
                                  training_date_time.strftime("%d_%m_%Y_%H_%M")
                                  ])
    else:
        predictions_date_time = datetime.datetime.now()
        id_name = ''.join([prediction_model, '_predictions_', predictions_date_time.strftime("%d_%m_%Y_%H_%M")])

    return id_name


def load_data_from_directory_v1(path_data, csv_annotations=None):
    """
        Give a path, creates two lists with the
        Parameters
        ----------
        path_data :

        Returns
        -------

        """

    images_path = list()
    labels_class = list()
    labels_domain = list()
    dictionary_labels = {}

    list_files = os.listdir(path_data)
    csv_list = [f for f in list_files if f.endswith('.csv')]
    if csv_list:
        csv_indir = csv_list.pop()
    if csv_annotations:
        data_frame = pd.read_csv(csv_annotations)
        list_imgs = data_frame['image_name'].tolist()
        list_classes = data_frame['tissue type'].tolist()
        list_domain = data_frame['imaging type'].tolist()

    list_unique_classes = np.unique(list_classes)
    list_unique_domains = np.unique(list_domain)
    list_files = list()
    list_path_files = list()

    for (dirpath, dirnames, filenames) in os.walk(path_data):
        list_files += [file for file in filenames]
        list_path_files += [os.path.join(dirpath, file) for file in filenames]

    for j, file in enumerate(list_imgs):
        if file in list_files:
            new_dictionary_labels = {file: {'image_name': file, 'path_file': list_path_files[j],
                                            'img_class': list_classes[j], 'img_domain': list_domain[j]}}
            dictionary_labels = {**dictionary_labels, **new_dictionary_labels}
        else:
            list_files.remove(file)

    print(f'Found {len(list_path_files)} images corresponding to {len(list_unique_classes)} classes and '
          f'{len(list_unique_domains)} domains at: {path_data}')

    return list_files, dictionary_labels


def load_data_from_directory(path_data, csv_annotations=None):
    """
    Give a path, creates two lists with the
    Parameters
    ----------
    path_data :

    Returns
    -------

    """

    images_path = list()
    labels = list()
    dictionary_labels = {}

    list_files = os.listdir(path_data)
    csv_list = [f for f in list_files if f.endswith('.csv')]
    if csv_list:
        csv_indir = csv_list.pop()
    if csv_annotations:
        data_frame = pd.read_csv(csv_annotations)
    #elif csv_indir:
    #    data_frame = pd.read_csv(os.path.join(path_data, csv_indir))

    list_unique_classes = np.unique([f for f in list_files if os.path.isdir(os.path.join(path_data, f))])
    """if data_frame:
        for j, unique_class in enumerate(list_unique_classes):
            path_images = ''.join([path_data, '/', unique_class, '/*'])
            added_images = sorted(glob(path_images))
            new_dictionary_labels = {image_name: unique_class for image_name in added_images}

            images_path = images_path + added_images
            added_labels = [j] * len(added_images)
            labels = labels + added_labels
            dictionary_labels = {**dictionary_labels, **new_dictionary_labels}

    elif list_unique_classes:"""
    for j, unique_class in enumerate(list_unique_classes):
        path_images = ''.join([path_data, '/', unique_class, '/*'])
        added_images = sorted(glob(path_images))
        new_dictionary_labels = {image_name: unique_class for image_name in added_images}

        images_path = images_path + added_images
        added_labels = [j] * len(added_images)
        labels = labels + added_labels
        dictionary_labels = {**dictionary_labels, **new_dictionary_labels}

    print(f'Found {len(images_path)} images corresponding to {len(list_unique_classes)} classes at: {path_data}')

    return images_path, labels, dictionary_labels


def read_stacked_images_npy(path_data):
    """

    Parameters
    ----------
    path_data : (bytes) path to the data
    preprocessing_input : Pr-processing input unit to be used in case some backbone is used in the classifier

    Returns
    -------

    """
    path_data = path_data.decode()
    if path_data.endswith('.npz'):
        img_array = np.load(path_data)
        img = img_array['arr_0']
    else:
        img = np.load(path_data)

    img = img.astype(np.float64)
    return img


def read_stacked_images_npy_predict(path_data):
    """

    Parameters
    ----------
    path_data : (bytes) path to the data
    preprocessing_input : Pr-processing input unit to be used in case some backbone is used in the classifier

    Returns
    -------

    """
    if path_data.endswith('.npz'):
        img_array = np.load(path_data)
        img = img_array['arr_0']
    else:
        img = np.load(path_data)

    img = img.astype(np.float64)
    return img


def imread_tf(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, 3)

    return _map_fn(img, crop_size=256)


def tf_parser_npy(x, y):

    def _parse(x, y):
        x = read_stacked_images_npy(x)
        out_y = np.zeros(NUM_CLASSES)
        out_y[y] = 1.
        out_y[y] = out_y[y].astype(np.float64)
        return x, out_y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([3, 256, 256, 3])
    y.set_shape([NUM_CLASSES])
    return x, y


def tf_parser_v2(x, y):
    # x is the key name in the dictionary y
    # y contains the name of the image, the path, the class and the domain of the image
    def _parse(x1, x2, y):
        img = imread_tf(x1.decode())
        out_y = np.zeros(NUM_CLASSES)
        x2_out = [float(x2)]
        out_y[y] = 1.
        out_y = out_y.astype(np.float64)
        return img, x2_out, out_y

    x1 = x['input_1']
    x2 = x['input_2']
    x1, x2, y = tf.numpy_function(_parse, [x1, x2, y], [tf.float64, tf.float64, tf.float64])
    x1.set_shape([256, 256, 3])
    x2.set_shape([1])
    y.set_shape([NUM_CLASSES])
    x_out = {"input_1": x1, "input_2": x2}
    return x_out, y


def tf_parser_v1(x, y):
    # x is the key name in the dictionary y
    # y contains the name of the image, the path, the class and the domain of the image
    def _parse(x, y):
        img = imread_tf(x.decode())
        out_y = np.zeros(NUM_CLASSES)
        out_y[y] = 1.
        out_y = out_y.astype(np.float64)
        print(type(y))
        print(type(x))
        return img, out_y

    x = imread_tf(x)
    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    print(type(x))
    print(type(y))

    x.set_shape([256, 256, 3])
    y.set_shape([NUM_CLASSES])
    return x, y


def generate_tf_dataset_v1(list_x, dictionary_info, batch_size=1, shuffle=False, buffer_size=10, preprocess_function=None,
                        input_size=(256, 256)):

    """
    Generates a tf dataset asd described in https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    Parameters
    ----------
    x : (list of strings) input data
    y : (list of int) target labels
    batch_size : int
    shuffle : (bool)

    Returns
    -------
    tensorflow Dataset
    """
    global PREPROCESS_FUNCTION
    global INPUT_SIZE
    global NUM_CLASSES
    global NUM_DOMAIS
    global UNIQUE_CLASSES
    global UNIQUE_DOMAINS

    path_imgs = list()
    images_class = list()
    images_domains = list()
    for img_name in list_x:
        path_imgs.append(dictionary_info[img_name]['path_file'])
        images_class.append(dictionary_info[img_name]['img_class'])
        images_domains.append(dictionary_info[img_name]['img_domain'])

    UNIQUE_CLASSES = np.unique(images_class)
    UNIQUE_DOMAINS = np.unique(images_domains)

    NUM_CLASSES = len(UNIQUE_CLASSES)
    NUM_DOMAINS = len(UNIQUE_DOMAINS)
    PREPROCESS_FUNCTION = preprocess_function
    INPUT_SIZE = input_size

    images_domains = [list(UNIQUE_DOMAINS).index(val) for val in images_domains]
    images_class = [list(UNIQUE_CLASSES).index(val) for val in images_class]


    #dataset = tf.data.Dataset.from_tensor_slices({"input_1": path_imgs, "input_2": images_domains}, images_class)
    dataset = tf.data.Dataset.from_tensor_slices((path_imgs, images_class))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size * batch_size)

    dataset = dataset.map(tf_parser_v1)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()

    return dataset


def generate_tf_dataset(x, y, batch_size=1, shuffle=False, buffer_size=10, preprocess_function=None,
                        input_size=(256, 256)):

    """
    Generates a tf dataset asd described in https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    Parameters
    ----------
    x : (list of strings) input data
    y : (list of int) target labels
    batch_size : int
    shuffle : (bool)

    Returns
    -------
    tensorflow Dataset
    """
    global PREPROCESS_FUNCTION
    global INPUT_SIZE
    global NUM_CLASSES

    NUM_CLASSES = len(np.unique(y))
    PREPROCESS_FUNCTION = preprocess_function
    INPUT_SIZE = input_size

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size * batch_size)

    dataset = dataset.map(tf_parser_npy)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()

    return dataset


def analyze_tf_dataset(dataset_dir, plot=True):

    """
    Given the directory to a classification directory, it generates a tf dataset and iterates over it to
    visualize it
    Parameters
    ----------
    dataset_dir : (str) Path to the dataset

    Returns
    -------

    """

    test_x, test_y, dataset_dictionary = load_data_from_directory(dataset_dir)
    test_dataset = generate_tf_dataset(test_x, test_y, batch_size=8, shuffle=False,
                                       buffer_size=500)

    unique_labels = np.unique([f for f in os.listdir(dataset_dir) if os.path.isdir(dataset_dir + f)])
    for j, element in enumerate(test_dataset):
        x, y = element
        x_array = x.numpy()
        y_array = y.numpy()
        print('x:', np.shape(x_array), np.amin(x_array), np.amax(x_array))
        print('y:', np.shape(y_array), np.unique(y_array))
        if plot is True:

            plt.figure()
            plt.subplot(3, 4, 1)
            plt.imshow(x_array[0][0])
            plt.axis('off')
            plt.gca().set_title(str(unique_labels[list(y_array[0]).index(1)]))
            plt.subplot(3, 4, 2)
            plt.imshow(x_array[1][0])
            plt.axis('off')
            plt.gca().set_title(str(unique_labels[list(y_array[1]).index(1)]))
            plt.subplot(3, 4, 3)
            plt.imshow(x_array[2][0])
            plt.axis('off')
            plt.gca().set_title(str(unique_labels[list(y_array[2]).index(1)]))
            plt.subplot(3, 4, 4)
            plt.imshow(x_array[3][0])
            plt.axis('off')
            plt.gca().set_title(str(unique_labels[list(y_array[3]).index(1)]))

            plt.subplot(3, 4, 5)
            plt.imshow(x_array[0][1])
            plt.axis('off')
            plt.subplot(3, 4, 6)
            plt.imshow(x_array[1][1])
            plt.axis('off')
            plt.subplot(3, 4, 7)
            plt.imshow(x_array[2][1])
            plt.axis('off')
            plt.subplot(3, 4, 8)
            plt.imshow(x_array[3][1])
            plt.axis('off')

            plt.subplot(3, 4, 9)
            plt.imshow(x_array[0][2])
            plt.axis('off')
            plt.subplot(3, 4, 10)
            plt.imshow(x_array[1][2])
            plt.axis('off')
            plt.subplot(3, 4, 11)
            plt.imshow(x_array[2][2])
            plt.axis('off')
            plt.subplot(3, 4, 12)
            plt.imshow(x_array[3][2])
            plt.axis('off')
            plt.show()


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
    input_size = (len(model.layers[0].output_shape[:]))

    return model, input_size


def load_pretrained_backbones(name_model, weights='imagenet', include_top=False, trainable=False, new_name=None):
    base_dir_weights = ''.join([os.getcwd(), '/scripts/classification/weights_pretrained_models/'])
    """
    Loads a pretrained model given a name
    :param name_model: (str) name of the model
    :param weights: (str) weights names (default imagenet)
    :return: sequential model with the selected weights
    """
    if name_model == 'vgg16':
        weights_dir = base_dir_weights + 'vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.vgg16.VGG16(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable
        input_size = (224, 224, 3)

    elif name_model == 'vgg19':
        weights_dir = base_dir_weights + 'vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.vgg19.VGG19(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable
        input_size = (224, 224, 3)

    elif name_model == 'inception_v3':
        weights_dir = base_dir_weights + 'inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.inception_v3.InceptionV3(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable
        input_size = (299, 299, 3)

    elif name_model == 'resnet50':
        weights_dir = base_dir_weights + 'resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.resnet50.ResNet50(include_top=include_top, weights=weights_dir)
        base_model.trainable = True
        input_size = (224, 224, 3)

    elif name_model == 'resnet101':
        weights_dir = base_dir_weights + 'resnet101/resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.resnet.ResNet101(include_top=include_top, weights=weights_dir)
        base_model.trainable = True
        #layer1 = base_model.layers[2]
        #global weights_1
        #weights_1 = layer1.weights

        #new_base_model = tf.keras.models.clone_model(base_model)
        #new_base_model.set_weights(base_model.get_weights())
        #layer2 = new_base_model.layers[2]
        #global weights_2
        #weights_2 = layer2.weights
        #print(np.array_equal(weights_1[0], weights_2[0]))

        #base_model.name = new_name

        input_size = (224, 224, 3)

    elif name_model == 'mobilenet':
        weights_dir = base_dir_weights + 'mobilenet/mobilenet_1_0_224_tf_no_top.h5'
        base_model = applications.mobilenet.MobileNet(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable
        input_size = (224, 224, 3)

    elif name_model == 'densenet121':
        weights_dir = base_dir_weights + 'densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.densenet.DenseNet121(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable
        input_size = (224, 224, 3)

    elif name_model == 'xception':
        weights_dir = base_dir_weights + 'xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.xception.Xception(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable
        input_size = (299, 299, 3)

    else:
        raise ValueError(f' MODEL: {name_model} not found')

    new_base_model = tf.keras.models.clone_model(base_model)
    new_base_model.set_weights(base_model.get_weights())
    return new_base_model

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


def _map_fn(img, crop_size=32):  # preprocessing
    img = tf.image.resize(img, [crop_size, crop_size])
    # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(img, crop_size)
    img = tf.clip_by_value(img, 0, 255) / 255.
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


def build_model_v1(backbones=['resnet101', 'resnet101', 'resnet101'], after_concat='globalpooling',
                dropout=False):

    input_sizes_models = {'vgg16': (224, 224), 'vgg19': (224, 224), 'inception_v3': (299, 299),
                          'resnet50': (224, 224), 'resnet101': (224, 244), 'mobilenet': (224, 224),
                          'densenet121': (224, 224), 'xception': (299, 299)}

    num_backbones = len(backbones)
    input_model = Input((3, 256, 256, 3))
    x1, x2, x3 = tf.split(input_model, 3, axis=1)
    input_backbone_1 = tf.squeeze(x1, axis=1)
    input_backbone_2 = tf.squeeze(x2, axis=1)
    input_backbone_3 = tf.squeeze(x3, axis=1)

    b1 = tf.image.resize(input_backbone_1, input_sizes_models[backbones[0]], method='bilinear')
    if backbones[0] == 'resnet101':
        b1 = tf.keras.applications.resnet.preprocess_input(b1)
    elif backbones[0] == 'resnet50':
        b1 = tf.keras.applications.resnet50.preprocess_input(b1)
    elif backbones[0] == 'densenet121':
        b1 = tf.keras.applications.densenet.preprocess_input(b1)
    elif backbones[0] == 'vgg19':
        b1 = tf.keras.applications.vgg19.preprocess_input(b1)
    elif backbones[0] == 'inception_v3':
        b1 = tf.keras.applications.inception_v3.preprocess_input(b1)

    backbone_model_1 = load_pretrained_backbones(backbones[0])
    backbone_model_1._name = 'backbone_1'
    for layer in backbone_model_1.layers:
        layer.trainable = False
    b1 = backbone_model_1(b1)
    l1 = GlobalAveragePooling2D()(b1)
    l1 = Dense(1024, activation='relu')(l1)
    l1 = Dropout(0.5)(l1)
    l1 = Dense(2048, activation='relu')(l1)
    l1 = Dropout(0.5)(l1)
    l1 = Dense(2048, activation='relu')(l1)
    l1 = Flatten()(l1)

    b2 = tf.image.resize(input_backbone_2, input_sizes_models[backbones[1]], method='bilinear')
    if backbones[1] == 'resnet101':
        b2 = tf.keras.applications.resnet.preprocess_input(b2)
    elif backbones[1] == 'resnet50':
        b2 = tf.keras.applications.resnet50.preprocess_input(b2)
    elif backbones[1] == 'densenet121':
        b2 = tf.keras.applications.densenet.preprocess_input(b2)
    elif backbones[1] == 'vgg19':
        b2 = tf.keras.applications.vgg19.preprocess_input(b2)
    elif backbones[1] == 'inception_v3':
        b2 = tf.keras.applications.inception_v3.preprocess_input(b2)
    backbone_model_2 = load_pretrained_backbones(backbones[1])
    backbone_model_2._name = 'backbone_2'
    for layer in backbone_model_2.layers:
        layer.trainable = False
    b2 = backbone_model_2(b2)
    l2 = GlobalAveragePooling2D()(b2)
    l2 = Dense(1024, activation='relu')(l2)
    l2 = Dropout(0.5)(l2)
    l2 = Dense(2048, activation='relu')(l2)
    l2 = Dropout(0.5)(l2)
    l2 = Dense(2048, activation='relu')(l2)
    l2 = Flatten()(l2)

    if num_backbones == 3:
        b3 = tf.image.resize(input_backbone_3, input_sizes_models[backbones[2]], method='bilinear')
        if backbones[2] == 'resnet101':
            b3 = tf.keras.applications.resnet.preprocess_input(b3)
        elif backbones[2] == 'resnet50':
            b3 = tf.keras.applications.resnet50.preprocess_input(b3)
        elif backbones[2] == 'densenet121':
            b3 = tf.keras.applications.densenet.preprocess_input(b3)
        elif backbones[2] == 'vgg19':
            b3 = tf.keras.applications.vgg19.preprocess_input(b3)
        elif backbones[2] == 'inception_v3':
            b3 = tf.keras.applications.inception_v3.preprocess_input(b3)
        backbone_model_3 = load_pretrained_backbones(backbones[2])
        backbone_model_3._name = 'backbone_3'
        for layer in backbone_model_3.layers:
            layer.trainable = False
        b3 = backbone_model_3(b3)
        l3 = GlobalAveragePooling2D()(b3)
        l3 = Dense(1024, activation='relu')(l3)
        l3 = Dropout(0.5)(l3)
        l3 = Dense(2048, activation='relu')(l3)
        l3 = Dropout(0.5)(l3)
        l3 = Dense(2048, activation='relu')(l3)
        l3 = Flatten()(l3)
        x = Concatenate()([l1, l2, l3])

    else:
        x = Concatenate()([l1, l2])

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    output_layer = Dense(5, activation='softmax')(x)

    return Model(inputs=input_model, outputs=output_layer, name='multi_input_output_classification')


def multi_output_model(backbones=['resnet101', 'resnet101', 'resnet101'], after_concat='globalpooling',
                dropout=False):

    input_sizes_models = {'vgg16': (224, 224), 'vgg19': (224, 224), 'inception_v3': (299, 299),
                          'resnet50': (224, 224), 'resnet101': (224, 244), 'mobilenet': (224, 224),
                          'densenet121': (224, 224), 'xception': (299, 299)}

    num_backbones = len(backbones)
    input_model = Input((3, 256, 256, 3))
    x1, x2, x3 = tf.split(input_model, 3, axis=1)
    input_backbone_1 = tf.squeeze(x1, axis=1)
    input_backbone_2 = tf.squeeze(x2, axis=1)
    input_backbone_3 = tf.squeeze(x3, axis=1)

    b1 = tf.image.resize(input_backbone_1, input_sizes_models[backbones[0]], method='bilinear')
    if backbones[0] == 'resnet101':
        b1 = tf.keras.applications.resnet.preprocess_input(b1)
    elif backbones[0] == 'resnet50':
        b1 = tf.keras.applications.resnet50.preprocess_input(b1)
    elif backbones[0] == 'densenet121':
        b1 = tf.keras.applications.densenet.preprocess_input(b1)
    elif backbones[0] == 'vgg19':
        b1 = tf.keras.applications.vgg19.preprocess_input(b1)
    elif backbones[0] == 'inception_v3':
        b1 = tf.keras.applications.inception_v3.preprocess_input(b1)

    backbone_model_1 = load_pretrained_backbones(backbones[0])
    backbone_model_1._name = 'backbone_1'
    for layer in backbone_model_1.layers:
        layer.trainable = False
    b1 = backbone_model_1(b1)

    b2 = tf.image.resize(input_backbone_2, input_sizes_models[backbones[1]], method='bilinear')
    if backbones[1] == 'resnet101':
        b2 = tf.keras.applications.resnet.preprocess_input(b2)
    elif backbones[1] == 'resnet50':
        b2 = tf.keras.applications.resnet50.preprocess_input(b2)
    elif backbones[1] == 'densenet121':
        b2 = tf.keras.applications.densenet.preprocess_input(b2)
    elif backbones[1] == 'vgg19':
        b2 = tf.keras.applications.vgg19.preprocess_input(b2)
    elif backbones[1] == 'inception_v3':
        b2 = tf.keras.applications.inception_v3.preprocess_input(b2)
    backbone_model_2 = load_pretrained_backbones(backbones[1])
    backbone_model_2._name = 'backbone_2'
    for layer in backbone_model_2.layers:
        layer.trainable = False
    b2 = backbone_model_2(b2)
    if num_backbones == 3:
        b3 = tf.image.resize(input_backbone_3, input_sizes_models[backbones[2]], method='bilinear')
        if backbones[2] == 'resnet101':
            b3 = tf.keras.applications.resnet.preprocess_input(b3)
        elif backbones[2] == 'resnet50':
            b3 = tf.keras.applications.resnet50.preprocess_input(b3)
        elif backbones[2] == 'densenet121':
            b3 = tf.keras.applications.densenet.preprocess_input(b3)
        elif backbones[2] == 'vgg19':
            b3 = tf.keras.applications.vgg19.preprocess_input(b3)
        elif backbones[2] == 'inception_v3':
            b3 = tf.keras.applications.inception_v3.preprocess_input(b3)
        backbone_model_3 = load_pretrained_backbones(backbones[2])
        backbone_model_3._name = 'backbone_3'
        for layer in backbone_model_3.layers:
            layer.trainable = False
        b3 = backbone_model_3(b3)
        x = Concatenate()([b1, b2, b3])

        l3 = GlobalAveragePooling2D()(b3)
        l3 = Dense(1024, activation='relu')(l3)
        l3 = Dropout(0.5)(l3)
        l3 = Dense(2048, activation='relu')(l3)
        l3 = Dropout(0.5)(l3)
        l3 = Dense(2048, activation='relu')(l3)
        l3 = Flatten()(l3)
        output_l3 = Dense(5, activation='softmax')(l3)


    else:
        x = Concatenate()([b1, b2])
    if after_concat == 'globalpooling':
        x = GlobalAveragePooling2D()(x)
    else:
        x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu')(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu')(x)
    x = Flatten()(x)
    output_layer = Dense(5, activation='softmax')(x)

    l1 = GlobalAveragePooling2D()(b1)
    l1 = Dense(1024, activation='relu')(l1)
    l1 = Dropout(0.5)(l1)
    l1 = Dense(2048, activation='relu')(l1)
    l1 = Dropout(0.5)(l1)
    l1 = Dense(2048, activation='relu')(l1)
    l1 = Flatten()(l1)
    output_l1 = Dense(5, activation='softmax')(l1)

    l2 = GlobalAveragePooling2D()(b2)
    l2 = Dense(1024, activation='relu')(l2)
    l2 = Dropout(0.5)(l2)
    l2 = Dense(2048, activation='relu')(l2)
    l2 = Dropout(0.5)(l2)
    l2 = Dense(2048, activation='relu')(l2)
    l2 = Flatten()(l2)
    output_l2 = Dense(5, activation='softmax')(l2)

    l4 = Concatenate()([l1, l2, l3])
    l4 = Flatten()(l4)
    l4 = Dense(1024, activation='relu')(l4)
    l4 = Dropout(0.5)(l4)
    l4 = Dense(1024, activation='relu')(l4)
    l4 = Flatten()(l4)
    output_l4 = Dense(5, activation='softmax')(l4)

    return Model(inputs=input_model, outputs=[output_layer, output_l1, output_l2, output_l3, output_l4], name='multi_input_output_classification')


def build_generator_base(G_A2B, G_B2A):

    input_model = Input((256, 256, 3))
    t_input = Input(shape=(1,), dtype=tf.int32, name="t_input")

    # branch 1 if target domain is NBI
    c1 = G_A2B(input_model)
    r1 = G_B2A(c1)
    #x1 = [c1, r1]

    # branch 2 if target domain is WLI
    c2 = G_B2A(input_model)
    r2 = G_A2B(c2)
    #x2 = [c2, r2]

    c = tf.keras.backend.switch(t_input, c1, c2)
    r = tf.keras.backend.switch(t_input, r1, r2)
    output_layer = [c, r]
    return Model(inputs=[t_input, input_model], outputs=output_layer, name='multi_input_output_classification')


def build_gan_model_merge_out():
    pass


def build_gan_model_features(backbones=['resnet101', 'resnet101', 'resnet101'], after_concat='globalpooling',
                dropout=False, gan_base='checkpoint_charlie'):

    input_sizes_models = {'vgg16': (224, 224), 'vgg19': (224, 224), 'inception_v3': (299, 299),
                          'resnet50': (224, 224), 'resnet101': (224, 244), 'mobilenet': (224, 224),
                          'densenet121': (224, 224), 'xception': (299, 299)}
    checkpoint_dir = ''.join([os.getcwd(), '/scripts/gan_models/CycleGan/sample_weights/', gan_base])

    # Generator Models
    G_A2B = ResnetGenerator(input_shape=(256, 256, 3))
    G_B2A = ResnetGenerator(input_shape=(256, 256, 3))

    Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), checkpoint_dir).restore()
    num_backbones = len(backbones)

    input_image = Input((256, 256, 3))
    t_input = Input(shape=(1,), dtype=tf.int32, name="t_input")

    # branch 1 if target domain is NBI
    c1 = G_A2B(input_image)
    r1 = G_B2A(c1)
    # x1 = [c1, r1]

    # branch 2 if target domain is WLI
    c2 = G_B2A(input_image)
    r2 = G_A2B(c2)
    # x2 = [c2, r2]

    c = tf.keras.backend.switch(t_input, c1, c2)
    r = tf.keras.backend.switch(t_input, r1, r2)

    input_backbone_1 = input_image
    input_backbone_2 = c
    input_backbone_3 = r
    b1 = tf.image.resize(input_image, input_sizes_models[backbones[0]], method='bilinear')

    if backbones[0] == 'resnet101':
        b1 = tf.keras.applications.resnet.preprocess_input(b1)
    elif backbones[0] == 'resnet50':
        b1 = tf.keras.applications.resnet50.preprocess_input(b1)
    elif backbones[0] == 'densenet121':
        b1 = tf.keras.applications.densenet.preprocess_input(b1)
    elif backbones[0] == 'vgg19':
        b1 = tf.keras.applications.vgg19.preprocess_input(b1)
    elif backbones[0] == 'inception_v3':
        b1 = tf.keras.applications.inception_v3.preprocess_input(b1)

    backbone_model_1 = load_pretrained_backbones(backbones[0])
    backbone_model_1._name = 'backbone_1'
    for layer in backbone_model_1.layers:
        layer.trainable = False
    b1 = backbone_model_1(b1)

    b2 = tf.image.resize(input_backbone_2, input_sizes_models[backbones[1]], method='bilinear')
    if backbones[1] == 'resnet101':
        b2 = tf.keras.applications.resnet.preprocess_input(b2)
    elif backbones[1] == 'resnet50':
        b2 = tf.keras.applications.resnet50.preprocess_input(b2)
    elif backbones[1] == 'densenet121':
        b2 = tf.keras.applications.densenet.preprocess_input(b2)
    elif backbones[1] == 'vgg19':
        b2 = tf.keras.applications.vgg19.preprocess_input(b2)
    elif backbones[1] == 'inception_v3':
        b2 = tf.keras.applications.inception_v3.preprocess_input(b2)
    backbone_model_2 = load_pretrained_backbones(backbones[1])
    backbone_model_2._name = 'backbone_2'
    for layer in backbone_model_2.layers:
        layer.trainable = False
    b2 = backbone_model_2(b2)
    if num_backbones == 3:
        b3 = tf.image.resize(input_backbone_3, input_sizes_models[backbones[2]], method='bilinear')
        if backbones[2] == 'resnet101':
            b3 = tf.keras.applications.resnet.preprocess_input(b3)
        elif backbones[2] == 'resnet50':
            b3 = tf.keras.applications.resnet50.preprocess_input(b3)
        elif backbones[2] == 'densenet121':
            b3 = tf.keras.applications.densenet.preprocess_input(b3)
        elif backbones[2] == 'vgg19':
            b3 = tf.keras.applications.vgg19.preprocess_input(b3)
        elif backbones[2] == 'inception_v3':
            b3 = tf.keras.applications.inception_v3.preprocess_input(b3)
        backbone_model_3 = load_pretrained_backbones(backbones[2])
        backbone_model_3._name = 'backbone_3'
        for layer in backbone_model_3.layers:
            layer.trainable = False
        b3 = backbone_model_3(b3)
        x = Concatenate()([b1, b2, b3])

    else:
        x = Concatenate()([b1, b2])
    if after_concat == 'globalpooling':
        x = GlobalAveragePooling2D()(x)
    else:
        x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu')(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu')(x)
    x = Flatten()(x)
    output_layer = Dense(5, activation='softmax')(x)

    return Model(inputs=[t_input, input_image], outputs=output_layer, name='gan_merge_classification')


def build_model(backbones=['resnet101', 'resnet101', 'resnet101'], after_concat='globalpooling',
                dropout=False):

    input_sizes_models = {'vgg16': (224, 224), 'vgg19': (224, 224), 'inception_v3': (299, 299),
                          'resnet50': (224, 224), 'resnet101': (224, 244), 'mobilenet': (224, 224),
                          'densenet121': (224, 224), 'xception': (299, 299)}

    num_backbones = len(backbones)
    input_model = Input((3, 256, 256, 3))
    x1, x2, x3 = tf.split(input_model, 3, axis=1)
    input_backbone_1 = tf.squeeze(x1, axis=1)
    input_backbone_2 = tf.squeeze(x2, axis=1)
    input_backbone_3 = tf.squeeze(x3, axis=1)

    b1 = tf.image.resize(input_backbone_1, input_sizes_models[backbones[0]], method='bilinear')
    if backbones[0] == 'resnet101':
        b1 = tf.keras.applications.resnet.preprocess_input(b1)
    elif backbones[0] == 'resnet50':
        b1 = tf.keras.applications.resnet50.preprocess_input(b1)
    elif backbones[0] == 'densenet121':
        b1 = tf.keras.applications.densenet.preprocess_input(b1)
    elif backbones[0] == 'vgg19':
        b1 = tf.keras.applications.vgg19.preprocess_input(b1)
    elif backbones[0] == 'inception_v3':
        b1 = tf.keras.applications.inception_v3.preprocess_input(b1)

    backbone_model_1 = load_pretrained_backbones(backbones[0])
    backbone_model_1._name = 'backbone_1'
    for layer in backbone_model_1.layers:
        layer.trainable = False
    b1 = backbone_model_1(b1)

    b2 = tf.image.resize(input_backbone_2, input_sizes_models[backbones[1]], method='bilinear')
    if backbones[1] == 'resnet101':
        b2 = tf.keras.applications.resnet.preprocess_input(b2)
    elif backbones[1] == 'resnet50':
        b2 = tf.keras.applications.resnet50.preprocess_input(b2)
    elif backbones[1] == 'densenet121':
        b2 = tf.keras.applications.densenet.preprocess_input(b2)
    elif backbones[1] == 'vgg19':
        b2 = tf.keras.applications.vgg19.preprocess_input(b2)
    elif backbones[1] == 'inception_v3':
        b2 = tf.keras.applications.inception_v3.preprocess_input(b2)
    backbone_model_2 = load_pretrained_backbones(backbones[1])
    backbone_model_2._name = 'backbone_2'
    for layer in backbone_model_2.layers:
        layer.trainable = False
    b2 = backbone_model_2(b2)
    if num_backbones == 3:
        b3 = tf.image.resize(input_backbone_3, input_sizes_models[backbones[2]], method='bilinear')
        if backbones[2] == 'resnet101':
            b3 = tf.keras.applications.resnet.preprocess_input(b3)
        elif backbones[2] == 'resnet50':
            b3 = tf.keras.applications.resnet50.preprocess_input(b3)
        elif backbones[2] == 'densenet121':
            b3 = tf.keras.applications.densenet.preprocess_input(b3)
        elif backbones[2] == 'vgg19':
            b3 = tf.keras.applications.vgg19.preprocess_input(b3)
        elif backbones[2] == 'inception_v3':
            b3 = tf.keras.applications.inception_v3.preprocess_input(b3)
        backbone_model_3 = load_pretrained_backbones(backbones[2])
        backbone_model_3._name = 'backbone_3'
        for layer in backbone_model_3.layers:
            layer.trainable = False
        b3 = backbone_model_3(b3)
        x = Concatenate()([b1, b2, b3])

    else:
        x = Concatenate()([b1, b2])
    if after_concat == 'globalpooling':
        x = GlobalAveragePooling2D()(x)
    else:
        x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu')(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu')(x)
    x = Flatten()(x)
    output_layer = Dense(5, activation='softmax')(x)

    return Model(inputs=input_model, outputs=output_layer, name='multi_input_classification')


def evaluate_and_predict(model, directory_to_evaluate, results_directory,
                         output_name='', results_id='', batch_size=1,
                         analyze_data=False, output_dir=''):
    print(f'Evaluation of: {directory_to_evaluate}')

    # load the data to evaluate and predict

    batch_size = 8
    test_x, test_y, dataset_dictionary = load_data_from_directory(directory_to_evaluate)
    test_dataset = generate_tf_dataset(test_x, test_y, batch_size=batch_size)
    test_steps = (len(test_x) // batch_size)

    if len(test_x) % batch_size != 0:
        test_steps += 1

    evaluation = model.evaluate(test_dataset, steps=test_steps)
    inference_times = []
    prediction_names = []
    prediction_outputs = []
    real_values = []
    print('Evaluation results:')
    print(evaluation)
    for i, (x, y) in tqdm.tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        real_values.append(dataset_dictionary[x])
        prediction_names.append(os.path.split(x)[-1])
        init_time = time.time()
        x = read_stacked_images_npy_predict(x)
        x = np.expand_dims(x, axis=0)
        y_pred = model.predict(x)
        prediction_outputs.append(y_pred[0])
        inference_times.append(time.time() - init_time)
        # determine the top-1 prediction class
        prediction_id = np.argmax(y_pred[0])

    print('Prediction times analysis')
    print(np.min(prediction_outputs), np.mean(prediction_outputs), np.max(prediction_outputs))

    unique_values = np.unique(real_values)
    label_index = [unique_values[np.argmax(pred)] for pred in prediction_outputs]

    x_pred = [[] for _ in range(len(unique_values))]
    for x in prediction_outputs:
        for i in range(len(x)):
            x_pred[i].append(x[i])

    header_column = ['class_' + str(i+1) for i in range(5)]
    header_column.insert(0, 'fname')
    header_column.append('over all')
    header_column.append('real values')
    df = pd.DataFrame(columns=header_column)
    df['fname'] = [os.path.basename(x_name) for x_name in test_x]
    df['real values'] = real_values
    for i in range(len(unique_values)):
        class_name = 'class_' + str(i+1)
        df[class_name] = x_pred[i]

    df['over all'] = label_index
    # save the predictions  of each case
    results_csv_file = ''.join([results_directory, 'predictions_', output_name, '_', results_id, '_.csv'])
    df.to_csv(results_csv_file, index=False)

    if analyze_data is True:
        pass

    return results_csv_file


def evalute_test_directory(model, test_data, results_directory, new_results_id, analyze_data=True):

    # determine if there are sub_folders or if it's the absolute path of the dataset
    sub_dirs = [f for f in os.listdir(test_data) if os.path.isdir(test_data + f)]
    if sub_dirs:
        print(f'sub-directoires {sub_dirs} found in test foler')
        for sub_dir in sub_dirs:
            test_data_dir = ''.join([test_data, '/', sub_dir])
            name_file = evaluate_and_predict(model, test_data_dir, results_directory,
                                             results_id=new_results_id, output_name='test',
                                             analyze_data=analyze_data)
            print(f'Evaluation results saved at {name_file}')

    else:
        name_file = evaluate_and_predict(model, test_data, results_directory,
                                         results_id=new_results_id, output_name='test',
                                         analyze_data=analyze_data)

        print(f'Evaluation results saved at {name_file}')


def compile_model(model, learning_rate, optimizer='adam', loss='categorical_crossentropy',
                metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]):

    if optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()

    return model


def fit_model(name_model, dataset_dir, epochs=50, learning_rate=0.0001, results_dir=os.getcwd() + '/results/', backbone_model=None,
              val_dataset=None, eval_val_set=None, eval_train_set=False, test_data=None,
              batch_size=16, buffer_size=50, backbones=['restnet50'], dropout=False, after_concat='globalpooling'):
    if len(backbones) > 3:
        raise ValueError('number maximum of backbones is 3!')
    mode = ''.join(['fit_dop_', str(dropout), '_', after_concat, '_'])
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # Decide how to act according to the mode (train/predict/train-backbone... )
    files_dataset_directory = [f for f in os.listdir(dataset_dir)]
    if 'train' in files_dataset_directory:
        path_train_dataset = os.path.join(dataset_dir, 'train')
    else:
        path_train_dataset = dataset_dir

    if 'val' in files_dataset_directory:
        path_val_dataset = os.path.join(dataset_dir, 'val')
    elif val_dataset:
        path_val_dataset = val_dataset
    else:
        raise 'Validation directory not found'

    print(f'train directory found at: {path_train_dataset}')
    print(f'validation directory found at: {path_val_dataset}')

    if mode == 'pre_built_dataset_merge_features' or mode == 'pre_built_dataset_merge_predicts_v1':
        train_x, train_y, dictionary_train = load_data_from_directory(path_train_dataset)
        train_dataset = generate_tf_dataset(train_x, train_y, batch_size=batch_size, shuffle=True,
                                           buffer_size=buffer_size)

        val_x, val_y, dictionary_val = load_data_from_directory(path_val_dataset)
        val_dataset = generate_tf_dataset(val_x, val_y, batch_size=batch_size, shuffle=True,
                                           buffer_size=buffer_size)
    else:
        csv_file_train = [f for f in os.listdir(path_train_dataset) if f.endswith('.csv')].pop()
        path_csv_file_train = os.path.join(path_train_dataset, csv_file_train)
        train_x, dictionary_train = load_data_from_directory_v1(path_train_dataset,
                                                                         csv_annotations=path_csv_file_train)
        train_dataset = generate_tf_dataset_v1(train_x, dictionary_train, batch_size=batch_size, shuffle=True,
                                            buffer_size=buffer_size)

        for j, element in enumerate(train_dataset):
            print(j)
            print(element)
            x, y = element
            print(type(x))
            print(type(y))
            x_array = x.numpy()
            y_array = y.numpy()
            print('x:', np.shape(x_array), np.amin(x_array), np.amax(x_array))
            print('y:', np.shape(y_array), np.unique(y_array))

        dataset = train_dataset.batch(2, drop_remainder=True)

        csv_file_val = [f for f in os.listdir(path_val_dataset) if f.endswith('.csv')].pop()
        path_csv_file_val = os.path.join(path_val_dataset, csv_file_val)
        val_x, dictionary_val = load_data_from_directory_v1(path_val_dataset, csv_annotations=path_csv_file_val)
        val_dataset = generate_tf_dataset_v1(val_x, dictionary_val, batch_size=batch_size, shuffle=True,
                                          buffer_size=buffer_size)

    train_steps = len(train_x) // batch_size
    val_steps = len(val_x) // batch_size

    if len(train_x) % batch_size != 0:
        train_steps += 1
    if len(val_x) % batch_size != 0:
        val_steps += 1

    # define a dir to save the results and Checkpoints
    # if results directory doesn't exist create it
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    # ID name for the folder and results
    backbone_model = ''.join([name_model + '_' for name_model in backbones])
    new_results_id = generate_experiment_ID(name_model=name_model, learning_rate=learning_rate,
                                            batch_size=batch_size, backbone_model=backbone_model,
                                            mode=mode)

    results_directory = ''.join([results_dir, new_results_id, '/'])
    # if results experiment doesn't exists create it
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    # Build the model
    if len(backbones) == 1:
        backbones = backbones*3
    print(f'list backbones:{backbones}')
    if name_model == 'pre_built_dataset_merge_features':
        model = build_model(backbones=backbones, dropout=dropout, after_concat=after_concat)
    elif name_model == 'pre_built_dataset_merge_predicts_v1':
        model = build_model_v1(backbones=backbones, dropout=dropout, after_concat=after_concat)
    elif name_model == 'gan_merge_features':
        model = build_gan_model_features(backbones=backbones, gan_base='checkpoint_charlie',
                                        dropout=dropout, after_concat=after_concat)
    elif name_model == 'gan_merge_predicts_v1':
        model = build_gan_model_merge_out(backbones=backbones, gan_base='checkpoint_charlie',
                                         dropout=dropout, after_concat=after_concat)

    model = compile_model(model, learning_rate)
    temp_name_model = results_directory + new_results_id + "_model.h5"
    callbacks = [
        ModelCheckpoint(temp_name_model,
                        monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', patience=25),
        CSVLogger(results_directory + 'train_history_' + new_results_id + "_.csv"),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]

    # track time
    start_time = datetime.datetime.now()
    # Train the model

    trained_model = model.fit(train_dataset,
                              epochs=epochs,
                              shuffle=True,
                              batch_size=batch_size,
                              validation_data=val_dataset,
                              steps_per_epoch=train_steps,
                              validation_steps=val_steps,
                              verbose=True,
                              callbacks=callbacks)

    model.save(''.join([results_directory, 'model_', new_results_id]))

    print('Total Training TIME:', (datetime.datetime.now() - start_time))
    print('History Model Keys:')
    print(trained_model.history.keys())
    # in case evaluate val dataset is True
    if eval_val_set is True:
        evaluate_and_predict(model, val_dataset, results_directory,
                             results_id=new_results_id, output_name='val',
                             )

    if eval_train_set is True:
        evaluate_and_predict(model, train_dataset, results_directory,
                             results_id=new_results_id, output_name='train',
                             )

    if 'test' in files_dataset_directory:
        path_test_dataset = os.path.join(dataset_dir, 'test')
        print(f'Test directory found at: {path_test_dataset}')
        evalute_test_directory(model, path_test_dataset, results_directory, new_results_id,
                                   )
    #if test_data != '':
    #    evalute_test_directory(model, test_data, results_directory, new_results_id,
    #                           )

    
def make_dataset(path, batch_size):

  def parse_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    return image

  def configure_for_performance(ds):
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.repeat()
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

  path_imgs = list()
  images_class = list()
  images_domains = list()
  csv_annotations_file = [path + f for f in os.listdir(path) if f.endswith('.csv')].pop()
  list_files, dictionary_labels = load_data_from_directory_v1(path, csv_annotations=csv_annotations_file)
  for img_name in list_files:
    path_imgs.append(dictionary_labels[img_name]['path_file'])
    images_class.append(dictionary_labels[img_name]['img_class'])
    images_domains.append(dictionary_labels[img_name]['img_domain'])

  unique_domains = list(np.unique(images_domains))
  unique_classes = list(np.unique(images_class))
  images_domains = [unique_domains.index(val) for val in images_domains]
  images_class = [unique_classes.index(val) for val in images_class]

  list_path_files = list()
  for (dirpath, dirnames, filenames) in os.walk(path):
      list_path_files += [os.path.join(dirpath, file) for file in filenames]

  list_path_files = [f for f in list_files if f.endswith('.png')]
  labels = [unique_classes.index(name.split('/')[-2]) for name in list_path_files]

  filenames_ds = tf.data.Dataset.from_tensor_slices(list_path_files)
  images_ds = filenames_ds.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  labels_ds = tf.data.Dataset.from_tensor_slices(labels)
  ds = tf.data.Dataset.zip((images_ds, labels_ds))
  ds = configure_for_performance(ds)

  return ds

def train_model(name_model, dataset_dir, epochs=50, learning_rate=0.0001, results_dir=os.getcwd() + '/results/', backbone_model=None,
              val_dataset=None, eval_val_set=None, eval_train_set=False, test_data=None,
              batch_size=16, buffer_size=50, backbones=['restnet50'], dropout=False, after_concat='globalpooling'):

    if len(backbones) > 3:
        raise ValueError('number maximum of backbones is 3!')
    mode = ''.join(['fit_dop_', str(dropout), '_', after_concat, '_'])
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # Decide how to act according to the mode (train/predict/train-backbone... )
    files_dataset_directory = [f for f in os.listdir(dataset_dir)]
    if 'train' in files_dataset_directory:
        path_train_dataset = os.path.join(dataset_dir, 'train')
    else:
        path_train_dataset = dataset_dir

    if 'val' in files_dataset_directory:
        path_val_dataset = os.path.join(dataset_dir, 'val')
    elif val_dataset:
        path_val_dataset = val_dataset
    else:
        raise 'Validation directory not found'

    print(f'train directory found at: {path_train_dataset}')
    print(f'validation directory found at: {path_train_dataset}')

    train_x, train_y, dictionary_train = load_data_from_directory(path_train_dataset)
    train_dataset = generate_tf_dataset(train_x, train_y, batch_size=batch_size, shuffle=True,
                                       buffer_size=buffer_size)

    val_x, val_y, dictionary_val = load_data_from_directory(path_val_dataset)
    val_dataset = generate_tf_dataset(val_x, val_y, batch_size=batch_size, shuffle=True,
                                       buffer_size=buffer_size)

    train_steps = len(train_x) // batch_size
    val_steps = len(val_x) // batch_size

    if len(train_x) % batch_size != 0:
        train_steps += 1
    if len(val_x) % batch_size != 0:
        val_steps += 1

    # define a dir to save the results and Checkpoints
    # if results directory doesn't exist create it
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    # ID name for the folder and results
    backbone_model = ''.join([name_model + '_' for name_model in backbones])
    new_results_id = generate_experiment_ID(name_model=name_model, learning_rate=learning_rate,
                                            batch_size=batch_size, backbone_model=backbone_model,
                                            mode=mode)

    results_directory = ''.join([results_dir, new_results_id, '/'])
    # if results experiment doesn't exists create it
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    # Build the model
    if len(backbones) == 1:
        backbones = backbones*3
    print(f'list backbones:{backbones}')
    if name_model == 'gan_merge_features':
        model = build_model(backbones=backbones, dropout=dropout, after_concat=after_concat)
    elif name_model == 'gan_merge_predicts_v1':
        model = build_model_v1(backbones=backbones, dropout=dropout, after_concat=after_concat)

    model = compile_model(model, learning_rate)


    # Instantiate a loss function.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Prepare the metrics.
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            train_acc_metric.update_state(y_batch_train, logits)

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %d samples" % ((step + 1) * batch_size))

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))


def predict(directory_model, file_to_predict):

    model, _ = load_model(directory_model)
    #if daa.check_file_isvid(file_to_predic):
    #    pass
    # elif os.path.isdir(file_to_predic):
    if os.path.isdir(file_to_predict):
        new_results_id = generate_experiment_ID(prediction_model=os.path.basename(os.path.normpath(directory_model)))
        results_directory = directory_model
        print(f'Test directory found at: {file_to_predict}')
        evalute_test_directory(model, file_to_predict, results_directory, new_results_id,
                               )
    elif file_to_predict == 'webcam':
        pass

    else:
        print(f'Format or dir {file_to_predict} not understood')

    pass


def main(_argv):

    name_model = FLAGS.name_model
    mode = FLAGS.mode
    train_dataset = FLAGS.dataset_dir
    test_dataset = FLAGS.test_dataset
    val_dataset = FLAGS.val_dataset
    backbone_model = FLAGS.backbone
    batch_size = FLAGS.batch_size
    buffer_size = FLAGS.buffer_size
    epochs = FLAGS.epochs
    directory_model = FLAGS.directory_model
    file_to_predic = FLAGS.file_to_predic

    if mode == 'analyze_dataset':
        analyze_tf_dataset(test_dataset)

    elif mode == 'fit':
        fit_model(name_model, train_dataset, val_dataset=val_dataset, epochs=epochs)
        #fit_model(name_model, train_dataset, backbone_model, val_dataset=val_dataset, batch_size=batch_size,
        #          buffer_size=buffer_size)
    elif mode == 'predict':
        predict(directory_model, file_to_predic)


if __name__ == '__main__':

    flags.DEFINE_string('name_model', 'gan_merge_features', 'name of the model')
    flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf', 'analyze_dataset', 'predict'],
                      'fit: model.fit, '
                      'eager_fit: model.fit(run_eagerly=True), '
                      'predict: predict in a dataset given a model'
                      'eager_tf: custom GradientTape,'
                      'analyze_dataset: analyze_dataset')
    flags.DEFINE_string('backbone', None, 'backbone network')
    flags.DEFINE_string('dataset_dir', os.getcwd() + 'data/', 'path to dataset')
    flags.DEFINE_string('val_dataset', None, 'path to validation dataset')
    flags.DEFINE_string('test_dataset', '', 'path to test dataset')
    flags.DEFINE_string('results_dir', os.getcwd() + 'results/', 'path to dataset')
    flags.DEFINE_integer('epochs', 1, 'number of epochs')
    flags.DEFINE_integer('batch_size', 4, 'batch size')
    flags.DEFINE_integer('buffer_size', 500, 'buffer size when shuffle dataset')
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    flags.DEFINE_string('weights', '', 'path to weights file')
    flags.DEFINE_bool('analyze_data', False, 'select if analyze data or not')
    flags.DEFINE_string('directory_model', '', 'indicate the path to the directory')
    flags.DEFINE_float('validation_split', 0.2, 'iif not validation dir but needed')
    flags.DEFINE_string('file_to_predic', '',
                        'Directory or file where to perform predictions if predict mode selected')
    flags.DEFINE_integer('trainable_layers', -1, 'Trainable layers in case backbone is trained')


    try:
        app.run(main)
    except SystemExit:
        pass