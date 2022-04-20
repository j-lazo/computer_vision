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
    csv_indir = [f for f in list_files if f.endswith('.csv')].pop()
    if csv_annotations:
        data_frame = pd.read_csv(csv_annotations)
    elif csv_indir:
        data_frame = pd.read_csv(os.path.join(path_data, csv_indir))

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
    dataset = dataset.map(tf_parser_npy)
    dataset = dataset.batch(batch_size)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size * batch_size)
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
    sub_dirs = os.listdir(test_data)
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
    elif name_model = 'gan_merge_predictions_v1':
        model = build_model_v1(backbones=backbones, dropout=dropout, after_concat=after_concat)
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

    if mode == 'analyze_dataset':
        analyze_tf_dataset(test_dataset)

    elif mode == 'fit':
        fit_model(name_model, train_dataset, val_dataset=val_dataset, epochs=epochs)
        #fit_model(name_model, train_dataset, backbone_model, val_dataset=val_dataset, batch_size=batch_size,
        #          buffer_size=buffer_size)


if __name__ == '__main__':

    flags.DEFINE_string('name_model', 'gan_merge_features', 'name of the model')
    flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf', 'analyze_dataset'],
                      'fit: model.fit, '
                      'eager_fit: model.fit(run_eagerly=True), '
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