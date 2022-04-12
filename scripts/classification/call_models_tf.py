import os

import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import tensorflow as tf


def load_data_from_directory(path_data):
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
    list_unique_classes = np.unique([f for f in list_files if os.path.isdir(os.path.join(path_data, f))])
    for j, unique_class in enumerate(list_unique_classes):
        path_images = ''.join([path_data, unique_class, '/*'])
        added_images = sorted(glob(path_images))
        new_dictionary_labels = {image_name: unique_class for image_name in added_images}

        images_path = images_path + added_images
        added_labels = [j] * len(added_images)
        labels = labels + added_labels
        dictionary_labels = {**dictionary_labels, **new_dictionary_labels}

    return images_path, labels, dictionary_labels


def read_stacked_images_npy(path_data, preprocessing_input=None):

    #if path_data.endswith('.npz'):
    #    img_array = np.load(path_data)
    #    img = img_array['arr_0']
    #else:
    #    img = np.load(path_data)
    img_array = np.load(path_data)
    img = img_array['arr_0']

    if preprocessing_input == 'inception_v3':
        print('simon')
        img = tf.keras.applications.inception_v3.preprocess_input(img)
    elif preprocessing_input == 'resnet50':
        img = tf.keras.applications.inception_v3.preprocess_input(img)

    else:
        img = img/255.

    return img


def tf_parser_npy(x, y):

    def _parse(x, y):
        x = read_stacked_images_npy(x, preprocessing_input=PREPROCESS_FUNCTION)
        y = np.zeros(1) + y

        print(type(y))
        print(np.shape(y))
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([3, 256, 256, 3])
    y.set_shape([1])
    return x, y


def generate_tf_dataset(x, y, batch_size=1, shuffle=False, buffer_size=10, preprocess_function=None):
    """

    Parameters
    ----------
    x :
    y :
    batch_size :
    shuffle :

    Returns
    -------

    """
    global PREPROCESS_FUNCTION
    PREPROCESS_FUNCTION = preprocess_function
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size * batch_size)

    dataset = dataset.map(tf_parser_npy)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()

    return dataset


def analyze_tf_dataset(dataset_dir):

    #
    test_x, test_y, dataset_dictionary = load_data_from_directory(dataset_dir)
    test_dataset = generate_tf_dataset(test_x, test_y, batch_size=8, shuffle=True,
                                       buffer_size=500)

    unique_labels = np.unique([f for f in os.listdir(dataset_dir) if os.path.isdir(dataset_dir + f)])
    for j, element in enumerate(test_dataset):
        x, y = element
        x_array = x.numpy()
        y_array = y.numpy()
        print(y)
        plt.figure()
        plt.subplot(3, 4, 1)
        plt.imshow(x_array[0][0])
        plt.axis('off')
        plt.gca().set_title(str(unique_labels[int(y_array[0][0])]))
        plt.subplot(3, 4, 2)
        plt.imshow(x_array[1][0])
        plt.axis('off')
        plt.gca().set_title(str(unique_labels[int(y_array[1][0])]))
        plt.subplot(3, 4, 3)
        plt.imshow(x_array[2][0])
        plt.axis('off')
        plt.gca().set_title(str(unique_labels[int(y_array[2][0])]))
        plt.subplot(3, 4, 4)
        plt.imshow(x_array[3][0])
        plt.axis('off')
        plt.gca().set_title(str(unique_labels[int(y_array[3][0])]))

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


if __name__ == "__main__":

    path_dataset = ''
    analyze_tf_dataset(path_dataset)
