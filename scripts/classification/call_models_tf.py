import os
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

    list_files = os.listdir(path_data)
    list_unique_classes = np.unique([f for f in list_files if os.path.isdir(os.path.join(path_data, f))])
    for j, unique_class in enumerate(list_unique_classes):
        path_images = ''.join([path_data, unique_class, '/*'])
        added_images = sorted(glob(path_images))
        images_path = images_path + added_images
        added_labels = [j] * len(added_images)
        labels = labels + added_labels

    return images_path, labels


def read_stacked_images_npy(path_data, preprocessing_input=None):
    img = np.load(path_data)
    if preprocessing_input == 'inception_v3':
        print('simon')
        img = tf.keras.applications.inception_v3.preprocess_input(img)
    elif preprocessing_input == 'resnet50':
        img = tf.keras.applications.inception_v3.preprocess_input(img)

    else:
        img = img/255.

    print(np.shape(img))
    return img


def tf_parser_npy(x, y):

    def _parse(x, y):
        x = read_stacked_images_npy(x, preprocessing_input=PREPROCESS_FUNCTION)
        y = y
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
    dataset = dataset.map(tf_parser_npy)
    dataset = dataset.batch(batch_size)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size * batch_size)
        dataset = dataset.repeat()

    return dataset


def main(dataset_dir):
    data_x, data_y = load_data_from_directory(dataset_dir)
    val_dataset = generate_tf_dataset(data_x, data_y, batch_size=8, preprocess_function='inception_v3', shuffle=True)
    for x in val_dataset.take(1):
        print(type(x))

    #x = next(val_dataset)
    #print(x)


if __name__ == "__main__":
    direct = ''
    main(direct)
