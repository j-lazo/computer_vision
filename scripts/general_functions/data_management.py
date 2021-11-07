import os
import scipy.io
from matplotlib import pyplot as plt
import numpy as np
import cv2
import tqdm
import pandas as pd

def discrepancies(file_1, file_2):
    """
    Check for discrepancies in the annotations between two datasets files.

    :param file_1: (str) path to the file 1 to compare
    :param file_2: (str) path to the file 2 to compare
    :return: (list) list of the elements in both files with discrepancies, and its classes.
    """

    list_discrepancies = []

    df1 = read_annotation_file(file_1)
    df2 = read_annotation_file(file_2)

    # 2DO: Read each file and iterate over the elements and classes that are shared. Then save
    # those which has different information

    return list_discrepancies


def create_annotations_file(directory, file_extension='.csv'):
    """
    Creates an annotation file according to the structure of the directory tree. The output files could be
    '.csv' or '.json'. The function only considers up to a maximum of two sub-directories.

    :param directory: (str) path to the directory to analyze
    :param file_extension: (str) .csv or .json
    :return: a file with the annotations according to the structure of the directory
    """

    dictionary = {}
    list_subfolders_with_paths = [f.path for f in os.scandir(directory) if f.is_dir()]
    for sub_folder in list_subfolders_with_paths:
        sub_sub_folders = [f for f in os.listdir(directory) if f.is_dir()]
        if sub_sub_folders:
            for 



def read_annotation_file(annotations_dir):
    """
    Given the path of the annotations of a dataset in either '.csv' or '.json' reads the data and
    return it in pandas data frame format

    :param annotations_dir: reads the annotations of a dataset from a .csv or .json
    :return: pandas data frame
    """
    if annotations_dir.endswith('.csv'):
        db = pd.read_csv(annotations_dir)
    elif annotations_dir.endswith('.json'):
        db = pd.read_json(annotations_dir)
    else:
        print('annotations format not compatible')

    return db


def build_dictionary_from_directory(path_dir, build_from_directory_tree=True, path_annotation_file=''):
    """
    Builds a dictionary given a path directory, if the directory contains more than one sub-folder it
    will assume that the structure in the directory corresponds to the different classes existing
    in the dataset. Otherwise it will look for the first .csv or .json file in that directory and
    consider it as the one with the annotations, if the annotations correspond to one specific file,
    or they are located in a different folder, the path can be indicated with 'annotation_file_path'

    :param path_dir: (str) path of the directory from which to build the dictionary
    :param build_from_directory_tree: (bool) The dataset is arranged in different folders according to
        its classes (True) otherwise (False)
    :param annotation_file_dir: (str) .csv or .json file containing the annotations of the dataset
    :return: (dict) Dictionary that maps the elements in the path (x) with its corresponding annotations (y)
    """

    dictionary = {}
    if build_from_directory_tree is True:
        print(f'More than 1 sub-folder found in the directory: {path_dir}')
        print('It will be assumed that the sub-folders found correspond to the different classes of the dataset')
        list_subfolders_with_paths = [f.path for f in os.scandir(path_dir) if f.is_dir()]
        for sub_folder in list_subfolders_with_paths:
            list_imgs = os.listdir(os.path.join(sub_folder))
            for image in list_imgs:
                dictionary.update({image:{'image_dir':os.path.join(sub_folder, image),
                                          'classification':sub_folder}})

    else:
        if path_annotation_file != '':
            annotation_file = [file for file in os.listdir if file.endswith('.csv') or file.endswith('.json')][0]
        # 2DO: add read database and compare with the existing images,
        # then build dictionary according to the images that actually exists
        database = read_annotation_file(annotation_file)

    return dictionary


def determine_if_subfolders_exists(path_dir):
    """
    Check a directory and determine if there is more than one sub-folder in the directory,
    If more than one exists returns True, otherwise it returns False

    :param path_dir: (str) path to analyze
    :return: (bool)
    """
    list_sub_folfers = [folder for folder in os.listdir(path_dir) if os.path]
    if len(list_sub_folfers) > 1:
        return True
    else:
        return False

def build_dictionary_data_labels(path_dir, path_annotation_file=''):

    """
    Build a dictionary given a path or a list of paths to different directories.

    :param path_dir: (str or list)
    :param annotation_file_dir: (str or list)
    :return:
    """

    dictionary = {}
    if type(path_dir) == str:

        sub_tree_structures = determine_if_subfolders_exists(path_dir)
        dictionary.update(build_dictionary_from_directory(path_dir, sub_tree_structures, path_annotation_file=path_annotation_file))

    elif type(path_dir) == list:
        for folder in path_dir:
            sub_tree_structures = determine_if_subfolders_exists(path_dir)
            dictionary.update(build_dictionary_from_directory(folder, sub_tree_structures, path_annotation_file=path_annotation_file))

    else:
        print('data type variable: path_dir, not recognized')

    return dictionary


def read_mat_files(file_dir):
    """
    Read a .mat file
    :param file_dir: (str) directory where the .mat file is located
    :return: (dict) the mat file in dictionary format
    """

    mat = scipy.io.loadmat(file_dir)
    return mat


def analysze_dataset(path_dataset):
    """
    Chek all the original image sizes inside a directory and its sub-directories

    :param path_dataset: (str) path of the dataset to analyze
    :return: (list) list with all the unique image sizes
    """

    print(f'path Dataset {path_dataset}')
    shape_imgs = []
    list_dirs = sorted([x[0] for x in os.walk(path_dataset)])
    print(list_dirs)
    for i, folder in enumerate(list_dirs):
        list_imgs = [image for image in os.listdir(folder) if (image.endswith('.png') or image.endswith('.jpg'))]
        if list_imgs:
            print(f'folder: {i}/{len(list_dirs)} {folder}')
            for j, image in enumerate(tqdm.tqdm(list_imgs, desc= f'analyzing {len(list_imgs)} images')):
                shape_img = cv2.imread(os.path.join(folder, image)).shape
                if shape_img not in shape_imgs:
                    shape_imgs.append(shape_img)
            print(shape_imgs)

    print(shape_imgs)


def visualize_roimat_and_image(file_dir, original_folder=os.getcwd()):

    """
    :param file_dir:
    :param original_folder:
    :return:
    """

    mat_file_dir = file_dir + 'roi.mat'
    mat = read_mat_files(mat_file_dir)
    list_frames = os.listdir(file_dir + 'frame/')
    list_masked_frames = os.listdir(file_dir + 'maskedframe/')
    print(len(list_frames), len(list_masked_frames))
    keys = [x for x in mat.keys()]
    print('keys:', keys)
    print(mat['fovmask'], np.shape(mat['fovmask']))
    """fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    image = cv2.imread(')
    ax1.imshow(image)
    ax1.plot(mat['fovmask'][0][0], mat['fovmask'][0][1], 'r*')
    ax1.plot(mat['fovmask'][0][0], mat['fovmask'][0][1] + mat['fovmask'][0][3], 'r*')
    ax1.plot(mat['fovmask'][0][0] + mat['fovmask'][0][2], mat['fovmask'][0][1], 'r*')
    ax1.plot(mat['fovmask'][0][0] + mat['fovmask'][0][2], mat['fovmask'][0][1] + mat['fovmask'][0][3], 'r*')
    plt.show()"""

def compare_images(image_1, image_2):

    """
    Compares if two image are the same, no matter their size. If one image is bigger than the other
    it will assume that one of the smaller image is a sub-crop of the second one and will verify that
    the small image corresponds to a part of the big one.

    :param image_1: (array) Img array
    :param image_2: (array) Img array
    :return:
    """


    def compare_imgs(image_1, image_2):
        difference = cv2.subtract(image_1, image_2)
        b, g, r = cv2.split(difference)
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            return True

    if image_1.shape == image_2.shape:
        compare_imgs(image_1, image_2)
    else:
        size_img1 = np.shape(image_1)[0]*np.shape(image_1)[1]
        size_img2 = np.shape(image_2)[0]*np.shape(image_2)[1]

        if size_img1 < size_img2:
            small_img = image_1
            big_img = image_2
        else:
            big_img = image_1
            small_img = image_2

        w_simage, l_simage, d_simage = small_img.shape
        w_bimage, l_bimage, d_bimage = big_img.shape

    for i in range(0, l_bimage-l_simage):
        for j in range(0, w_bimage-w_simage):
            print(i, j)
            #print(i, j) # the numbers bellow need to be changed
            croped_image = big_img[j:j+round(719.7607), i:i+round(704.4466)]
            resized_img = cv2.resize(croped_image, (l_simage, w_simage))
            if compare_imgs(resized_img, small_img) is True:
                print(i, j)
                break

    return 0





