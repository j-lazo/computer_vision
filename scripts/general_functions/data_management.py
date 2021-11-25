import os
import scipy.io
from matplotlib import pyplot as plt
import numpy as np
import cv2
import tqdm
import pandas as pd
import keras
import tensorflow as tf
import ast


def generate_mask_from_points(img, mask_points):
    """

    :param img: (image)
    :param mask_points: (list with tuples)
    :return: (array) binary image
    """
    img_shape = np.shape(img)
    mask = np.zeros([img_shape[0], img_shape[1]])

    # in case there is more than one polygon
    if np.shape(mask_points)[0] > 1:
        for mask_point in mask_points:
            mp = []
            mp.append(mask_point)
            mp = np.array(mp, dtype=np.int32)
            drawing = cv2.fillPoly(mask, mp, color=255)

    else:
        mask_point = np.array(mask_points, dtype=np.int32)
        drawing = cv2.fillPoly(mask, mask_point, color=255)

    return drawing


def convert_cvs_data_to_imgs(base_directory, csv_file_dir='', directory_imgs='', output_directory='masks'):

    """
    Given a directory looks for a csv vile and the images in it's corresponding images.
    The default directory to look for images in "images" inside the base directory, it can be customized with its respective flag.
    The csv file should be in VGG annotator format.
    The masks are saved in png format. A folder "masks" is created if it doesn't exists.

    :param base_directory: (str)
    :param csv_file_dir: (str)
    :param directory_imgs: (str)
    :param output_directory: (str)
    :return:
    """
    if output_directory == 'masks':
        output_directory = base_directory + 'masks'

    # check the output directory exists
    if not(os.path.isdir(output_directory)):
        os.mkdir(output_directory)

    if directory_imgs == '':
        directory_imgs = base_directory + '/images/'

    list_images = [file for file in os.listdir(directory_imgs) if file.endswith('.png')]
    if csv_file_dir == '':
        csv_file = [f for f in os.listdir(base_directory) if f.endswith('.csv')][0]
        csv_file_dir = base_directory + csv_file

    data_frame = pd.read_csv(csv_file_dir)
    list_name_imgs = data_frame['filename'].tolist()
    list_points = data_frame['region_shape_attributes'].tolist()
    for img in list_images:
        print('img:', img)
        if img in list_name_imgs:
            indexes = [i for i, element in enumerate(list_name_imgs) if element == img]
            contours = []
            for index in indexes:
                list_points[index]
                res = ast.literal_eval(list_points[index])
                if res!= {}:
                    points_x = res.get('all_points_x')
                    points_y = res.get('all_points_y')
                    contour = []
                    for i, x in enumerate(points_x):
                        contour.append([[x, points_y[i]]])
                        array_contour = np.array(contour, dtype=np.int32)

                    contours.append(array_contour)
                    image = cv2.imread(directory_imgs + img)
                    mask = generate_mask_from_points(image, contours)
                    mask_name = ''.join([output_directory, '/', img])
                    cv2.imwrite(mask_name, mask)
        else:
            mask = np.zeros(np.shape(img))
            cv2.imwrite(''.join([output_directory, '/', img]), mask)

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


def update_dictionary(dictionary, image_name, utility='', clearness='', resolution='', procedure='', imaging_type='',
                      type_artifact='', tissue_type='', fov_shape='', roi=[]):

    """
    updates a dictionary and its keys, the only mandatory parameter is 'image_name'

    :param dictionary: (dict)
    :param image_name: (str)
    :param utility: (str)
    :param clearness: (str)
    :param resolution: (str)
    :param procedure: (str)
    :param imaging_type: (str)
    :param type_artifact: (str)
    :param tissue_type: (str)
    :param fov_shape: (str) Field of View Shape
    :param roi: (list) Region of Interest
    :return: (dict) dictionary with the update entry
    """

    if procedure == '':
        if image_name[:3] == 'urs':
            procedure = 'ureteroscopy'
        elif image_name[:3] == 'cys':
            procedure = 'cystoscopy'

    dictionary.update({image_name: {'useful': utility, 'clear': clearness, 'resolution': resolution,
                                    'procedure': procedure, 'imaging type': imaging_type, 'type artifact': type_artifact,
                                    'tissue type': tissue_type, 'fov shape': fov_shape, 'ROI': roi}})
    return dictionary


def create_annotations_file(directory, file_name='', file_extension='.csv'):
    """
    Creates an annotation file according to the structure of the directory tree. The output files could be
    '.csv' or '.json'. The function only considers up to a maximum of two sub-directories.

    :param directory: (str) path to the directory to analyze
    :param file_name: (str) name to save the file
    :param file_extension: (str) .csv or .json
    :return: a file with the annotations according to the structure of the directory
    """

    dictionary = {}
    list_subfolders_with_paths = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    for sub_folder in list_subfolders_with_paths:
        sub_folder_dir = os.path.join(directory, sub_folder)
        sub_sub_folders = [f for f in os.listdir(sub_folder_dir) if os.path.isdir(os.path.join(sub_folder_dir, f))]
        if sub_sub_folders:
            print(f'sub folder found: {sub_sub_folders} in {sub_folder}')
            for sub_sub_folder in sub_sub_folders:
                list_imgs = sorted(os.listdir(os.path.join(directory, sub_folder, sub_sub_folder)))
                for i, image_name in enumerate(tqdm.tqdm(list_imgs, desc='Reading images')):
                    imaging_type = sub_sub_folder
                    tissue_type = sub_folder
                    dictionary = update_dictionary(dictionary, image_name, imaging_type=imaging_type, tissue_type=tissue_type)


        else:
            print(f'no sub-folders found in {sub_folder}')
            list_imgs = sorted(os.listdir(os.path.join(directory, sub_folder)))
            for i, image_name in enumerate(tqdm.tqdm(list_imgs, desc='Reading images')):
                imaging_type = 'WLI'
                tissue_type = sub_folder
                dictionary = update_dictionary(dictionary, image_name, imaging_type=imaging_type, tissue_type=tissue_type)

    df = pd.DataFrame(data=dictionary).T
    name_file = save_data_frame_to_file(df, file_name, file_extension, directory)
    print(f'data file saved at {name_file}')


def save_data_frame_to_file(data_frame, file_name, file_extension, directory=os.getcwd()):
    """
    Save a data frame object into a .csv or .json

    :param data_frame: Pandas data frame
    :param file_name: (str) name of the file
    :param file_extension: either .json or .csv
    :param directory: directory where to save the file. Default value: current directory
    :return: name of the path where the file was saved
    """

    def save_file(file_path, file_extension):

        if file_extension == '.csv':
            data_frame.to_csv(file_path)
        elif file_extension == '.json':
            data_frame.to_json(file_path)
        else:
            print('File format not recognized')

    if file_name == '':
        file_path = ''.join([directory, '/', 'annotations', file_extension])
    else:
        file_path = ''.join([directory, '/', file_name, file_extension])

    if not(os.path.isfile(file_path)):
        save_file(file_path, file_extension)
    else:
        list_files = [f for f in os.listdir(directory) if file_name in f]
        file_path = ''.join([directory, '/', 'annotations', '_',str(len(list_files)+1), file_extension])

    save_file(file_path,file_extension)

    return file_path


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
    :param path_annotation_file: (str) .csv or .json file containing the annotations of the dataset
    :return: (dict) Dictionary that maps the elements in the path (x) with its corresponding annotations (y)
    """

    dictionary = {}
    if build_from_directory_tree is True:
        print(f'More than 1 sub-folder found in the directory: {path_dir}')
        print('It will be assumed that the sub-folders found correspond to the different classes of the dataset')
        #list_subfolders_with_paths = [f.path for f in os.scandir(path_dir) if f.is_dir()]
        list_subfolders = [f for f in os.listdir(path_dir) if os.path.isdir(os.path.join(path_dir, f))]
        for sub_folder in list_subfolders:
            sub_folder_dir = os.path.join(path_dir, sub_folder)
            list_imgs = os.listdir(sub_folder_dir)
            for image in list_imgs:
                dictionary.update({image:{'image_dir':os.path.join(sub_folder_dir, image),
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
    :return: (dict)
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


def analyze_dataset(path_dataset):
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





