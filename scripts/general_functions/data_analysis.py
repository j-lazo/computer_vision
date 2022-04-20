from matplotlib import pyplot as plt
import csv
from absl import app, flags, logging
from absl.flags import FLAGS
import os
import scipy.io
import numpy as np
import cv2
import tqdm
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import pandas as pd
import seaborn as sns
import datetime
import glob
import re
import string
import sys
import cv2
import re
import ast
import shutil
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
import copy
import collections


def _check_ext(path, default_ext):
    name, ext = os.path.splitext(path)
    if ext == '':
        if default_ext[0] == '.':
            default_ext = default_ext[1:]
        path = name + '.' + default_ext
    return path


def save_yaml(path, data, **kwargs):

    import oyaml as yaml
    path = _check_ext(path, 'yml')

    with open(path, 'w') as f:
        yaml.dump(data, f, **kwargs)

def convert_categorical_str_to_numerical(category_list):
    """
    Takes a category list of strings and converts it to integers, e.g:
    category_list = [dog, cat, horse, dog, cow]
    return: [0, 1, 2, 0, 3]

    :param category_list: (list) list of string categories
    :return: (list)
    """

    unique = list(np.unique(category_list))
    return [unique.index(u) for u in category_list]


def match_pair_of_data(data_file_1, data_file_2):
    """
    matches pairs of data from two csv files
    :param data_file_1: (str) CSV file absolute path
    :param data_file_2: (str) CSV file absolute path
    :return: (list, list) list of numerical values for a list of inputs that matches name
    """
    y_test = []
    y_pred = []

    data_file1 = pd.read_csv(data_file_1)
    data_file2 = pd.read_csv(data_file_2)

    gt_categories = convert_categorical_str_to_numerical(data_file2['tissue type'].tolist())
    gt_file_names = data_file2['image_name'].tolist()
    predict_fnames = data_file1['fname'].tolist()
    predict_categories = data_file1['class_2'].tolist()

    print(f'found {len(gt_file_names)} cases in file 1 and {len(predict_fnames)} cases in file 2')

    for i, name in enumerate(predict_fnames):
        if name in gt_file_names:
            y_pred.append(float(predict_categories[i]))
            y_test.append(float(gt_categories[gt_file_names.index(name)]))

    print(f'{len(y_test)} cases matched names')
    return y_test, y_pred


def calculate_auc_and_roc(predicted, real, case_name, plot=True, results_directory='',
                          results_id='', save_plot=False):
    """

    :param predicted:
    :param real:
    :param case_name:
    :param plot:
    :param results_directory:
    :param results_id:
    :param save_plot:
    :return:
    """
    y_test, y_pred = match_pair_of_data(predicted, real)
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)
    auc_keras = auc(fpr_keras, tpr_keras)

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label=case_name + '(area = {:.3f})'.format(auc_keras))
    # plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    if save_plot is True:
        name_fig = ''.join(['roc_', results_id, '_.png'])
        plt.savefig(results_directory + name_fig)
    if plot is True:
        plt.show()

    plt.close()

    return auc_keras


def check_file_isvid(filename):
    """
    checks if a file has a video extension, accepted files are: '.mp4', '.mpg', '.avi'
    :param filename: (str) name of the file
    :return: (bool)
    """
    list_extensions = ['.mpg', '.MPG', '.mp4', '.MP4', '.AVI', '.avi']
    if filename[-4:] in list_extensions:
        return True
    else:
        return False


def get_video_files_in_dir(dir_dataset):
    """
    Given a directory checks if there are any video files and return the absolute path of the video files in a list
    :param dir_dataset: (str) directory
    :return: (list) list of video files
    """

    initial_list_files = os.listdir(dir_dataset)
    list_folders = []
    list_video_files = []

    for file_name in initial_list_files:
        if os.path.isdir(dir_dataset + file_name):
            list_folders.append(file_name)
        else:
            if file_name[-4:] not in list_video_files:
                list_video_files.append(dir_dataset + file_name)

    for folder in list_folders:
        list_files = os.listdir(dir_dataset + folder)
        for file_name in list_files:
            if file_name[-4:] not in list_video_files:
                list_video_files.append(''.join([dir_dataset, folder, '/', file_name]))

    return list_video_files


def analyze_video_dataset(dir_dataset):
    """
    Analyzes a dataset of video showing the number of frames of each video
    :param dir_dataset: (str) directory of the dataset
    :return:
    """
    list_video_files = get_video_files_in_dir(dir_dataset)
    print(f"found {len(list_video_files)} video files")
    num_frames = []
    name_videos = []
    for path_to_video in list_video_files:
        cap = cv2.VideoCapture(path_to_video)
        name_videos.append(path_to_video.replace(dir_dataset, ''))
        num_frames.append(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    df = pd.DataFrame(data={"name file": name_videos, "num frames": num_frames})


def find_pattern_names(string_name, str_pattern):
    """
    Looks for a pattern name in a string and returns the number after it
    :param string_name: the string where to look for a pattern
    :param str_pattern: the pattern that needs to be found
    :return:
    """
    match = re.search(str_pattern + '(\d+)', string_name)
    if match:
        return match.group(1)
    else:
        return np.nan


def determine_type_procedure(file_name):
    """
    Determine which type of procedure is according to the name of the file
    :param file_name:
    :return:
    """

    types_procedures = ['cys', 'urs']
    for kind in types_procedures:
        if kind in file_name:
            return kind


def analyze_dataset_patterns(dataset_dir, pattern_str):
    """
    Analyze a dataset to find a patter after a string
    :param dataset_dir:
    :param pattern_str:
    :return:
    """
    list_files = os.listdir(dataset_dir)
    unique_names = []
    for file_name in list_files:
        pattern = find_pattern_names(file_name, pattern_str)
        type_procedure = determine_type_procedure(file_name)
        combination = [type_procedure, pattern]
        if combination not in unique_names:
            unique_names.append(combination)

    return unique_names


def read_mask(dir_image):
    """
    :param dir_image:
    :return:
    """
    original_img = cv2.imread(dir_image)

    if original_img is None:
        print('Could not open or find the image:', dir_image)
        exit(0)

    height, width, depth = original_img.shape
    img = cv2.resize(original_img, (256, 256))
    img = img / 255
    img = (img > 0.9) * 1.0
    return img


def read_img_results(dir_image):
    """

    :param dir_image:
    :return:
    """
    original_img = cv2.imread(dir_image)

    if original_img is None:
        print('Could not open or find the image:', dir_image)
        exit(0)

    height, width, depth = original_img.shape
    img = cv2.resize(original_img, (256, 256))

    return img


def compare_box_plots(general_directory = '', name_test_csv_file='', name_validation_csv_file='',
                      save_directory='', condition_name=''):

    """
    :param general_directory:
    :param name_test_csv_file:
    :param name_validation_csv_file:
    :param save_directory:
    :param condition_name:
    :return:
    2DO: Handle list of dirs and exclusion conditions
    """

    predictions_path = ''.join([general_directory, 'predictions'])
    prediction_folders = sorted([f for f in os.listdir(predictions_path)])
    file_names = []
    dsc_values = {}
    prec_values = {}
    rec_values = {}
    acc_values = {}
    if general_directory != '' and type(general_directory) == str:
        csv_files = sorted([f for f in os.listdir(general_directory) if 'evaluation_results' in f and f.endswith('.csv')])
        print(csv_files)
        count_id = 0
        for i, folder in enumerate(prediction_folders):
            if folder in csv_files[i]:
                file_names.append(folder)
            else:
                file_names.append('dataset_'+str(count_id))
                count_id =+1
            data_file = pd.read_csv(general_directory + csv_files[i])
            dsc_values[file_names[i]] = data_file['DSC'].tolist()
            prec_values[file_names[i]] = data_file['Precision'].tolist()
            rec_values[file_names[i]] = data_file['Recall'].tolist()
            acc_values[file_names[i]] = data_file['Accuracy'].tolist()

    else:
        pass

    dsc_data = pd.DataFrame.from_dict(dsc_values, orient='index').T
    prec_data = pd.DataFrame.from_dict(prec_values, orient='index').T
    rec_data = pd.DataFrame.from_dict(rec_values, orient='index').T
    acc_data = pd.DataFrame.from_dict(acc_values, orient='index').T

    # build the image to plot

    fig1 = plt.figure(1, figsize=(11,7))
    ax1 = fig1.add_subplot(221)
    ax1 = sns.boxplot(data=dsc_data)
    ax1 = sns.swarmplot(data=dsc_data, color=".25")
    ax1.set_ylim([0, 1.0])
    ax1.title.set_text('$DSC$')

    ax2 = fig1.add_subplot(222)
    ax2 = sns.boxplot(data=prec_data)
    ax2 = sns.swarmplot(data=prec_data, color=".25")
    ax1.set_ylim([0, 1.0])
    ax2.title.set_text('$PREC$')

    ax3 = fig1.add_subplot(223)
    ax3 = sns.boxplot(data=rec_data)
    ax3 = sns.swarmplot(data=rec_data, color=".25")
    ax1.set_ylim([0, 1.0])
    ax3.title.set_text('$REC$')

    ax4 = fig1.add_subplot(224)
    ax4 = sns.boxplot(data=acc_data)
    ax4 = sns.swarmplot(data=acc_data, color=".25")
    ax1.set_ylim([0, 1.0])
    ax4.title.set_text('$ACC$')

    plt.show()
    if save_directory == '':
        save_directory = general_directory

        date_analysis = datetime.datetime.now()

    # ID name for the plot results
    plot_save_name = ''.join([save_directory + 'boxplots_results_',
                              date_analysis.strftime("%d_%m_%Y_%H_%M"),
                              '_.png'])

    plt.savefig(plot_save_name)
    plt.close()
    text_file_name = ''.join([save_directory + 'data_used_boxplots',
                              date_analysis.strftime("%d_%m_%Y_%H_%M"),
                              '_.txt'])

    textfile = open(text_file_name, "w")
    np.savetxt(textfile, csv_files, delimiter="\n", fmt="%s")
    textfile.close()


def get_confusion_matrix_intersection_mats(groundtruth, predicted):
    """ Returns dict of 4 boolean numpy arrays with True at TP, FP, FN, TN
    """

    confusion_matrix_arrs = {}

    groundtruth_inverse = np.logical_not(groundtruth)
    predicted_inverse = np.logical_not(predicted)

    confusion_matrix_arrs['tp'] = np.logical_and(groundtruth, predicted)
    confusion_matrix_arrs['tn'] = np.logical_and(groundtruth_inverse, predicted_inverse)
    confusion_matrix_arrs['fp'] = np.logical_and(groundtruth_inverse, predicted)
    confusion_matrix_arrs['fn'] = np.logical_and(groundtruth, predicted_inverse)

    return confusion_matrix_arrs


def get_confusion_matrix_overlaid_mask(image, groundtruth, predicted, alpha, colors):
    """
    Returns overlay the 'image' with a color mask where TP, FP, FN, TN are
    each a color given by the 'colors' dictionary
    """
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    masks = get_confusion_matrix_intersection_mats(groundtruth, predicted)
    color_mask = np.zeros_like(image, dtype=np.float32)
    for label, mask in masks.items():
        color = colors[label]
        mask_rgb = np.zeros_like(image, dtype=np.float32)
        # mask_rgb = mask_rgb.astype(int)
        size_x, size_y, channels = np.shape(mask)
        plt.figure()
        plt.title(label)
        plt.imshow(mask.astype(np.float32))

        for x_index in range(size_x):
            for y_index in range(size_y):
                if mask[
                    x_index, y_index, 0] != 0:  # and mask[x_index, y_index, 1] == 0 and mask[x_index, y_index, 2] == 0:
                    mask_rgb[x_index, y_index, :] = color
                    # print(mask_rgb[x_index, y_index, :])

        color_mask += mask_rgb
        plt.close()

        """for label, mask in masks.items():
                color = colors[label]
                mask_rgb = np.zeros_like(image)
                mask_rgb[mask != 0] = color
                color_mask += mask_rgb
    return cv2.addWeighted(image, alpha, color_mask, 1 - alpha, 0)"""

    return color_mask.astype(np.float32)  # cv2.addWeighted(image, 0.1, color_mask, 0.5, 0)


def compare_results_overlay(experiment_id = '', base_directory= '', dir_predictions='', selected_data = 'test',
                            dir_groundtruth='', dir_csv_file='', save_directory=''):
    """

    :param experiment_id:
    :param base_directory:
    :param dir_predictions:
    :param selected_data:
    :param dir_groundtruth:
    :param dir_csv_file:
    :param save_directory:
    :return:
    """
    if experiment_id != '':
        if base_directory !='':
            directory_experiment = ''.join([base_directory, 'results/', experiment_id])
            list_predicted_dataset = [folder for folder in os.listdir(directory_experiment + '/predictions/') if selected_data in folder]
            csv_file_name = [f for f in os.listdir(directory_experiment) if f.endswith('.csv') and 'evaluation_results_' in f][0]
            dir_predictions = ''.join([base_directory, 'results/', experiment_id, '/predictions/', list_predicted_dataset[0], '/'])
            path_images_folder = ''.join([base_directory, 'dataset/', list_predicted_dataset[0], '/images/'])
            path_masks_folder = ''.join([base_directory, 'dataset/', list_predicted_dataset[0], '/masks/'])
            data_results = pd.read_csv(''.join([directory_experiment, '/', csv_file_name]))

        else:
            sys.exit("base_directory needed")
    else:

        path_images_folder = dir_groundtruth + 'images/'
        path_masks_folder = dir_groundtruth + 'masks/'
        data_results = pd.read_csv(dir_csv_file)

    list_dice_values = data_results['DSC'].tolist()
    list_imgs_csv = data_results['Image'].tolist()

    if save_directory == '':
        save_directory = ''.join([directory_experiment, '/overlay_results/'])
        if not(os.path.isdir(save_directory)):
            os.mkdir(save_directory)


    image_list = [f for f in os.listdir(path_images_folder) if os.path.isfile(os.path.join(path_images_folder, f))]
    mask_list = [f for f in os.listdir(path_masks_folder) if os.path.isfile(os.path.join(path_masks_folder, f))]
    predicted_masks = [f for f in os.listdir(dir_predictions) if os.path.isfile(os.path.join(dir_predictions, f))]

    for image in predicted_masks[:]:
        if image in mask_list:

            path_image = ''.join([path_images_folder, image])
            path_mask = ''.join([path_masks_folder, image])
            path_predicted = ''.join([dir_predictions, image])
            image_frame = read_img_results(path_image)
            mask_image = read_mask(path_mask)

            for counter, element in enumerate(list_imgs_csv):

                print(element)
                if image == element:
                    dice_value = float(list_dice_values[counter])
                    predicted_mask = read_mask(path_predicted)
                    dice_value = float("{:.3f}".format(dice_value))

                    alpha = 0.5
                    confusion_matrix_colors = {
                        'tp': (50, 100, 0),  # cyan
                        'fp': (125, 0, 125),  # magenta
                        'fn': (0, 100, 50),  # blue
                        'tn': (0, 0, 0)  # black
                    }

                    overlay = get_confusion_matrix_overlaid_mask(image_frame, mask_image, predicted_mask, alpha,
                                                                 confusion_matrix_colors)

                    my_dpi = 96

                    # Use the one bellow
                    #fig = plt.figure()
                    #ax1 = fig1.add_subplot(131)

                    plt.figure(3, figsize=(640 / my_dpi, 480 / my_dpi), dpi=my_dpi)

                    plt.subplot(141)
                    title = 'DSC: ' + str(dice_value)
                    plt.title(title)
                    plt.imshow(image_frame)
                    plt.axis('off')

                    plt.subplot(142)
                    plt.title('Mask')
                    plt.imshow(mask_image)
                    plt.axis('off')

                    plt.subplot(143)
                    plt.title('Predicted')
                    plt.imshow(predicted_mask)
                    plt.axis('off')

                    plt.subplot(144)
                    plt.title('Overlay')
                    plt.imshow(overlay)
                    plt.axis('off')
                    plt.savefig(''.join([save_directory, image]))
                    plt.close()


def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def calculate_rates(image_1, image_2):

    """
    Takes two black and white images and calculates recall, precision,
    average precision  and accuracy
    :param image_1: array
    :param image_2: array
    :return: list with the values of precision, recall, average precision and accuracy
    """

    image_1 = np.asarray(image_1).astype(np.bool)
    image_2 = np.asarray(image_2).astype(np.bool)
    image_1 = image_1.flatten()
    image_2 = image_2.flatten()

    if image_1.shape != image_2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    accuracy_value = accuracy_score(image_1, image_2)

    if (np.unique(image_1) == [False]).all() and (np.unique(image_1) == [False]).all():
        recall_value = 1.
        precision_value = 1.
        average_precision = 1.

    else:
        recall_value = recall_score(image_1, image_2)
        precision_value = precision_score(image_1, image_2)
        average_precision = average_precision_score(image_1, image_2)

    return precision_value, recall_value, average_precision, accuracy_value


def calculate_performance(dir_results, dir_groundtruth):

    """
    Calculate the performance metrics given two directories with results dataset and ground truth dataset
    The performance metrics calculated are: Dice Coefficient, Precision, Recall, Average Precision and
    Accuracy.
    :param dir_results: Directory of the results images
    :param dir_groundtruth: Directory of the ground truth images
    :return: Pandas Dataframe with the image name and the metrics
    """

    imgs_name = []
    dice_values = []
    precision_values = []
    recall_values = []
    avg_precision_values = []
    accuracy_values = []

    img_list_results = sorted([file for file in os.listdir(dir_results) if file.endswith('.png')])
    img_groundtruth = sorted([file for file in os.listdir(dir_groundtruth) if file.endswith('.png')])

    for i, image in enumerate(tqdm.tqdm(img_list_results, desc=f'Analyzing {len(img_list_results)} images')):
        if image in img_groundtruth:
            img1 = read_img_results(''.join([dir_results, '/', image]))
            img2 = read_img_results(dir_groundtruth + image)
            imgs_name.append(image)
            dice_values.append(dice(img1, img2))
            precision, recall, average_precision, accuracy = calculate_rates(img1, img2)
            precision_values.append(precision)
            recall_values.append(recall)
            avg_precision_values.append(average_precision)
            accuracy_values.append(accuracy)

    data_results = pd.DataFrame(np.array([imgs_name, dice_values, precision_values, recall_values,
                                  avg_precision_values, accuracy_values]).T,
                        columns=['Image', 'DSC', 'Precision', 'Recall', 'Avg. Precision', 'Accuracy'])

    return data_results


def analyze_performances(project_dir, exclude=[]):

    """
    Analyze the performance of a directory or a list of directories
    :param project_dir:
    :param exclude:
    :return:
    """
    # 2DO: recognize directory or list

    if type(project_dir) == str:

        results_list = sorted(os.listdir(project_dir + 'results/'))
        results_list.remove('temp')
        results_list.remove('analysis')

        if exclude:
            for elem in exclude:
                exclude.remove(elem)

        for experiment_id in results_list:
            print(experiment_id)
            folders_prediction = os.listdir(os.path.join(project_dir, 'results', experiment_id, 'predictions'))

            for folder in folders_prediction:
                dir_results = os.path.join(project_dir, 'results', experiment_id, 'predictions', folder)
                dir_gt = os.path.join(project_dir, 'dataset', folder, 'masks/')

                if os.path.isdir(dir_gt):
                    results_data = calculate_performance(dir_results, dir_gt)
                    name_results_file = ''.join([project_dir, 'results/', experiment_id, '/',
                                                 'evaluation_results_', folder, '_',experiment_id, '_.csv'])

                    results_data.to_csv(name_results_file)
                    print(f"results saved at: {name_results_file}")
                else:
                    print(f' folder: {dir_gt} not found')


def save_boxplots(project_dir):
    """
    Given a folder with results it saves the boxplots of the datasets were inferences were made
    you need to have inside your directory a folder "predictions" and the csv files with the precitions
    for each of folder(s)
    :param project_dir: (str) directory to analyze
    :return:
    """

    if type(project_dir) == str:
        compare_box_plots(project_dir)
        #olders_prediction = os.listdir(os.path.join(project_dir, 'results', experiment_id, 'predictions'))

    elif type(project_dir) == list:
        results_list = os.listdir(project_dir + 'results/')
        results_list.remove('temp')
        for experiment_id in results_list:

            folders_prediction = os.listdir(os.path.join(project_dir, 'results', experiment_id, 'predictions'))

            for folder in folders_prediction:
                dir_results = os.path.join(project_dir, 'results', experiment_id, 'predictions', folder)
                dir_gt = os.path.join(project_dir, 'dataset', folder, 'masks/')

                if os.path.isdir(dir_gt):
                    results_data = calculate_performance(dir_results, dir_gt)
                    name_results_file = ''.join([project_dir, 'results/', experiment_id, '/',
                                                 'evaluation_results_', folder, '_',experiment_id, '_.csv'])

                    results_data.to_csv(name_results_file)
                    print(f"results saved at: {name_results_file}")
                else:
                    print(f' folder: {dir_gt} not found')
    else:
        print('type(project dir) not understood')


def extract_information_from_name(string_name):
    """
    :param string_name:
    :return:
    """
    model_name = re.search('evaluation_results_test_0_(.+?)_lr_', string_name).group(1)
    date_experiment = re.search('_rgb_(.+?)_.csv', string_name).group(1)
    lr = re.search('lr_(.+?)_', string_name).group(1)
    bs = re.search('bs_(.+?)_', string_name).group(1)

    return lr, bs, model_name, date_experiment


def plot_training_history(list_csv_files, save_dir=''):
    """
    Plots the training history of a model given the list of csv files (in case that there are different training stages)


    Parameters
    ----------
    list_csv_files (list): list of the csv files with the training history
    save_dir (str): The directory where to save the file, if empty, the current working directory

    Returns
    -------

    """
    if len(list_csv_files) > 1:
        print(list_csv_files[0])
        fine_tune_history = pd.read_csv(list_csv_files[0])
        fine_tune_lim = fine_tune_history['epoch'].tolist()[-1]
        header_1 = fine_tune_history.columns.values.tolist()
        train_history = pd.read_csv(list_csv_files[-1])
        header_2 = train_history.columns.values.tolist()
        # mix the headers in case they are different among files
        dictionary = {header_2[i]:name for i, name in enumerate(header_1)}
        train_history.rename(columns=dictionary, inplace=True)
        # append the dataframes in a single one
        train_history = fine_tune_history.append(train_history, ignore_index=True)

    else:
        fine_tune_lim = 0
        train_history = pd.read_csv(list_csv_files[0])

    fig = plt.figure(1, figsize=(12, 9))

    ax1 = fig.add_subplot(221)
    ax1.title.set_text('$ACC$')
    ax1.plot(train_history['accuracy'].tolist(), label='train')
    ax1.plot(train_history['val_accuracy'].tolist(), label='val')
    ax1.fill_between((0, fine_tune_lim), 0, 1, facecolor='orange', alpha=0.4)

    plt.legend(loc='best')

    ax2 = fig.add_subplot(222)
    ax2.title.set_text('PREC')
    ax2.plot(train_history['precision'].tolist(), label='train')
    ax2.plot(train_history['val_precision'].tolist(), label='val')
    ax2.fill_between((0, fine_tune_lim), 0, 1, facecolor='orange', alpha=0.4)
    plt.legend(loc='best')

    ax3 = fig.add_subplot(223)
    ax3.title.set_text('$LOSS$')
    ax3.plot(train_history['loss'].tolist(), label='train')
    ax3.plot(train_history['val_loss'].tolist(), label='val')
    max_xval = np.amax([train_history['loss'].tolist(), train_history['val_loss'].tolist()])
    ax3.fill_between((0, fine_tune_lim), 0, max_xval, facecolor='orange', alpha=0.4)
    plt.legend(loc='best')

    ax4 = fig.add_subplot(224)
    ax4.title.set_text('$REC$')
    ax4.plot(train_history['recall'].tolist(), label='train')
    ax4.plot(train_history['val_recall'].tolist(), label='val')
    ax4.fill_between((0, fine_tune_lim), 0, 1, facecolor='orange', alpha=0.4)
    plt.legend(loc='best')

    if save_dir == '':
        dir_save_figure = os.getcwd() + '/training_history.png'
    else:
        dir_save_figure = save_dir + 'training_history.png'

    print(f'figure saved at: {dir_save_figure}')
    plt.savefig(dir_save_figure)
    plt.close()


def analyze_dataset_distribution(dataset_dir, plot_figure=False, dir_save_fig=''):
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
    list_cases = sorted([f for f in os.listdir(dataset_dir) if os.path.isdir(dataset_dir + f)])
    cases_ocurence = {'CIS WLI': 0, 'CIS NBI': 0, 'HGC WLI': 0, 'HGC NBI': 0, 'HLT WLI': 0, 'HLT NBI': 0,
                      'LGC WLI': 0, 'LGC NBI': 0, 'NTL WLI': 0, 'NTL NBI': 0}
    class_cases_dict = {case: copy.copy(cases_ocurence) for case in list_cases}
    unique_combinations = list()
    total_imgs = list()
    for case in list_cases[:]:

        combinations = list()
        csv_file = [f for f in os.listdir(dataset_dir + case) if f.endswith('.csv')].pop()
        csv_file_dir = os.path.join(dataset_dir, case, csv_file)
        df = pd.read_csv(csv_file_dir)
        list_tissue_types = df['tissue type'].tolist()
        list_imaging_type = df['imaging type'].tolist()
        # obtain all the unique combinations in the different cases

        for i, tissue in enumerate(list_tissue_types):
            combination = ''.join([tissue, ' ', list_imaging_type[i]])
            combinations.append(combination)
            if combination not in unique_combinations:
                unique_combinations.append(combination)

        total_imgs.append(len(combinations))
        for combination in np.unique(combinations):
            class_cases_dict[case][combination] = combinations.count(combination)

    # create an empty array
    plot_array = np.zeros([len(list_cases), len(unique_combinations)])
    normalized_array = copy.copy(plot_array)
    # Now lets fill the array that corresponds to the ocurrence of each class for each patient case
    for i, case in enumerate(list_cases[:]):
        for j, key in enumerate(class_cases_dict[case].keys()):
            plot_array[i][j] = class_cases_dict[case][key]
            normalized_array[i][j] = class_cases_dict[case][key]/total_imgs[i]
            xticklabels = list(class_cases_dict[case].keys())

    print(cases_ocurence)

    plt.figure()
    labels = np.asarray(plot_array).reshape(len(list_cases), len(unique_combinations))
    sns.heatmap(normalized_array, cmap='YlOrBr', cbar=False, linewidths=.5,
                yticklabels=list_cases, xticklabels=xticklabels, annot=labels)

    plt.xlabel('Classes')
    plt.ylabel('Cases')
    plt.show()


def compute_confusion_matrix(gt_data, predicted_data, plot_figure=False, dir_save_fig=''):
    """
    Compute the confusion Matrix given the ground-truth data (gt_data) and predicted data (predicted_data)
    in list format. If Plot is True shows the matrix .
    Parameters
    ----------
    gt_data : list
    predicted_data : list
    plot_figure :
    dir_save_fig :

    Returns
    -------

    """
    uniques_predicts = np.unique(predicted_data)
    uniques_gt = np.unique(gt_data)
    if collections.Counter(uniques_gt) == collections.Counter(uniques_predicts):
        uniques = uniques_gt
    else:
        uniques = np.unique([*uniques_gt, *uniques_predicts])

    ocurrences = [gt_data.count(unique) for unique in uniques]
    conf_matrix = confusion_matrix(gt_data, predicted_data)
    group_percentages = [conf_matrix[i]/ocurrences[i] for i, row in enumerate(conf_matrix)]

    size = len(list(uniques))
    list_uniques = list(uniques)
    xlabel_names = list()
    for name in list_uniques:
        # if the name of the unique names is longer than 4 characters will split it
        if len(name) > 4:
            name_split = name.split('-')
            new_name = ''
            for splits in name_split:
                new_name = new_name.join([splits[0]])

            xlabel_names.append(new_name)
        else:
            xlabel_names.append(name)

    labels = np.asarray(group_percentages).reshape(size, size)
    sns.heatmap(group_percentages, cmap='Blues', cbar=False, linewidths=.5,
                yticklabels=list(uniques), xticklabels=list(xlabel_names), annot=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Real Class')

    if plot_figure is True:
        plt.show()

    if dir_save_fig == '':
            dir_save_figure = os.getcwd() + '/confusion_matrix.png'

    else:
        if not dir_save_fig.endswith('.png'):
          dir_save_figure = dir_save_fig + 'confusion_matrix.png'
        else:
            dir_save_figure = dir_save_fig

    print(f'figure saved at: {dir_save_figure}')

    plt.savefig(dir_save_figure)
    plt.close()

    return conf_matrix


def analyze_multiclass_experiment(gt_data_file, predictions_data_dir, plot_figure=False, dir_save_figs=None,
                                  analyze_training_history=False):
    """
    Analyze the results of a multi-class classification experiment

    Parameters
    ----------
    gt_data_file :
    predictions_data_dir :
    plot_figure :
    dir_save_fig :

    Returns
    -------
    History plot, Confusion Matrix

    """
    wli_imgs = []
    nbi_imgs = []
    predictions_nbi = []
    predictions_wli = []
    wli_tissue_types = []
    nbi_tissue_types = []

    list_prediction_files = [f for f in os.listdir(predictions_data_dir) if 'predictions' in f and '(_pre' not in f]
    file_predictiosn = list_prediction_files.pop()
    path_file_predictions = predictions_data_dir + file_predictiosn
    print(f'file predictions found: {file_predictiosn}')

    df_ground_truth = pd.read_csv(gt_data_file)
    df_preditc_data = pd.read_csv(path_file_predictions)

    predictions_names = df_preditc_data['fname'].tolist()
    predictions_vals = df_preditc_data['over all'].tolist()

    gt_names = df_ground_truth['image_name'].tolist()
    gt_vals = df_ground_truth['tissue type'].tolist()
    imaging_type = df_ground_truth['imaging type'].tolist()

    existing_gt_vals = list()
    ordered_predictiosn = list()
    for name in predictions_names:
        if name in gt_names:
            index = predictions_names.index(name)
            ordered_predictiosn.append(predictions_vals[index])
            index_gt = gt_names.index(name)
            existing_gt_vals.append(gt_vals[index_gt])

            if imaging_type[index_gt] == 'NBI':
                nbi_imgs.append(name)
                predictions_nbi.append(predictions_vals[index])
                nbi_tissue_types.append(gt_vals[index_gt])
            if imaging_type[index_gt] == 'WLI':
                wli_imgs.append(name)
                predictions_wli.append(predictions_vals[index])
                wli_tissue_types.append(gt_vals[index_gt])

    # dri to save the figures
    if dir_save_figs:
        dir_save_fig = dir_save_figs
    else:
        dir_save_fig = predictions_data_dir

    data_yaml = {'Accuracy ALL ': float(accuracy_score(existing_gt_vals, ordered_predictiosn)),
                 'Accuracy WLI ': float(accuracy_score(wli_tissue_types, predictions_wli)),
                 'Accuracy NBI ': float(accuracy_score(nbi_tissue_types, predictions_nbi))
                 }

    # ACCURACY
    print('Accuracy ALL: ', accuracy_score(existing_gt_vals, ordered_predictiosn))
    print('Accuracy WLI: ', accuracy_score(wli_tissue_types, predictions_wli))
    print('Accuracy NBI: ', accuracy_score(nbi_tissue_types, predictions_nbi))
    # Precision
    print('Precision ALL: ', precision_score(existing_gt_vals, ordered_predictiosn, average=None))
    print('Precision WLI: ', precision_score(wli_tissue_types, predictions_wli, average=None))
    print('Precision NBI: ', precision_score(nbi_tissue_types, predictions_nbi, average=None, zero_division=1))
    # Recall
    print('Recall ALL: ', recall_score(existing_gt_vals, ordered_predictiosn, average=None))
    print('Recall WLI: ', recall_score(wli_tissue_types, predictions_wli, average=None))
    print('Recall NBI: ', recall_score(nbi_tissue_types, predictions_nbi, average=None, zero_division=1))

    # Confusion Matrices
    compute_confusion_matrix(existing_gt_vals, ordered_predictiosn, plot_figure=False,
                             dir_save_fig=dir_save_fig + 'confusion_matrix_all.png')

    compute_confusion_matrix(wli_tissue_types, predictions_wli,
                             dir_save_fig=dir_save_fig + 'confusion_matrix_wli.png')

    compute_confusion_matrix(nbi_tissue_types, predictions_nbi,
                             dir_save_fig=dir_save_fig + 'confusion_matrix_nbi.png')

    dir_data_yaml = dir_save_fig + 'performance_analysis.yaml'
    save_yaml(dir_data_yaml, data_yaml)

    gt_values = []
    for name in predictions_names:
        if name in gt_names:
            index = gt_names.index(name)
            gt_values.append(gt_vals[index])

    new_df = df_preditc_data.copy()
    data_top = list(new_df.columns)
    new_df.insert(len(data_top), "real values", gt_values, allow_duplicates=True)
    name_data_save = path_file_predictions
    new_df.to_csv(name_data_save, index=False)
    print(f'results saved at {name_data_save}')

    # analyze the history
    if analyze_training_history is True:
        list_history_files = [f for f in os.listdir(predictions_data_dir) if 'train_history' in f]
        ordered_history = list()
        fine_tune_file = [f for f in list_history_files if 'fine_tune' in f]
        if fine_tune_file:
            fine_tune_file_dir = predictions_data_dir + fine_tune_file.pop()
            ordered_history.append(fine_tune_file_dir)

        ordered_history.append(predictions_data_dir + list_history_files[-1])
        plot_training_history(ordered_history, save_dir=dir_save_fig)


def compare_experiments(dir_folder_experiments, selection_criteria=['evaluation_results_test_0'], dir_save_results='',
                        exclude=[], top_results=1.0):
    """
    Compre the DSC, Prec, Rec and ACC of different experiments and save the boxplots comparison
    :param dir_folder_experiments:
    :param selection_criteria:
    :param dir_save_results:
    :param top_results:
    :return:
    """

    date_analysis = datetime.datetime.now()
    names_analyzed_files = []
    dsc_values = {}
    prec_values = {}
    rec_values = {}
    acc_values = {}
    median_dsc = []

    list_experiments = [dir_folder for dir_folder in sorted(glob.glob(dir_folder_experiments + '*' )) if 'analysis' not in dir_folder or 'temp' not in dir_folder]

    if exclude:
        for experiment_folder in list_experiments:
            for exclusion_case in exclude:
                if exclusion_case in experiment_folder:
                    list_experiments.remove(experiment_folder)

    for j, dir_experiment in enumerate(list_experiments):
        for selection in selection_criteria:
            list_results_files = [f for f in os.listdir(dir_experiment) if selection in f]
            for results in list_results_files:
                names_analyzed_files.append(results)
                data_file = pd.read_csv(os.path.join(dir_experiment, results))
                dsc_values[results] = data_file['DSC'].tolist()
                median_dsc.append(np.median(data_file['DSC'].tolist()))
                prec_values[results] = data_file['Precision'].tolist()
                rec_values[results] = data_file['Recall'].tolist()
                acc_values[results] = data_file['Accuracy'].tolist()

    zipped_results = [(x, y)  for x, y in sorted(zip(median_dsc, names_analyzed_files), reverse=True)]
    # save x.x% top results in a list
    top_list = zipped_results[:int(top_results*len(zipped_results))]

    dsc_values = {pair[1]: dsc_values[pair[1]] for pair in top_list if pair[1] in dsc_values}
    prec_values = {pair[1]: prec_values[pair[1]] for pair in top_list if pair[1] in prec_values}
    rec_values = {pair[1]: rec_values[pair[1]] for pair in top_list if pair[1] in rec_values}
    acc_values = {pair[1]: acc_values[pair[1]] for pair in top_list if pair[1] in acc_values}

    print_names = [f'{extract_information_from_name(file_name)[2]} bs:{extract_information_from_name(file_name)[0]} ' \
                   f'lr:{extract_information_from_name(file_name)[1]} DSC: {score:.2f}' for score, file_name in top_list]

    dsc_data = pd.DataFrame.from_dict(dsc_values, orient='index').T
    prec_data = pd.DataFrame.from_dict(prec_values, orient='index').T
    rec_data = pd.DataFrame.from_dict(rec_values, orient='index').T
    acc_data = pd.DataFrame.from_dict(acc_values, orient='index').T

    alphabet_string = string.ascii_uppercase
    alphabet_list = list(alphabet_string)

    rename_headers = {element:alphabet_list[i] for i, element in enumerate(dsc_data)}
    dsc_data = dsc_data.rename(rename_headers, axis=1)
    prec_data = prec_data.rename(rename_headers, axis=1)
    rec_data = rec_data.rename(rename_headers, axis=1)

    # re-arrange the data for analysis
    acc_data = acc_data.rename(rename_headers, axis=1)
    dict_temp = pd.DataFrame.to_dict(acc_data)
    new_dict = {}
    index = 0
    for element in dict_temp:
        for i, sub_elem in enumerate(dict_temp[element]):
            new_dict[index] = {'acc': dict_temp[element][i], 'experiment': element}
            index += 1
    new_acc_vals = pd.DataFrame.from_dict(new_dict, orient='index')

    # build the image to plot
    fig1 = plt.figure(1, figsize=(16,9))
    ax1 = fig1.add_subplot(221)
    sns.boxplot(data=dsc_data)
    #ax1 = sns.swarmplot(data=dsc_data, color=".25")
    ax1.set_ylim([0, 1.0])
    ax1.title.set_text('$DSC$')

    ax2 = fig1.add_subplot(222)
    ax2 = sns.boxplot(data=prec_data)
    #ax2 = sns.swarmplot(data=prec_data, color=".25")
    ax2.set_ylim([0, 1.0])
    ax2.title.set_text('$PREC$')

    ax3 = fig1.add_subplot(223)
    ax3 = sns.boxplot(data=rec_data)
    #ax3 = sns.swarmplot(data=rec_data, color=".25")
    ax3.set_ylim([0, 1.0])
    ax3.title.set_text('$REC$')

    ax4 = fig1.add_subplot(224)
    ax4 = sns.boxplot(x='experiment', y='acc', hue='experiment', data=new_acc_vals, fliersize=0)
    handles, labels = ax4.get_legend_handles_labels()
    ax4.set_ylim([0, 1.0])
    ax4.title.set_text('$ACC$')
    ax4.legend(handles, print_names, loc="right")

    # create a folder analysis inside the directory in case it doesnt exits to save the figures
    if dir_save_results == '':
        analysis_folder = dir_folder_experiments + 'analysis/'
        if not(os.path.isdir(analysis_folder)):
            os.mkdir(analysis_folder)

        dir_save_results = ''.join([analysis_folder, 'analysis_', date_analysis.strftime("%d_%m_%Y_%H_%M"), '/'])
        if not(os.path.isdir(dir_save_results)):
            os.mkdir(dir_save_results)

    # Save the figure of the box plots considered in the analysis
    plot_save_name = ''.join([dir_save_results, 'Boxplot_comparisons_', date_analysis.strftime("%d_%m_%Y_%H_%M"), '_.png'])
    plt.savefig(plot_save_name)
    #plt.show()
    plt.close()

    # Save a CSV file stating the data considered in the analysis
    if len(zipped_results) < 26:
        ID_results = list(alphabet_string)[:len(zipped_results)]

    else:
        ID_results = list(alphabet_string)
        while len(ID_results) < len(zipped_results):
            ID_results.append('')

    print(len(ID_results))
    list_batch_sizes = []
    list_learning_rates = []
    list_models = []
    list_dates_experiment = []
    dsc_list = []
    filename_list = []
    list_experiments_save = []
    for result in zipped_results:
        dsc, filename = result
        dsc_list.append(dsc)
        filename_list.append(filename)
        lookfor_name = filename.replace('_.csv', '')
        lookfor_name = lookfor_name.replace('evaluation_results_test_0_', '')
        for experiment in list_experiments:
            if lookfor_name in experiment:
                list_experiments_save.append(experiment.replace(dir_folder_experiments, ''))
        lr, bs, model_name, date_experiment = extract_information_from_name(filename)
        list_batch_sizes.append(bs)
        list_learning_rates.append(lr)
        list_models.append(model_name)
        list_dates_experiment.append(date_experiment)

    export_data_used = pd.DataFrame(np.array([ID_results, list_experiments_save, dsc_list, list_batch_sizes,
                                             list_learning_rates, list_models, list_dates_experiment, filename_list]).T,
                                    columns=['ID', 'Experiment', 'DSC', 'batch size', 'learning rate',
                                             'Model name', 'date experiment', 'file experiment'])

    name_data_save = ''.join([dir_save_results, 'Data_used_', date_analysis.strftime("%d_%m_%Y_%H_%M"), '_.csv'])
    export_data_used.to_csv(name_data_save)
    print(f'results saved at {dir_save_results}')


def analyze_dataset_multihead_pie_chart(dir_csv_file, headers=['imaging type', 'tissue type']):
    df = pd.read_csv(dir_csv_file)

    img_names = df['image_name'].tolist()
    tissue_types = df['tissue type'].tolist()
    imaging_types = df['imaging type'].tolist()

    wli_tissues = []
    nbi_tissues = []
    all_tissues = []

    for i, names in enumerate(img_names):
        if imaging_types[i] == 'WLI':
            wli_tissues.append(tissue_types[i])
            all_tissues.append('WLI ' + tissue_types[i])
        if imaging_types[i] == 'NBI':
            nbi_tissues.append(tissue_types[i])
            all_tissues.append('NBI ' + tissue_types[i])

    ocurrences_nbi = [nbi_tissues.count(unique) for unique in np.unique(nbi_tissues)]
    ocurrences_wli = [wli_tissues.count(unique) for unique in np.unique(wli_tissues)]
    ocurrences_all = [all_tissues.count(unique) for unique in np.unique(all_tissues)]

    print(np.unique(all_tissues))
    print('ALL:', ocurrences_all)
    dictionary_all = {}

    for i, unique_type in enumerate(np.unique(all_tissues)):
        dictionary_all[unique_type] = ocurrences_all[i]

    print('ALL', dictionary_all)

    fig1 = plt.figure(1, figsize=(12, 7))
    ax1 = fig1.add_subplot(131)
    ax1.pie(ocurrences_all, labels=np.unique(all_tissues), autopct='%.0f%%')
    ax1.title.set_text('ALL')

    ax2 = fig1.add_subplot(132)
    ax2.pie(ocurrences_wli, labels=np.unique(wli_tissues), autopct='%.0f%%')
    ax2.title.set_text('WLI')

    ax3 = fig1.add_subplot(133)
    ax3.pie(ocurrences_nbi, labels=np.unique(nbi_tissues), autopct='%.0f%%')
    ax3.title.set_text('NBI')

    plt.show()


def analyze_dataset_pie_chart(dir_csv_file, header):
    df = pd.read_csv(dir_csv_file)
    data_to_analyze = df[header].tolist()
    unique_values = np.unique(data_to_analyze)
    ocurrences = [data_to_analyze.count(unique) for unique in unique_values]
    print(unique_values, ocurrences)
    labels = unique_values
    plot_data = ocurrences
    plt.figure()
    plt.pie(plot_data, labels=labels, autopct='%.0f%%')
    plt.show()


def analyze_results_nbi_wli(gt_file, predictions_file, plot_figure=False, dir_save_fig='',
                    analyze_multiclass=False):

    df_ground_truth = pd.read_csv(gt_file)
    df_predicted_data = pd.read_csv(predictions_file)

    predictions_names = df_predicted_data['fname'].tolist()
    predictions_vals = df_predicted_data['over all'].tolist()

    gt_names = df_ground_truth['image_name'].tolist()
    gt_vals = df_ground_truth['tissue type'].tolist()
    imaging_type = df_ground_truth['imaging type'].tolist()

    wli_imgs = []
    nbi_imgs = []

    predictions_nbi = []
    predictions_wli = []

    wli_tissue_types = []
    nbi_tissue_types = []

    for name in predictions_names:
        if name in gt_names:
            index = predictions_names.index(name)
            if imaging_type[index] == 'NBI':
                nbi_imgs.append(name)
                predictions_nbi.append(predictions_vals[index])
                nbi_tissue_types.append(gt_vals[index])
            if imaging_type[index] == 'WLI':
                wli_imgs.append(name)
                predictions_wli.append(predictions_vals[index])
                wli_tissue_types.append(gt_vals[index])

    # predictions wli
    print('Accuracy: ', accuracy_score(wli_tissue_types, predictions_wli))
    print('Precision: ', precision_score(wli_tissue_types, predictions_wli, average=None))
    print('Recall: ', recall_score(wli_tissue_types, predictions_wli, average=None))
    dir_save_figs = os.path.split(predictions_file)[0]

    # predictions nbi
    compute_confusion_matrix(wli_tissue_types, predictions_wli, dir_save_fig=dir_save_figs + '/confusion_matrix_wli.png')
    print('Accuracy: ', accuracy_score(nbi_tissue_types, predictions_nbi))
    print('Precision: ', precision_score(nbi_tissue_types, predictions_nbi, average=None, zero_division=1))
    print('Recall: ', recall_score(nbi_tissue_types, predictions_nbi, average=None, zero_division=1))
    compute_confusion_matrix(nbi_tissue_types, predictions_nbi, dir_save_fig=dir_save_figs + '/confusion_matrix_nbi.png')


def naive_ensembles(file_1, file_2):

    ordered_classes = ['CIS', 'HGC', 'HLT', 'LGC', 'NTL']

    df1 = pd.read_csv(file_1)
    df2 = pd.read_csv(file_2)

    list_names_1 = df1['fname'].tolist()
    list_names_2 = df2['fname'].tolist()

    list_class_11 = df1['class_1'].tolist()
    list_class_21 = df1['class_2'].tolist()
    list_class_31 = df1['class_3'].tolist()
    list_class_41 = df1['class_4'].tolist()
    list_class_51 = df1['class_5'].tolist()

    list_class_12 = df2['class_1'].tolist()
    list_class_22 = df2['class_2'].tolist()
    list_class_32 = df2['class_3'].tolist()
    list_class_42 = df2['class_4'].tolist()
    list_class_52 = df2['class_5'].tolist()

    group_class_1 = list()
    group_class_2 = list()
    group_class_3 = list()
    group_class_4 = list()
    group_class_5 = list()
    over_all_names = list()
    over_all_class = list()

    for j, name in enumerate(list_names_1):
        if name in list_names_2:
            over_all_names.append(name)
            index_1 = list_names_1.index(name)
            index_2 = list_names_2.index(name)
            group_class_1.append(np.mean([list_class_11[index_1], list_class_12[index_2]]))
            group_class_2.append(np.mean([list_class_21[index_1], list_class_22[index_2]]))
            group_class_3.append(np.mean([list_class_31[index_1], list_class_32[index_2]]))
            group_class_4.append(np.mean([list_class_41[index_1], list_class_42[index_2]]))
            group_class_5.append(np.mean([list_class_51[index_1], list_class_52[index_2]]))

    for j, name in enumerate(over_all_names):
        scores = [group_class_1[j], group_class_2[j], group_class_3[j], group_class_4[j], group_class_5[j]]
        over_all_class.append(ordered_classes[scores.index(np.amax(scores))])

    d = {'fname': over_all_names, 'class_1': group_class_1, 'class_2': group_class_2, 'class_3': group_class_3,
        'class_4': group_class_4, 'class_5': group_class_5, 'over all': over_all_class}

    pd.DataFrame(data=d)
    new_df = pd.DataFrame(d)
    print(new_df)
    file_name = 'predictions_ensemble.csv'
    new_df.to_csv(file_name, index=False)


def analyze_simple_gan_results(csv_file_path):

    df_preditc_data = pd.read_csv(csv_file_path)
    headers = list(df_preditc_data.columns)

    predictions_names = df_preditc_data['fname'].tolist()
    real_values = df_preditc_data['real class'].tolist()

    predictions_original = df_preditc_data['prediction original'].tolist()
    predictions_converted = df_preditc_data['prediction converted'].tolist()
    predictions_reconverted = df_preditc_data['prediction reconverted'].tolist()
    if 'average prediction' in headers:
        predictions_average = df_preditc_data['average prediction'].tolist()
    if 'weighted prediction' in headers:
        predictions_weighted = df_preditc_data['weighted prediction'].tolist()

    if 'average prediction 2' in headers:
        predictions_average = df_preditc_data['average prediction 2'].tolist()
    if 'weighted prediction 2' in headers:
        predictions_weighted = df_preditc_data['weighted prediction 2'].tolist()

    imaging_type = df_preditc_data['original imaging'].tolist()
    output_images_dir = os.path.split(csv_file_path)[0]

    predictions_original_nbi = []
    predictions_original_wli = []

    predictions_converted_nbi = []
    predictions_converted_wli = []

    predictions_reconverted_nbi = []
    predictions_reconverted_wli = []

    predictions_average_nbi = []
    predictions_average_wli = []

    predictions_weighted_nbi = []
    predictions_weighted_wli = []

    wli_tissue_types = []
    nbi_tissue_types = []

    for name in predictions_names:

        index = predictions_names.index(name)
        if imaging_type[index] == 'NBI':
            nbi_tissue_types.append(real_values[index])
            predictions_original_nbi.append(predictions_original[index])
            predictions_converted_nbi.append(predictions_converted[index])
            predictions_reconverted_nbi.append(predictions_reconverted[index])
            predictions_average_nbi.append(predictions_average[index])
            predictions_weighted_nbi.append(predictions_weighted[index])

        if imaging_type[index] == 'WLI':
            wli_tissue_types.append(real_values[index])
            predictions_original_wli.append(predictions_original[index])
            predictions_converted_wli.append(predictions_converted[index])
            predictions_reconverted_wli.append(predictions_reconverted[index])
            predictions_average_wli.append(predictions_average[index])
            predictions_weighted_wli.append(predictions_weighted[index])

    data_yaml = {'Accuracy ALL ORIGINAL imgs': float(accuracy_score(real_values, predictions_original)),
    'Accuracy WLI ORIGINAL imgs': float(accuracy_score(wli_tissue_types, predictions_original_wli)),
    'Accuracy NBI ORIGINAL imgs': float(accuracy_score(nbi_tissue_types, predictions_original_nbi)),
    'Accuracy ALL CONVERTED imgs': float(accuracy_score(real_values, predictions_converted)),
    'Accuracy WLI CONVERTED imgs': float(accuracy_score(wli_tissue_types, predictions_converted_wli)),
    'Accuracy NBI CONVERTED imgs': float(accuracy_score(nbi_tissue_types, predictions_converted_nbi)),
    'Accuracy ALL RE-CONVERTED imgs': float(accuracy_score(real_values, predictions_reconverted)),
    'Accuracy WLI RE-CONVERTED imgs': float(accuracy_score(wli_tissue_types, predictions_reconverted_wli)),
    'Accuracy NBI RE-CONVERTED imgs': float(accuracy_score(nbi_tissue_types, predictions_reconverted_nbi)),
    'Accuracy ALL AVERAGE imgs': float(accuracy_score(real_values, predictions_average)),
    'Accuracy WLI AVERAGE imgs': float(accuracy_score(wli_tissue_types, predictions_average_wli)),
    'Accuracy NBI AVERAGE imgs': float(accuracy_score(nbi_tissue_types, predictions_average_nbi)),
    'Accuracy ALL WEIGHTED imgs': float(accuracy_score(real_values, predictions_weighted)),
    'Accuracy WLI WEIGHTED imgs': float(accuracy_score(wli_tissue_types, predictions_weighted_wli)),
    'Accuracy NBI WEIGHTED imgs': float(accuracy_score(nbi_tissue_types, predictions_weighted_nbi)),
                 }
    # Accuracy
    print('Accuracy ALL ORIGINAL Images: ', accuracy_score(real_values, predictions_original))
    print('Accuracy WLI ORIGINAL imgs: ', accuracy_score(wli_tissue_types, predictions_original_wli))
    print('Accuracy NBI ORIGINAL imgs: ', accuracy_score(nbi_tissue_types, predictions_original_nbi))
    print('Accuracy ALL CONVERTED Images: ', accuracy_score(real_values, predictions_converted))
    print('Accuracy WLI CONVERTED imgs: ', accuracy_score(wli_tissue_types, predictions_converted_wli))
    print('Accuracy NBI CONVERTED imgs: ', accuracy_score(nbi_tissue_types, predictions_converted_nbi))
    print('Accuracy ALL RE-CONVERTED Images: ', accuracy_score(real_values, predictions_reconverted))
    print('Accuracy WLI RE-CONVERTED imgs: ', accuracy_score(wli_tissue_types, predictions_reconverted_wli))
    print('Accuracy NBI RE-CONVERTED imgs: ', accuracy_score(nbi_tissue_types, predictions_reconverted_nbi))
    print('Accuracy ALL AVERAGE Images: ', accuracy_score(real_values, predictions_average))
    print('Accuracy WLI AVERAGE imgs: ', accuracy_score(wli_tissue_types, predictions_average_wli))
    print('Accuracy NBI AVERAGE imgs: ', accuracy_score(nbi_tissue_types, predictions_average_nbi))
    print('Accuracy ALL WEIGHTED Images: ', accuracy_score(real_values, predictions_weighted))
    print('Accuracy WLI WEIGHTED imgs: ', accuracy_score(wli_tissue_types, predictions_weighted_wli))
    print('Accuracy NBI WEIGHTED imgs: ', accuracy_score(nbi_tissue_types, predictions_weighted_nbi))

    # Precision
    print('Precision all ORIGINAL Images: ', precision_score(real_values, predictions_original, average=None))
    print('Precision WLI ORIGINAL imgs: ', precision_score(wli_tissue_types, predictions_original_wli, average=None))
    print('Precision NBI ORIGINAL imgs: ',
          precision_score(nbi_tissue_types, predictions_original_nbi, average=None, zero_division=1))
    print('Precision all CONVERTED Images: ', precision_score(real_values, predictions_converted, average=None))
    print('Precision WLI CONVERTED imgs: ', precision_score(wli_tissue_types, predictions_converted_wli, average=None))
    print('Precision NBI CONVERTED imgs: ',
          precision_score(nbi_tissue_types, predictions_converted_nbi, average=None, zero_division=1))
    print('Precision all RE-CONVERTED Images: ', precision_score(real_values, predictions_reconverted, average=None))
    print('Precision WLI RE-CONVERTED imgs: ',
          precision_score(wli_tissue_types, predictions_reconverted_wli, average=None))
    print('Precision NBI RE-CONVERTED imgs: ',
          precision_score(nbi_tissue_types, predictions_reconverted_nbi, average=None, zero_division=1))
    print('Precision all AVERAGE Images: ', precision_score(real_values, predictions_average, average=None))
    print('Precision WLI AVERAGE imgs: ', precision_score(wli_tissue_types, predictions_average_wli, average=None))
    print('Precision NBI AVERAGE imgs: ',
          precision_score(nbi_tissue_types, predictions_average_nbi, average=None, zero_division=1))
    print('Precision all WEIGHTED Images: ', precision_score(real_values, predictions_weighted, average=None))
    print('Precision WLI WEIGHTED imgs: ', precision_score(wli_tissue_types, predictions_weighted_wli, average=None))
    print('Precision NBI WEIGHTED imgs: ',
          precision_score(nbi_tissue_types, predictions_weighted_nbi, average=None, zero_division=1))

    # Recall
    print('Recall all ORIGINAL images: ', recall_score(real_values, predictions_original, average=None))
    print('Recall WLI ORIGINAL imgs: ', recall_score(wli_tissue_types, predictions_original_wli, average=None))
    print('Recall NBI ORIGINAL imgs: ',
          recall_score(nbi_tissue_types, predictions_original_nbi, average=None, zero_division=1))
    print('Recall all CONVERTED images: ', recall_score(real_values, predictions_converted, average=None))
    print('Recall WLI CONVERTED imgs: ', recall_score(wli_tissue_types, predictions_converted_wli, average=None))
    print('Recall NBI CONVERTED imgs: ',
          recall_score(nbi_tissue_types, predictions_converted_nbi, average=None, zero_division=1))
    print('Recall all RE-CONVERTED images: ', recall_score(real_values, predictions_reconverted, average=None))
    print('Recall WLI RE-CONVERTED imgs: ', recall_score(wli_tissue_types, predictions_reconverted_wli, average=None))
    print('Recall NBI RE-CONVERTED imgs: ',
          recall_score(nbi_tissue_types, predictions_reconverted_nbi, average=None, zero_division=1))
    print('Recall all AVERAGE images: ', recall_score(real_values, predictions_average, average=None))
    print('Recall WLI AVERAGE imgs: ', recall_score(wli_tissue_types, predictions_average_wli, average=None))
    print('Recall NBI AVERAGE imgs: ',
          recall_score(nbi_tissue_types, predictions_average_nbi, average=None, zero_division=1))
    print('Recall all WEIGHTED images: ', recall_score(real_values, predictions_weighted, average=None))
    print('Recall WLI WEIGHTED imgs: ', recall_score(wli_tissue_types, predictions_weighted_wli, average=None))
    print('Recall NBI WEIGHTED imgs: ',
          recall_score(nbi_tissue_types, predictions_weighted_nbi, average=None, zero_division=1))

    # Confusion Matrices
    # Original Images
    compute_confusion_matrix(real_values, predictions_original, plot_figure=False,
                             dir_save_fig=output_images_dir + '/original_images_all_confusion_matrix.png')
    compute_confusion_matrix(wli_tissue_types, predictions_original_wli,
                             dir_save_fig=output_images_dir + '/original_images_WLI_confusion_matrix.png')
    compute_confusion_matrix(nbi_tissue_types, predictions_original_nbi,
                             dir_save_fig=output_images_dir + '/original_images_NBI_confusion_matrix.png')

    # Converted Images
    compute_confusion_matrix(real_values, predictions_converted, plot_figure=False,
                             dir_save_fig=output_images_dir + '/converted_images_all_confusion_matrix.png')
    compute_confusion_matrix(wli_tissue_types, predictions_converted_wli,
                             dir_save_fig=output_images_dir + '/converted_images_WLI_confusion_matrix.png')
    compute_confusion_matrix(nbi_tissue_types, predictions_converted_nbi,
                             dir_save_fig=output_images_dir + '/converted_images_NBI_confusion_matrix.png')

    # Re-converted Images
    compute_confusion_matrix(real_values, predictions_reconverted, plot_figure=False,
                             dir_save_fig=output_images_dir + '/reconverted_images_all_confusion_matrix.png')
    compute_confusion_matrix(wli_tissue_types, predictions_reconverted_wli,
                             dir_save_fig=output_images_dir + '/reconverted_images_WLI_confusion_matrix.png')
    compute_confusion_matrix(nbi_tissue_types, predictions_reconverted_nbi,
                             dir_save_fig=output_images_dir + '/reconverted_images_NBI_confusion_matrix.png')

    # Average
    compute_confusion_matrix(real_values, predictions_average, plot_figure=False,
                             dir_save_fig=output_images_dir + '/average_images_all_confusion_matrix.png')
    compute_confusion_matrix(wli_tissue_types, predictions_average_wli,
                             dir_save_fig=output_images_dir + '/average_images_WLI_confusion_matrix.png')
    compute_confusion_matrix(nbi_tissue_types, predictions_average_nbi,
                             dir_save_fig=output_images_dir + '/average_images_NBI_confusion_matrix.png')

    # Weighted
    compute_confusion_matrix(real_values, predictions_weighted, plot_figure=False,
                             dir_save_fig=output_images_dir + '/weighted_images_all_confusion_matrix.png')
    compute_confusion_matrix(wli_tissue_types, predictions_weighted_wli,
                             dir_save_fig=output_images_dir + '/weighted_images_WLI_confusion_matrix.png')
    compute_confusion_matrix(nbi_tissue_types, predictions_weighted_nbi,
                             dir_save_fig=output_images_dir + '/weighted_images_NBI_confusion_matrix.png')
    path_yaml = output_images_dir + '/performance_analysis.yalm'
    save_yaml(path_yaml, data_yaml)


def merge_reconstructed_gan_results(csv_file_dir, output_dir=None, gt_file=None):
    """
    Merge the results from a file containing Cycle-GAN generated results, with the names converted, and re-converted
    Parameters
    ----------
    csv_file_dir :
    output_dir :
    gt_file :

    Returns
    -------

    """
    classes = ['CIS', 'HGC', 'HLT', 'LGC', 'NTL']
    # read the csv file
    data_frame = pd.read_csv(csv_file_dir)
    # prepare the headers for the new data-frame
    original_header = list(data_frame.columns)
    #original_header.remove('real values')
    #original_header.remove('real values.1')
    original_header = [f for f in original_header if 'real values' not in f]
    original_header.remove('over all')
    headers = list()
    headers += original_header
    header_1 = ['converted' if x == 'fname' else x + '_C' for x in original_header]
    header_2 = ['reconverted' if x == 'fname' else x + '_R' for x in original_header]

    headers += header_1
    headers += header_2
    headers.append('original imaging')
    headers.append('prediction original')
    headers.append('prediction converted')
    headers.append('prediction reconverted')
    headers.append('average prediction')
    headers.append('weighted prediction')
    headers.append('real class')
    new_df = pd.DataFrame(columns=headers)

    if gt_file:
        df_gt = pd.read_csv(gt_file)
        list_gt_images = df_gt['image_name'].tolist()
        img_type_gt_images = df_gt['imaging type'].tolist()
        list_real_values = df_gt['tissue type'].tolist()
        unique_classes = np.unique(list_real_values)

    list_img_predictions = data_frame['fname'].tolist()
    print(unique_classes)
    original_names = [f for f in list_img_predictions if 'converted' not in f]
    original_names = [f for f in original_names if 'reconverted' not in f]
    #for j, row in enumerate(data_frame.iterrows()):
    #    if 'reconverted' in row['fname']:
    #        print(j, row)

    for i, image_name in enumerate(tqdm.tqdm(original_names[:], desc='Arranging data')):
        indexs = [list_img_predictions.index(name) for name in list_img_predictions if image_name.replace('.png', '') in name]
        if image_name in list_gt_images:
            index_gt = list_gt_images.index(image_name)
            imaging_type = img_type_gt_images[index_gt]
            real_value_class = list_real_values[index_gt]
        else:
            imaging_type = None
            real_value_class = None
        data_original = data_frame.iloc[indexs[0]]
        data_converted = data_frame.iloc[indexs[1]]
        data_reconverted = data_frame.iloc[indexs[2]]
        predictions_original = [data_original['class_1'], data_original['class_2'], data_original['class_3'], data_original['class_4'], data_original['class_5']]
        predictions_converted = [data_converted['class_1'], data_converted['class_2'], data_converted['class_3'], data_converted['class_4'], data_converted['class_5']]
        predictions_reconverted = [data_reconverted['class_1'], data_reconverted['class_2'], data_reconverted['class_3'], data_reconverted['class_4'], data_reconverted['class_5']]

        average = (np.array(predictions_original) + np.array(predictions_converted) + np.array(predictions_reconverted))/5.
        weighted = (np.array(predictions_original)*0.6 + np.array(predictions_converted)*0.2 + np.array(predictions_reconverted)*0.2)
        index_average = list(average).index(np.max(average))
        index_weighted = list(weighted).index(np.max(weighted))
        if unique_classes[index_average] != classes[index_average]:
            print('average:', unique_classes[index_average], classes[index_average])
        if unique_classes[index_weighted] != classes[index_weighted]:
            print('weighted:', unique_classes[index_weighted], classes[index_weighted])

        # Add the information to the new data-frame according to the previous information
        new_row = pd.Series(data={'fname': data_original['fname'], 'class_1': data_original['class_1'], 'class_2': data_original['class_2'],
                                  'class_3': data_original['class_3'], 'class_4': data_original['class_4'], 'class_5': data_original['class_5'],
                                  'prediction original': data_original['over all'],
                                  'converted': data_converted['fname'], 'class_1_C': data_converted['class_1'], 'class_2_C': data_converted['class_2'],
                                  'class_3_C': data_converted['class_3'], 'class_4_C': data_converted['class_4'], 'class_5_C': data_converted['class_5'],
                                  'prediction converted': data_converted['over all'],
                                  're-converted': data_reconverted['fname'], 'class_1_R': data_reconverted['class_1'], 'class_2_R': data_reconverted['class_2'],
                                  'class_3_R': data_reconverted['class_3'], 'class_4_R': data_reconverted['class_4'], 'class_5_R': data_reconverted['class_5'],
                                  'prediction reconverted': data_reconverted['over all'],
                                  'original imaging': imaging_type,
                                  'average prediction': unique_classes[index_average],
                                  'weighted prediction': unique_classes[index_weighted],
                                  'real class': real_value_class})

        new_df.loc[i] = new_row

    if output_dir is None:
        destination_dir = os.path.split(csv_file_dir)[0]

    output_csv_file_dir = destination_dir + '/predictions_summarized.csv'
    new_df.to_csv(output_csv_file_dir, index=False)
    print(new_df)
    print(f'File saved at:{output_csv_file_dir}')


def merge_continuous_frames_results(csv_file_dir):
    print(csv_file_dir)

    ordered_classes = ['CIS', 'HGC', 'HLT', 'LGC', 'NTL']
    df = pd.read_csv(csv_file_dir)
    list_names = df['fname'].tolist()
    selected_names = [name for i, name in enumerate(list_names) if i % 8 ==0]
    list_class_1 = df['class_1'].tolist()
    list_class_2 = df['class_2'].tolist()
    list_class_3 = df['class_3'].tolist()
    list_class_4 = df['class_4'].tolist()
    list_class_5 = df['class_5'].tolist()

    group_class_1 = list()
    group_class_2 = list()
    group_class_3 = list()
    group_class_4 = list()
    group_class_5 = list()
    over_all_names = list()

    for i in range(len(selected_names)):
        init = i*8
        end = ((i+1)*8)
        group_class_1.append(np.mean(list_class_1[init:end]))
        group_class_2.append(np.mean(list_class_2[init:end]))
        group_class_3.append(np.mean(list_class_3[init:end]))
        group_class_4.append(np.mean(list_class_4[init:end]))
        group_class_5.append(np.mean(list_class_5[init:end]))

    for j, name in enumerate(selected_names):
        scores = [group_class_1[j], group_class_2[j], group_class_3[j], group_class_4[j], group_class_5[j]]
        over_all_names.append(ordered_classes[scores.index(np.amax(scores))])

    d = {'fname': selected_names, 'class_1': group_class_1, 'class_2': group_class_2, 'class_3': group_class_3,
        'class_4': group_class_4, 'class_5': group_class_5, 'over all': over_all_names}

    pd.DataFrame(data=d)
    new_df = pd.DataFrame(d)
    print(new_df)
    file_name = os.path.split(csv_file_dir)[0] + '/predictions_temporal_grouped.csv'
    new_df.to_csv(file_name, index=False)


def merge_multidomian_results(predictions_directory, gt_file, output_dir=None):

    """
    Given results of original + converted + re-converted merge the information of the different networks
    Parameters
    ----------
    predictions_directory :
    gt_file :
    output_dir :

    Returns
    -------

    """

    list_files = [f for f in os.listdir(predictions_directory) if f.endswith('.csv')]
    file_original = [f for f in list_files if 'original' in f].pop()
    file_reconverted = [f for f in list_files if 'reconverted' in f].pop()
    list_files = [f for f in list_files if 'reconverted' not in f]
    file_converted = [f for f in list_files if 'converted' in f].pop()

    df_original = pd.read_csv(predictions_directory + file_original)
    df_converted = pd.read_csv(predictions_directory + file_converted)
    df_reconverted = pd.read_csv(predictions_directory + file_reconverted)

    list_results_original = df_original['fname'].tolist()
    list_results_converted = df_converted['fname'].tolist()
    list_results_reconverted = df_reconverted['fname'].tolist()

    df_gt = pd.read_csv(gt_file)
    list_gt_images = df_gt['image_name'].tolist()
    img_type_gt_images = df_gt['imaging type'].tolist()
    list_real_values = df_gt['tissue type'].tolist()
    unique_classes = np.unique(list_real_values)

    original_header = list(df_original.columns)
    original_header = [f for f in original_header if 'real values' not in f]

    original_header.remove('over all')
    headers = list()
    headers += original_header
    header_1 = ['converted' if x == 'fname' else x + '_C' for x in original_header]
    header_2= ['converted' if x == 'fname' else x + '_R' for x in original_header]

    headers += header_1
    headers += header_2
    headers.append('original imaging')
    headers.append('prediction original')
    headers.append('prediction converted')
    headers.append('prediction reconverted')
    headers.append('average prediction 2')
    headers.append('weighted prediction 2')
    headers.append('average prediction 3')
    headers.append('weighted prediction 3')
    headers.append('real class')
    new_df = pd.DataFrame(columns=headers)

    for i, name in enumerate(tqdm.tqdm(list_results_original[:], desc='Analyzing data')):
        search_name = name.replace('.png', '')

        if name in list_gt_images:
            index_gt = list_gt_images.index(name)
            imaging_type = img_type_gt_images[index_gt]
            real_value_class = list_real_values[index_gt]
        else:
            imaging_type = None
            real_value_class = None

        index_original = list_results_original.index(name)
        data_original = df_original.iloc[index_original]

        img_name = [im_name for im_name in list_results_converted if search_name in im_name].pop()
        r_img_name = [im_name for im_name in list_results_reconverted if search_name in im_name].pop()

        index_converted = list_results_converted.index(img_name)
        data_converted = df_converted.iloc[index_converted]

        index_reconverted = list_results_reconverted.index(r_img_name)
        data_reconverted = df_reconverted.iloc[index_reconverted]

        predictions_original = [data_original['class_1'], data_original['class_2'], data_original['class_3'],
                                data_original['class_4'], data_original['class_5']]
        predictions_converted = [data_converted['class_1'], data_converted['class_2'],
                                 data_converted['class_3'], data_converted['class_4'],
                                 data_converted['class_5']]
        predictions_reconverted = [data_reconverted['class_1'], data_reconverted['class_2'],
                                 data_reconverted['class_3'], data_reconverted['class_4'],
                                 data_reconverted['class_5']]

        average_2 = (np.array(predictions_original) + np.array(predictions_converted)) / 5.
        weighted_2 = (np.array(predictions_original) * 0.7 + np.array(predictions_converted) * 0.3)

        average_3 = (np.array(predictions_original) + np.array(predictions_converted)
                     + np.array(predictions_reconverted)) / 5.
        weighted_3 = (np.array(predictions_original) * 0.7 + np.array(predictions_converted) * 0.2
                      + np.array(predictions_reconverted) * 0.1)

        index_average_2 = list(average_2).index(np.max(average_2))
        index_weighted_2 = list(weighted_2).index(np.max(weighted_2))

        index_average_3 = list(average_3).index(np.max(average_3))
        index_weighted_3 = list(weighted_3).index(np.max(weighted_3))

        new_row = pd.Series(data={'fname': data_original['fname'], 'class_1': data_original['class_1'],
                                  'class_2': data_original['class_2'],
                                  'class_3': data_original['class_3'], 'class_4': data_original['class_4'],
                                  'class_5': data_original['class_5'],
                                  'prediction original': data_original['over all'],
                                  'converted': data_converted['fname'],
                                  'class_1_C': data_converted['class_1'],
                                  'class_2_C': data_converted['class_2'],
                                  'class_3_C': data_converted['class_3'],
                                  'class_4_C': data_converted['class_4'],
                                  'class_5_C': data_converted['class_5'],
                                  'prediction converted': data_converted['over all'],
                                  'reconverted': data_reconverted['fname'],
                                  'class_1_R': data_reconverted['class_1'],
                                  'class_2_R': data_reconverted['class_2'],
                                  'class_3_R': data_reconverted['class_3'],
                                  'class_4_R': data_reconverted['class_4'],
                                  'class_5_R': data_reconverted['class_5'],
                                  'prediction reconverted': data_reconverted['over all'],
                                  'original imaging': imaging_type,
                                  'average prediction 2': unique_classes[index_average_2],
                                  'weighted prediction 2': unique_classes[index_weighted_2],
                                  'average prediction 3': unique_classes[index_average_3],
                                  'weighted prediction 3': unique_classes[index_weighted_3],
                                  'real class': real_value_class})
        new_df.loc[i] = new_row

    if output_dir is None:
        destination_dir = os.path.split(predictions_directory)[0]

    output_csv_file_dir = destination_dir + '/predictions_summarized.csv'
    new_df.to_csv(output_csv_file_dir, index=False)
    print(new_df)
    print(f'File saved at:{output_csv_file_dir}')



if __name__ == '__main__':
    dir_folder_experiments = ''
    compare_experiments(dir_folder_experiments, top_results=0.4)
