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
#from pandas_ml import ConfusionMatrix

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
    list_extensions = ['.mpg', '.mp4', '.MP4']
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


def compute_confusion_matrix(gt_data, predicted_data, plot_figure=False, dir_save_fig=''):
    conf_matrix = confusion_matrix(gt_data, predicted_data)

    if plot_figure is True:

        uniques = np.unique(gt_data)
        group_percentages = [value for value in conf_matrix.flatten() / np.sum(conf_matrix)]

        size = len(list(uniques))
        list_uniques = list(uniques)
        short_list = list()
        for name in list_uniques:
            name_split = name.split('-')
            new_name = ''
            for splits in name_split:
                new_name = new_name.join([splits[0]])

            short_list.append(new_name)

        labels = np.asarray(group_percentages).reshape(size, size)
        sns.heatmap(conf_matrix, cmap='Blues', cbar=False, linewidths=.5,
                    yticklabels=list(uniques), xticklabels=list(short_list), annot=labels)
        plt.xlabel('True positives')
        if dir_save_fig != '':
            plt.savefig(dir_save_fig + '/confusion_matrix.png')

        plt.close()
        #plt.show()

    return conf_matrix

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

    dsc_values = {pair[1]:dsc_values[pair[1]] for pair in top_list if pair[1] in dsc_values}
    prec_values = {pair[1]:prec_values[pair[1]] for pair in top_list if pair[1] in prec_values}
    rec_values = {pair[1]:rec_values[pair[1]] for pair in top_list if pair[1] in rec_values}
    acc_values = {pair[1]:acc_values[pair[1]] for pair in top_list if pair[1] in acc_values}

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


if __name__ == '__main__':
    dir_folder_experiments = ''
    compare_experiments(dir_folder_experiments, top_results=0.4)
