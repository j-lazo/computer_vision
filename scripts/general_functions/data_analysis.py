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
import ast
import shutil
import random


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

    #plt.show()
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


def compare_results_overlay(dir_groundtruth, dir_predictions, dir_csv_file, save_directory=''):
    path_images_folder = dir_groundtruth + 'images/'
    path_masks_folder = dir_groundtruth + 'masks/'
    data_results = pd.read_csv(dir_csv_file)
    list_dice_values = data_results['DSC'].tolist()
    list_imgs_csv = data_results['Image'].tolist()

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


def analyze_performances(project_dir):

    """
    Analyze the performance of a directory or a list of directories
    :param project_dir:
    :return:
    """
    # 2DO: recognize directory or list

    results_list = os.listdir(project_dir + 'results/')
    results_list.remove('temp')
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
        print(project_dir)
        #olders_prediction = os.listdir(os.path.join(project_dir, 'results', experiment_id, 'predictions'))
        compare_box_plots(project_dir)


    elif type(project_dir) == list:
        results_list = os.listdir(project_dir + 'results/')
        results_list.remove('temp')
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
    else:
        print('type(project dir) not understood')


def extract_information_from_name(string_name):

    model_name = re.search('evaluation_results_test_0_(.+?)_lr_', string_name).group(1)

    #if 'Residual_Unet' in model_name:
    #    model_name = 'R-Unet'
    #elif 'Transpose_Res' in model_name:
    #    model_name = 'TR-Unet'

    date_experiment = re.search('_rgb_(.+?)_.csv', string_name).group(1)
    lr = re.search('lr_(.+?)_', string_name).group(1)
    bs = re.search('bs_(.+?)_', string_name).group(1)


    return lr, bs, model_name, date_experiment


def compare_experiments(dir_folder_experiments, selection_criteria=['evaluation_results_test_0'], dir_save_results='',
                        top_results=1.0):
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
    #pass
    #try:
    #    app.run(analyze_data)
    #except SystemExit:
    #    pass