from matplotlib import pyplot as plt
import csv
from absl import app, flags, logging
from absl.flags import FLAGS
import os
import scipy.io
import numpy as np
import cv2
import tqdm
import pandas as pd
import ast
import shutil
import random

flags.DEFINE_string('directory', '', 'name of the model')
flags.DEFINE_list('sub_directories', [], 'train or predict')
flags.DEFINE_string('', '', 'backbone network')


def read_mask(dir_image):
    original_img = cv2.imread(dir_image)

    if original_img is None:
        print('Could not open or find the image:', args.input)
        exit(0)

    height, width, depth = original_img.shape
    img = cv2.resize(original_img, (256, 256))
    img = img / 255
    img = (img > 0.9) * 1.0
    return img


def read_img_results(dir_image):
    print(dir_image)
    original_img = cv2.imread(dir_image)

    if original_img is None:
        print('Could not open or find the image:', dir_image)
        exit(0)

    height, width, depth = original_img.shape
    img = cv2.resize(original_img, (256, 256))

    return img


def read_results_csv_plot(file_path):
    dice_values = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            dice_values.append([row[1], row[2]])

        return dice_values


def read_results_csv(file_path, row_id=0):
    dice_values = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            dice_values.append(float(row[row_id]))

        return dice_values


def print_box_plots(name_test_csv_file, name_validation_csv_file, save_directory):
    path_file_1 = name_test_csv_file
    path_file_2 = name_validation_csv_file

    list_dice_values_file_1 = read_results_csv(path_file_1, 2)
    list_dice_values_file_2 = read_results_csv(path_file_2, 2)
    data_dice = [list_dice_values_file_1, list_dice_values_file_2]

    list_precision_values_file_1 = read_results_csv(path_file_1, 3)
    list_precision_values_file_2 = read_results_csv(path_file_2, 3)
    data_precision_values = [list_precision_values_file_1, list_precision_values_file_2]

    list_recall_file_1 = read_results_csv(path_file_1, 4)
    list_recall_file_2 = read_results_csv(path_file_2, 4)
    data_recall = [list_recall_file_1, list_recall_file_2]

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(131)
    ax1.boxplot(data_dice[0], 1, 'gD')
    ax2 = fig1.add_subplot(132)
    ax2.boxplot(data_precision_values[0], 1, 'gD')
    ax3 = fig1.add_subplot(133)
    ax3.boxplot(data_recall[0], 1, 'gD')
    ax1.title.set_text('Dice Coeff')
    ax2.title.set_text('Precision')
    ax3.title.set_text('Recall')
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    ax3.set_ylim(0, 1)

    plt.savefig(save_directory + 'results_test.png')
    plt.close()

    fig2 = plt.figure(2)
    ax1 = fig2.add_subplot(131)
    ax1.boxplot(data_dice[1], 1, 'gD')
    ax2 = fig2.add_subplot(132)
    ax2.boxplot(data_precision_values[1], 1, 'gD')
    ax3 = fig2.add_subplot(133)
    ax3.boxplot(data_recall[1], 1, 'gD')
    ax1.title.set_text('Dice Coeff')
    ax2.title.set_text('Precision')
    ax3.title.set_text('Recall')
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    ax3.set_ylim(0, 1)
    plt.savefig(save_directory + 'results_val.png')
    plt.close()


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


def compare_results(dir_groundtruth, dir_predictions, dir_csv_file, save_directory):
    path_images_folder = dir_groundtruth + 'image/rgb/'
    path_masks_folder = dir_groundtruth + 'label/'
    list_dice_values = read_results_csv_plot(dir_csv_file)

    image_list = [f for f in os.listdir(path_images_folder) if os.path.isfile(os.pathjoin(path_images_folder, f))]
    mask_list = [f for f in os.listdir(path_masks_folder) if os.path.isfile(os.path.join(path_masks_folder, f))]
    predicted_masks = [f for f in os.listdir(dir_predictions) if os.path.isfile(os.path.join(dir_predictions, f))]

    for image in predicted_masks[:]:

        if image in mask_list:

            path_image = ''.join([path_images_folder, image])
            path_mask = ''.join([path_masks_folder, image])
            path_predicted = ''.join([dir_predictions, image])
            image_frame = read_img_results(path_image)
            mask_image = read_mask(path_mask)

            for counter, element in enumerate(list_dice_values):

                print(element[0])
                if image == element[0]:
                    dice_value = float(element[1])
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


if __name__ == '__main__':
    try:
        app.run(analyze_data)
    except SystemExit:
        pass