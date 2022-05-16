import copy
import os
import scipy.io
import numpy as np
import cv2
import tqdm
import pandas as pd
import ast
import shutil
import random
import colorsys
from scipy.ndimage import zoom
from PIL import Image
import re
import datetime

from matplotlib import pyplot as plt
import tensorflow as tf

#import skimage


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y: start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(255, 255)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


def clipped_zoom(img, zoom_factor, **kwargs):
    """
    :param img:
    :param zoom_factor:
    :param kwargs:
    :return:
    """

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top + zh, left:left + zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        # trim_top = ((out.shape[0] - h) // 2)
        # trim_left = ((out.shape[1] - w) // 2)
        # out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


def adjust_brightness(image, gamma=1.0):
    """

    :param image:
    :param gamma:
    :return:
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def shift_hue(arr, hout):
    """

    :param arr:
    :param hout:
    :return:
    """

    rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
    hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)
    r, g, b, a = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv(r, g, b)
    h = hout
    r, g, b = hsv_to_rgb(h, s, v)
    arr = np.dstack((r, g, b, a))
    return arr


def colorize(image, hue):
    """
    Colorize PIL image `original` with the given
    `hue` (hue within 0-360); returns another PIL image.
    :param image:
    :param hue:
    :return:

    """

    arr = np.array(np.asarray(image).astype('float'))
    new_img = Image.fromarray(shift_hue(arr, hue/360.).astype('uint8'), 'RGBA')

    return new_img


def augment_image(img, mask=[]):
    """

    :param img:
    :param mask:
    :return:
    """
    augmented_imgs = []
    augmented_masks = []
    list_operations = []

    augmented_imgs.append(img)
    if mask:
        augmented_masks.append(mask)

    rows, cols, channels = img.shape
    # define the rotation matrixes
    rot1 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    rot2 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
    rot3 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 270, 1)

    # rotate the images
    im_rot1 = cv2.warpAffine(img, rot1, (cols, rows))
    im_rot2 = cv2.warpAffine(img, rot2, (cols, rows))
    im_rot3 = cv2.warpAffine(img, rot3, (cols, rows))
    augmented_imgs.append(im_rot1)
    augmented_imgs.append(im_rot2)
    augmented_imgs.append(im_rot3)

    # rotate the masks
    if mask:
        mask_rot1 = cv2.warpAffine(mask, rot1, (cols, rows))
        mask_rot2 = cv2.warpAffine(mask, rot2, (cols, rows))
        mask_rot3 = cv2.warpAffine(mask, rot3, (cols, rows))
        augmented_masks.append(mask_rot1)
        augmented_masks.append(mask_rot2)
        augmented_masks.append(mask_rot3)

    # flip images
    horizontal_img = cv2.flip(img, 0)
    vertical_img = cv2.flip(img, 1)
    augmented_imgs.append(horizontal_img)
    augmented_imgs.append(vertical_img)
    # flip masks
    if mask:
        horizontal_mask = cv2.flip(mask, 0)
        vertical_mask = cv2.flip(mask, 1)
        augmented_masks.append(horizontal_mask)
        augmented_masks.append(vertical_mask)

    list_of_images = copy.copy(augmented_imgs)
    if mask:
        list_of_masks = copy.copy(augmented_masks)

    # change brightness
    gammas = [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]

    for i in range(4):
        index = random.randint(0, len(list_of_images) - 1)
        img_choice = list_of_images[index]
        if mask:
            mask_choice = list_of_masks[index]
        image_brg = adjust_brightness(img_choice, random.choice(gammas))
        augmented_imgs.append(image_brg)
        if mask:
            augmented_masks.append(mask_choice)

    # zoom in
    index_2 = random.randint(0, len(list_of_images) - 1)
    img_choice_2 = list_of_images[index_2]
    zoom_in_img = clipped_zoom(img_choice_2, 1.2)
    if mask:
        mask_choice_2 = list_of_masks[index_2]
        zoom_in_mask = clipped_zoom(mask_choice_2, 1.2)

    augmented_imgs.append(zoom_in_img)
    if mask:
        augmented_masks.append(zoom_in_mask)

    # zoom out
    index_3 = random.randint(0, len(list_of_images) - 1)
    img_choice_3 = list_of_images[index_3]
    zoom_out_img = clipped_zoom(img_choice_3, 0.8)
    augmented_imgs.append(zoom_out_img)

    if mask:
        mask_choice_3 = list_of_masks[index_3]
        zoom_out_mask = clipped_zoom(mask_choice_3, 0.8)
        augmented_masks.append(zoom_out_mask)


    # change hue
    #index_4 = random.randint(0, len(list_of_images) - 1)
    #img_choice_4 = list_of_images[index_4]
    #mask_choice_4 = list_of_masks[index_4]
    #colorized = colorize(img_choice_4, np.random.randint(70, 220))
    #augmented_imgs.append(colorized)
    #augmented_masks.append(mask_choice_4)

    # change contrast
    index_5 = random.randint(0, len(list_of_images) - 1)
    img_choice_5 = list_of_images[index_5]
    stateless_random_contrast = tf.image.stateless_random_contrast(img_choice_5, lower=0.5, upper=0.9, seed = (i, 0))
    augmented_imgs.append(stateless_random_contrast.numpy())

    if mask:
        mask_choice_5 = list_of_masks[index_5]
        augmented_masks.append(mask_choice_5)


    #index_6 = random.randint(0, len(list_of_images) - 1)
    #img_choice_6 = list_of_images[index_6]
    #mask_choice_6 = list_of_masks[index_6]
    #noisy_image = skimage.util.random_noise(img_choice_6, mode='s&p')
    #augmented_imgs.append(noisy_image)
    #augmented_masks.append(mask_choice_6)

    index_7 = random.randint(0, len(list_of_images) - 1)
    img_choice_7 = list_of_images[index_7]
    saturated = tf.image.adjust_saturation(img_choice_7, 3)
    augmented_imgs.append(saturated.numpy())

    if mask:
        mask_choice_7 = list_of_masks[index_7]
        augmented_masks.append(mask_choice_7)

    if mask:
        return augmented_imgs, augmented_masks, list_operations
    else:
        return augmented_imgs, list_operations


def create_video_from_frames(dir_frames, name_video=''):
    img_array = []
    images_folder = sorted(os.listdir(dir_frames))

    for i, filename in enumerate(tqdm.tqdm(images_folder[:], desc='Reading Images:')):
        img = cv2.imread(dir_frames + filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    if name_video == '':
        path_frames = dir_frames.split('/')
        name_folder = path_frames[-2]
        dir_video = ''.join([dir_frames.replace(name_folder + '/', ''), name_folder, '.mp4'])

    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter(dir_video, fourcc, 15, size)
    print(f'video saved at: {dir_video}')

    # out = cv2.VideoWriter(name_video, cv2.VideoWriter_fourcc(*'avi'), 15, size)
    #for i, img_array in enumerate(tqdm.tqdm(images_folder[:], desc='Generating Video:')):

    for i in range(len(img_array)):
        print(i)
        out.write(img_array[i])
    out.release()


def arrange_data_from_gan(path_datafile, csv_annotations_dir, destination_dir=''):
    """

    Parameters
    ----------
    path_datafile :
    csv_annotations :

    Returns
    -------

    """

    list_imgs = [f.replace('.jpg', '') for f in os.listdir(path_datafile) if f.endswith('.jpg') or f.endswith('.png')]
    list_imgs = [f.replace('.png', '') for f in list_imgs]
    csv_path = [f for f in os.listdir(csv_annotations_dir) if f.endswith('.csv')].pop()
    df = pd.read_csv(csv_annotations_dir + csv_path)
    headers = list(df.columns)
    new_df = pd.DataFrame(columns=headers)
    list_labeled_images = df['image_name'].tolist()
    list_labeled_images = [f.replace('.png', '') for f in list_labeled_images]
    classification_images = df['tissue type'].tolist()
    unique_classes = np.unique(classification_images)

    if destination_dir == '':
        destination_dir = path_datafile

    destination_dirs = {}
    for element in unique_classes:
        destination_dirs[element] = destination_dir + element
        if not(os.path.isdir(destination_dir + element)):
            os.mkdir(destination_dir + element)

    j = 0
    for i, image_name in enumerate(tqdm.tqdm(list_labeled_images[:], desc='Arranging images')):
        for img_name in list_imgs:
            if image_name in img_name:
                index = list_labeled_images.index(image_name)
                new_row = df.iloc[index]
                new_row['image_name'] = img_name + '.png'
                new_df.loc[j] = new_row
                class_type = classification_images[index]
                j = j + 1
                img = cv2.imread("".join([path_datafile, img_name, '.jpg']))
                destination_path = ''.join([destination_dirs[class_type], '/', img_name, '.png'])
                cv2.imwrite(destination_path, img)

    print(new_df)
    output_csv_file_dir = destination_dir + 'annotations.csv'
    new_df.to_csv(output_csv_file_dir, index=False)
    print(f'File saved at:{output_csv_file_dir}')


def augment_dataset(files_path, destination_path='', visualize_augmentation=False, augment_maks=False):
    """
    Performs data augmentation given a directory containing images and masks
    :param files_path:
    :param destination_path:
    :return:
    """
    if augment_maks is True:
        files = os.listdir(files_path + 'images/')
        masks = os.listdir(files_path + 'masks/')

    else:
        files = os.listdir(files_path)

    if destination_path == '':
        destination_path = files_path
    else:
        if not(os.path.isdir(destination_path)):
            os.mkdir(destination_path)

            if augment_maks is True:
                os.mkdir(destination_path + 'images/')
                os.mkdir(destination_path + 'masks/')

    for i, element in enumerate(tqdm.tqdm(files[:], desc='Augmenting Dataset')):
        if augment_maks is True:
            if element not in masks:
                print(f'{element}, has no pair')

        if augment_maks is True:
            img = cv2.imread("".join([files_path, 'images/', element]))
            mask = cv2.imread("".join([files_path, 'masks/', element]))

        else:
            img = cv2.imread("".join([files_path, element]))

        if augment_maks is True:
            list_images, list_masks, list_operations, = augment_image(img, mask)

        else:
            list_images, list_operations, = augment_image(img)

        # visualize the augmentation

        if visualize_augmentation is True:
            plt.figure()
            for i in range(len(list_images)-1):
                plt.subplot(4, 4, i+1)
                plt.imshow(list_images[i])

            plt.show()

        # save the images
        for i, image in enumerate(list_images):
            if augment_maks is True:
                cv2.imwrite("".join([destination_path, 'images/', element[:-4], '_', str(i).zfill(3), '.png']), list_images[i])
                cv2.imwrite("".join([destination_path, 'masks/', element[:-4], '_', str(i).zfill(3), '.png']), list_masks[i])
            else:
                cv2.imwrite("".join([destination_path, element[:-4], '_', str(i).zfill(3), '.png']),
                            list_images[i])


def check_folder_exists(folder_dir, create_folder=False):
    """
    Checks if a directory exits. If the flag create folder is selected it creates the sub-folders 'images' and 'masks'
    inside the created directory
    :param folder_dir: The directory path to determine its existance or not
    :param create_folder: a flag to determine if the sub-structure images and mask should be created
    :return:
    """
    if not(os.path.isdir(folder_dir)):
        exists = False
        if create_folder is True:
            os.mkdir(folder_dir)
            os.mkdir(folder_dir + 'images')
            os.mkdir(folder_dir + 'masks')

    else:
        exists = True
    return exists


def convert_image_format(dir_images, input_format, target_format, output_directory):
    """

    :param dir_images:
    :param target_format:
    :param output_dir:
    :return:
    """

    def _read_img(img_dir):
        img =cv2.imread(img_dir)
        return img

    list_imgs = os.listdir(dir_images)
    for image in list_imgs:
        img = _read_img(dir_images + image)
        cv2.imwrite(''.join([output_directory, '/', img]).replace(input_format, target_format), img)


def convert_dataset_png2jpg(directory_dataset, output_directory=''):

    if output_directory == '':
        output_directory = directory_dataset

    list_imgs = [f for f in os.listdir(directory_dataset) if f.endswith('.png')]

    for i, image_name in enumerate(tqdm.tqdm(list_imgs, desc='Coping images')):
        image_dir = ''.join([directory_dataset, '/', image_name])
        image = cv2.imread(image_dir)
        save_name = ''.join([output_directory, image_name.replace('.png', '.jpg')])
        cv2.imwrite(save_name, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def generate_imaging_dataset(directory_dataset, target_directory):

    list_files = os.listdir(directory_dataset)
    list_sub_folders = [f for f in list_files if os.path.isdir(directory_dataset + f)]

    csv_file_name = [f for f in list_files if f.endswith('.csv')].pop()
    df = pd.read_csv(directory_dataset + csv_file_name)
    name_files_csv = df['image_name'].tolist()
    imaging_type_csv = df['imaging type'].tolist()
    save_dir_class_A = ''.join([target_directory, 'A/'])
    save_dir_class_B = ''.join([target_directory, 'B/'])
    if not os.path.isdir(save_dir_class_A):
        os.mkdir(save_dir_class_A)

    if not os.path.isdir(save_dir_class_B):
        os.mkdir(save_dir_class_B)

    for folder in list_sub_folders:
        directory_subfolder = directory_dataset + folder
        list_imgs = [f for f in os.listdir(directory_subfolder) if f.endswith('.png')]
        for i, image_name in enumerate(tqdm.tqdm(list_imgs, desc='Coping images')):
            if image_name in name_files_csv:
                index = name_files_csv.index(image_name)
                image_dir = ''.join([directory_dataset, folder, '/', image_name])
                image = cv2.imread(image_dir)
                if imaging_type_csv[index] == 'WLI':
                    save_name = ''.join([target_directory, 'A/', image_name.replace('.png', '.jpg')])
                if imaging_type_csv[index] == 'NBI':
                    save_name = ''.join([target_directory, 'B/', image_name.replace('.png', '.jpg')])

                cv2.imwrite(save_name, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def generate_training_and_validation_classification_sets(input_directory, output_directory,
                                          case_probabilities=0.5, test_dataset=False, sub_dirs=[], convert_data=[]):

    def _check_folder_exists(directory_path, sub_folders):
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)
        for folder in sub_folders:
            directory_sub_dir = ''.join([directory_path, '/', folder])
            if not os.path.isdir(directory_sub_dir):
               os.mkdir(directory_sub_dir)

    def _copy_imgs(input_directory, list_destination_folders, unique_cases=[], case_probabilities=0.5):

        if unique_cases != []:
            for sub_dir in unique_cases:
                files_path_images = "".join([input_directory, '/', sub_dir, '/'])
                original_images = sorted(os.listdir(files_path_images))

                for i, image_name in enumerate(tqdm.tqdm(original_images, desc='Reading images')):
                    destination_dir = random.choices(list_destination_folders, weights=case_probabilities).pop()
                    if convert_data:

                        input_format = convert_data[0]
                        target_format = convert_data[1]
                        img = cv2.imread(''.join([files_path_images, image_name]))
                        destination_dir = ''.join([destination_dir, sub_dir, '/',
                                                   image_name.replace(input_format, target_format)])
                        cv2.imwrite(destination_dir, img)

                    else:
                        original_dir = ''.join([files_path_images, image_name])
                        destination_dir = ''.join([destination_dir, sub_dir, '/', image_name])
                        shutil.copy(original_dir, destination_dir)
        else:
            files_path_images = input_directory
            original_images = sorted(os.listdir(files_path_images))

            for i, image_name in enumerate(tqdm.tqdm(original_images, desc='Reading images')):
                destination_dir = random.choices(list_destination_folders, weights=case_probabilities).pop()
                if convert_data:

                    input_format = convert_data[0]
                    target_format = convert_data[1]
                    img = cv2.imread(''.join([files_path_images, image_name]))
                    destination_dir = ''.join([destination_dir,
                                               image_name.replace(input_format, target_format)])
                    cv2.imwrite(destination_dir, img)

                else:
                    original_dir = ''.join([files_path_images, image_name])
                    destination_dir = ''.join([destination_dir, image_name])
                    shutil.copy(original_dir, destination_dir)

    unique_cases = []
    if sub_dirs:
        for sub_dir in sub_dirs:
            copy_dir = ''.join([input_directory, sub_dir, '/'])
            unique_cases.append([f for f in os.listdir(copy_dir) if os.path.isdir(copy_dir + f)])

    else:
        unique_cases.append([f for f in os.listdir(input_directory) if os.path.isdir(input_directory + f)])

    unique_cases = np.unique(unique_cases)
    print(f'Sub-classes found: {unique_cases}')

    training_dir = output_directory + 'train/'
    validation_dir = output_directory + 'val/'

    _check_folder_exists(training_dir, unique_cases)
    _check_folder_exists(validation_dir, unique_cases)
    list_destination_folders = [training_dir, validation_dir]

    if test_dataset is True:
        test_dir = output_directory + 'test/'
        _check_folder_exists(test_dir, unique_cases)
        list_destination_folders.append(test_dir)

    if sub_dirs:
        for sub_dir in sub_dirs:
            print(sub_dir)
            input_dir = ''.join([input_directory, sub_dir, '/'])
            _copy_imgs(input_dir, list_destination_folders, unique_cases, case_probabilities)

    else:
        _copy_imgs(input_directory, list_destination_folders, unique_cases, case_probabilities)


def generate_training_and_validation_segmenation_sets(input_directory, output_directory,
                                          training_percentage=0.5, test_dataset='False',
                                          input_sub_dirs=[], pairs_of_data=[], convert_data=[]):

    """
    By default splits an image and mask dataset into training/validation with a 0.5/0.5 rate
    If test dataset is selected, the percentage for each of them should be input in list form.
    By default it considers that input directory contains two folders: images and masks, where masks
    and images have the same name.
    If several sources of input data folders are considered used
    :param input_directory: A directory where the dataset to be split is located
    :param output_directory: Location of the output directory
    :param test_dataset: (bool) False by default. Splits the dataset into train/val/test
    :param input_sub_dirs (list of strings) If the original dataset is contained in different folders, indicate the name of
    the folders to be considered inside 'Input directory'
    :param training_percentage: (float, list) default 0.5 If test_dataset is True, a list with the
    percentages for [train, val, test] should be indicated
    :param pairs_of_data: (list of strings) default []. If the pairs of data between the annotations and the data have different
    indication indicate its extensions [data_extension, annotation_extension] e.g.: ['.npy', '.png']
    : param convert_data: (list) [] ['original_format', ['target_format']] if selected converts data from a format to another, 
    for example ['.jpg', '.png'] converts images from '.jpg' to '.png'
    :return:
    """

    training_dir = output_directory + 'train/'
    validation_dir = output_directory + 'val/'

    check_folder_exists(training_dir, create_dir=True)
    check_folder_exists(validation_dir, create_dir=True)
    list_destination_folders = [training_dir, validation_dir]

    if test_dataset is True:
        test_dir = output_directory + 'test/'
        check_folder_exists(test_dir, create_dir=True)
        list_destination_folders.append(test_dir)

    def _copy_imgs(input_directory, list_destination_folders, training_percentage):
        files_path_images = "".join([input_directory, 'images/'])
        files_path_masks = "".join([input_directory, 'masks/'])
        original_images = sorted(os.listdir(files_path_images))
        label_images = sorted(os.listdir(files_path_masks))

        original_images = [image[:-4] for image in original_images]
        label_images = [image[:-4] for image in label_images]

        for i, image_name in enumerate(tqdm.tqdm(original_images, desc='Reading images and masks')):

            destination_dir = random.choices(list_destination_folders, weights=training_percentage).pop()
            if image_name in label_images:

                if not convert_data:
                    if pairs_of_data:
                        name_img = ''.join([image_name, pairs_of_data[0]])
                        name_mask = ''.join([image_name, pairs_of_data[1]])
                    else:
                        name_img = ''.join([image_name, '.png'])
                        name_mask = ''.join([image_name, '.png'])

                    shutil.copy(''.join([files_path_images, name_img]), ''.join([destination_dir, 'images/', name_img]))
                    shutil.copy(''.join([files_path_masks, name_mask]), ''.join([destination_dir, 'masks/', name_mask]))

                else:
                    input_format = convert_data[0]
                    target_format = convert_data[1]

                    img = cv2.imread(''.join([files_path_images, image_name, input_format]))
                    cv2.imwrite(''.join([destination_dir, 'images/', image_name, target_format]), img)

                    mask = cv2.imread(''.join([files_path_masks, image_name, input_format]))
                    cv2.imwrite(''.join([destination_dir, 'masks/', image_name, target_format]), mask)


            else:
                print(f'the pair of {image_name} does not exists')


    if not input_sub_dirs :
        _copy_imgs(input_directory, list_destination_folders, training_percentage)

    else:
        for sub_folder in input_sub_dirs:
            print(sub_folder)
            input_dir = ''.join([input_directory, sub_folder, '/'])
            _copy_imgs(input_dir, list_destination_folders, training_percentage)


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


def create_annotations_file_tissue_classification(directory, file_name='', file_extension='.csv'):
    """
    Creates an annotation file according to the structure of the directory tree. The output files could be
    '.csv' or '.json'. The function only considers up to a maximum of two sub-directories.

    :param directory: (str) path to the directory to analyze
    :param file_name: (str) name to save the file
    :param file_extension: (str) .csv or .json
    :return: a file with the annotations according to the structure of the directory
    """

    def update_dictionary_class(dictionary, image_name, utility='', clearness='', resolution='',
                                        anatomical_region='', imaging_type='', type_artifact='',
                                        tissue_type='', fov_shape='', roi=[]):

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

        anatomical_region = 'bladder'

        dictionary.update({image_name: {'useful': utility, 'clear': clearness, 'resolution': resolution,
                                        'anatomical region': anatomical_region, 'imaging type': imaging_type,
                                        'type artifact': type_artifact,
                                        'tissue type': tissue_type, 'fov shape': fov_shape, 'ROI': roi}})
        return dictionary

    dictionary = {}
    list_subfolders_with_paths = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    print(f'sub folder found: {list_subfolders_with_paths} in {directory}')

    for sub_folder in list_subfolders_with_paths:
        sub_folder_dir = os.path.join(directory, sub_folder)
        list_imgs = sorted(os.listdir(sub_folder_dir))

        for i, image_name in enumerate(tqdm.tqdm(list_imgs, desc='Reading images')):
            tissue_type = sub_folder
            dictionary = update_dictionary_class(dictionary, image_name, tissue_type=tissue_type)

    df = pd.DataFrame(data=dictionary).T
    name_file = save_data_frame_to_file(df, file_name, file_extension, directory)
    print(f'data file saved at {name_file}')


def create_annotations_file_kvasir_classification(directory, file_name='', file_extension='.csv'):
    """
    Creates an annotation file according to the structure of the directory tree. The output files could be
    '.csv' or '.json'. The function only considers up to a maximum of two sub-directories.

    :param directory: (str) path to the directory to analyze
    :param file_name: (str) name to save the file
    :param file_extension: (str) .csv or .json
    :return: a file with the annotations according to the structure of the directory
    """

    def update_dictionary_kvassir_class(dictionary, image_name, utility='', clearness='', resolution='',
                                        anatomical_region='', imaging_type='', type_artifact='',
                                        tissue_type='', fov_shape='', roi=[]):

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
        list_upper_gi = ['esophagitis-b-d', 'esophagitis-a', 'barretts-short-segment', 'barretts']
        list_lower_gi = ['hemorrhoids', 'polyps', 'ulcerative-colitis-grade-0-1', 'ulcerative-colitis-grade-1',
                         'ulcerative-colitis-grade-1-2', 'ulcerative-colitis-grade-2', 'ulcerative-colitis-grade-2-3',
                         'ulcerative-colitis-grade-3']

        if tissue_type in list_upper_gi:
            anatomical_region = 'upper-gi-tract'
        elif tissue_type in list_lower_gi:
            anatomical_region = 'lower-gi-tract'

        dictionary.update({image_name: {'useful': utility, 'clear': clearness, 'resolution': resolution,
                                        'anatomical region': anatomical_region, 'imaging type': imaging_type,
                                        'type artifact': type_artifact,
                                        'tissue type': tissue_type, 'fov shape': fov_shape, 'ROI': roi}})
        return dictionary

    dictionary = {}
    list_subfolders_with_paths = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    print(f'sub folder found: {list_subfolders_with_paths} in {directory}')

    for sub_folder in list_subfolders_with_paths:
        sub_folder_dir = os.path.join(directory, sub_folder)
        list_imgs = sorted(os.listdir(sub_folder_dir))

        for i, image_name in enumerate(tqdm.tqdm(list_imgs, desc='Reading images')):
            tissue_type = sub_folder
            dictionary = update_dictionary_kvassir_class(dictionary, image_name, tissue_type=tissue_type)

    df = pd.DataFrame(data=dictionary).T
    name_file = save_data_frame_to_file(df, file_name, file_extension, directory)
    print(f'data file saved at {name_file}')


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


def create_annotations_file_old_data(directory, file_name='', file_extension='.csv'):
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
    :return: (bool), list
    """
    test_list = os.listdir(path_dir)
    list_sub_folfers = [folder for folder in os.listdir(path_dir)
                        if os.path.isdir(os.path.join(path_dir, folder))]
    if len(list_sub_folfers) > 1:
        return True, list_sub_folfers
    else:
        return False, list_sub_folfers


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


def copy_data_folder(original_dir, destination_dir, selected_items=[]):
    """

    :param original_dir:
    :param destination_dir:
    :param selected_items:
    :return:
    """

    list_files_original_dir = os.listdir(original_dir)
    for i, element in enumerate(tqdm.tqdm(list_files_original_dir[:], desc='Copying files ')):
        source_file = original_dir + element
        destination_file = destination_dir + element
        if selected_items:
            if element in selected_items:
                shutil.copy(source_file, destination_file)
        else:
            shutil.copy(source_file, destination_file)


def check_folder_exists(path_dir, create_dir=False):
    """
    Check if path exists if create path, then creates directory
    :param path_dir: path directory
    :return: (bool)
    """
    exists = os.path.isdir(path_dir)

    if not exists and create_dir is True:
        os.mkdir(path_dir)

    return exists


def rearrange_data(dir_data, destination_dir=''):
    list_patient_cases = [f for f in os.listdir(dir_data) if os.path.isdir(dir_data + f)]
    list_patient_cases.remove('all_cases')
    if destination_dir == '':
        new_dir = dir_data + 'new_dir/'
        os.mkdir(new_dir)
        destination_dir = new_dir
    for case in list_patient_cases:
        destination_case_dir = destination_dir + case
        sub_folders, list_subfolders = determine_if_subfolders_exists(dir_data + case)
        if '.DS_Store' in list_subfolders:
            list_subfolders.remove('.DS_Store')
        if sub_folders:
            for sub_folder in list_subfolders:
                sub_path = ''.join([dir_data, case, '/', sub_folder])
                ss_folder, ss_folders = determine_if_subfolders_exists(sub_path)
                if ss_folder:
                    pass
                else:
                    list_imgs = os.listdir(sub_path)


def reshape_data_blocks(directory_data):

    list_data = os.listdir(directory_data)
    list_data.remove('.DS_store')
    for element in list_data:
        data_file = np.load(directory_data + element, allow_pickle=True)
        print(np.shape(data_file))


def arrange_dataset(directory_path):

    cases_subdirs = [f for f in os.listdir(directory_path) if os.path.isdir(directory_path + f)]
    cases_subdirs.remove('all_cases')

    for case in cases_subdirs:
        print(case)
        list_sub_dirs = os.listdir(directory_path + case)
        csv_file = [f for f in list_sub_dirs if f.endswith('.csv')].pop()
        list_sub_dirs.remove(csv_file)
        df = pd.read_csv(directory_path + case + '/' + csv_file)
        heads = list(df.columns)
        file_dict = df.set_index(heads[0]).T.to_dict('list')
        for sub_dir in list_sub_dirs:
            list_imgs = os.listdir(os.path.join(directory_path, case, sub_dir))
            for image in list_imgs:
                if image in file_dict:
                    if ' ' in image:
                        new_image = image.replace(' ', '')
                        print('renaming: ', image, new_image)
                        file_dict[new_image] = file_dict.pop(image)
                        old_img_name = ''.join([directory_path, case, '/', sub_dir, '/', image])
                        new_img_name = ''.join([directory_path, case, '/', sub_dir, '/', new_image])
                        os.rename(old_img_name, new_img_name)

        new_df = pd.DataFrame.from_dict(file_dict, orient='index')
        new_df.rename(columns={x: heads[i+1] for i, x in enumerate(new_df.columns)}, inplace=True)
        new_csv_file_name = csv_file.replace('.csv', '_(1).csv')
        df.to_csv(directory_path + case + '/' + new_csv_file_name)


def build_csv_from_other_csv(directory_files, csv_dir, output_csv_file_dir=''):
    """
    Given a directory path (directory_files), and a csv file (csv_dir) that contains the names of the files
    in the directory and more, builds another csv file with only the information of the files existing in
    directory_files

    Args:
        directory_files (str): directory of the files
        csv_dir (str): csv from where to copy the files
        output_csv_file_dir (str): output absolute path of the file where to save the ones found in directory_files

    Returns:

    """
    list_files = sorted(os.listdir(directory_files))
    data_frame = pd.read_csv(csv_dir)
    list_names_imgs = data_frame['image_name'].tolist()
    headers = list(data_frame.columns)
    new_df = pd.DataFrame(columns=headers)
    sub_folders = [f for f in list_files if os.path.isdir(directory_files + f)]
    if sub_folders:
        list_imgs = list()
        for sub_folder in sub_folders:
            list_imgs += [f for f in os.listdir(''.join([directory_files, sub_folder, '/'])) if f.endswith('.png')]

        i = 0
        for image in list_imgs:
            if image in list_names_imgs:
                index = list_names_imgs.index(image)
                new_df.loc[data_frame.index[i]] = data_frame.iloc[index]
                i += 1

    else:
        i = 0
        for image in list_files:
            if image in list_names_imgs:
                index = list_names_imgs.index(image)
                new_df.loc[data_frame.index[i]] = data_frame.iloc[index]
                i += 1

    print(new_df)
    if output_csv_file_dir == '':
        case_name = directory_files.split('/')[-2]
        output_csv_file_dir = ''.join([directory_files, 'annotations_', case_name, '.csv'])

    new_df.to_csv(output_csv_file_dir, index=False)
    print(f'File saved at:{output_csv_file_dir}')


def copy_matching_string_data(matching_string, original_dir, destination_dir):
    """
    Given the directory to a dataset of images, if some name of the images matches a specified string, it copies all
    the matching instances to the destination dir
    Parameters
    ----------
    matching_string : (str)
    original_dir : (str)
    destination_dir : (str)

    Returns
    -------

    """

    list_imgs = [file for file in os.listdir(original_dir) if matching_string in file]

    for i, file in enumerate(tqdm.tqdm(list_imgs[:], desc='Copying files')):
        shutil.copy(original_dir + file, destination_dir + file)


def merge_annotations_data(annotations_list, selected_elements=[], output_dir=''):

    data_frames = [[] for _ in range(len(annotations_list))]

    for i, file in enumerate(tqdm.tqdm(annotations_list[:], desc='Reading csv files')):
        data_frames[i] = pd.read_csv(file).set_index('image_name')

    df = pd.concat(data_frames)
    print(df)

    list_index = df.index.values.tolist()
    for name in list_index:
        df = df.rename(index={name: name.replace(' ', '')})

    list_index = df.index.values.tolist()
    new_df = pd.DataFrame(columns=df.columns)
    print(new_df)

    if selected_elements:
        for element in selected_elements:
            if element in list_index:
                idx = list_index.index(element)
                rowData = df.loc[element, :]
                new_df = new_df.append(rowData)
            else:
                print(element, 'not found')

    print(new_df)
    if output_dir != '':
        save_dir = output_dir + '/all_annotations.csv'
    else:
        save_dir = os.getcwd() + '/all_annotations.csv'
    df.to_csv(save_dir, index=False)


def merge_multi_domain_data(directory_dataset, keywords=['converted', 'reconverted'], csv_annotations=None,
                            output_dir=None, output_shape=(256,256), compress=False):
    """

    Parameters
    ----------
    directory_dataset :
    keywords :
    csv_annotations :
    output_dir :

    Returns
    -------

    """

    list_img_files = os.listdir(directory_dataset)

    if output_dir is None:
        output_dir = directory_dataset

    if csv_annotations:
        df_gt = pd.read_csv(csv_annotations)
        list_gt_images = df_gt['image_name'].tolist()
        img_type_gt_images = df_gt['imaging type'].tolist()
        list_real_values = df_gt['tissue type'].tolist()
        unique_classes = np.unique(list_real_values)

    original_names = copy.copy(list_img_files)

    # eliminate images that have repeated names according to the matching keywords
    for key_word in keywords:
        original_names = [f for f in original_names if key_word not in f]

    # iterate over each of the "base" names
    for i, image_name in enumerate(tqdm.tqdm(original_names[:], desc='Arranging data')):
        search_name = image_name[:-8]
        transformation_id = re.findall(r"(\d+).png", image_name).pop() + '.png'
        matching_names = list()
        conv_name = ''.join([search_name, '_', 'converted__', transformation_id])
        reconv_name = ''.join([search_name, '_', 'reconverted__', transformation_id])

        ordered_matching_names = [image_name]
        if conv_name in list_img_files:
            ordered_matching_names.append(conv_name)
            if reconv_name in list_img_files:
                ordered_matching_names.append(reconv_name)
                array_imgs = np.zeros((3, output_shape[0], output_shape[1], 3)).astype(int)
                for j, img_name in enumerate(ordered_matching_names):
                    img = tf.keras.utils.load_img(directory_dataset + img_name, target_size=output_shape,
                                                  interpolation='bilinear')
                    img_array = tf.keras.utils.img_to_array(img).astype(int)
                    array_imgs[j] = img_array.astype(int)

                    if compress is False:
                        save_name = ''.join([search_name, '_', transformation_id.replace('.png', '.npy')])
                        save_dir_name = output_dir + save_name
                        np.save(save_dir_name, array_imgs)

                    else:
                        save_name = ''.join([search_name, '_', transformation_id.replace('.png', '.npz')])
                        save_dir_name = output_dir + save_name
                        np.savez_compressed(save_dir_name, array_imgs)

    print(f'files saved at:{output_dir}')


def generate_annotations_npy_from_csv(directory_npy_data, csv_sample_file, output_name=None):

    df = pd.read_csv(csv_sample_file)
    new_df = df.copy()
    list_names = df['image_name'].tolist()
    list_names = [name.replace('.png', '') for name in list_names]

    list_sub_dirs = [f for f in os.listdir(directory_npy_data) if os.path.isdir(directory_npy_data + f)]
    for sub_dir_name in list_sub_dirs:
        sub_dir = os.path.join(directory_npy_data, sub_dir_name)
        list_files = os.listdir(sub_dir)
        for i, npy_file in enumerate(tqdm.tqdm(list_files[:], desc=f'Reading data in {sub_dir_name}')):
            if npy_file.replace('.npz', '') in list_names:
                index = list_names.index(npy_file.replace('.npz', ''))
                row = new_df.iloc[index]
                row['image_name'] = npy_file
                new_df.loc[index] = row
            else:
                print(f'.png match of {npy_file} not found')

    if output_name:
        output_file_name = directory_npy_data + output_name
    else:
        output_file_name = directory_npy_data + 'annotations_dataset.csv'

    print(new_df)
    new_df.to_csv(output_file_name, index=False)
    print(f'file saved at {output_file_name}')


def generate_annotations_augmented_dataset(directory_augmented_data, csv_sample_file, output_name=None):

    df = pd.read_csv(csv_sample_file)
    headers = list(df.columns)
    new_df = pd.DataFrame(columns=headers)
    list_names = df['image_name'].tolist()
    list_names = [name.replace('.png', '') for name in list_names]
    j = 0
    list_sub_dirs = [f for f in os.listdir(directory_augmented_data) if os.path.isdir(directory_augmented_data + f)]
    for sub_dir_name in list_sub_dirs:
        sub_dir = os.path.join(directory_augmented_data, sub_dir_name)
        list_files = os.listdir(sub_dir)
        for i, augmented_file in enumerate(tqdm.tqdm(list_files[:], desc=f'Reading data in {sub_dir_name}')):
            if augmented_file[:-8] in list_names:
                index = list_names.index(augmented_file[:-8])
                new_row = df.iloc[index].copy()
                new_row['image_name'] = augmented_file
                new_df.loc[j] = new_row
                #print(new_df.loc[j])
                j += 1
            else:
                print(f'match of {augmented_file} not found')

    if output_name:
        output_file_name = directory_augmented_data + output_name
    else:
        output_file_name = directory_augmented_data + 'annotations_dataset.csv'

    print(new_df)
    new_df.to_csv(output_file_name, index=False)
    print(f'file saved at {output_file_name}')


def extract_grames_GI_dataset(folder_dir):
    list_videos = [f for f in os.listdir(folder_dir) if f.endswith('.mp4')]

    case_name = folder_dir.split('/')[-2]

    for video in list_videos:
        path_video = folder_dir + video
        cap = cv2.VideoCapture(path_video)
        name_video = video.replace('.mp4', '')

        if not os.path.isdir(folder_dir + name_video):
            os.mkdir(folder_dir + name_video)

        if not cap.isOpened():
            print("could not open :", path_video)
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        percent = 0.03
        for frame_no in tqdm.tqdm(range(total_frames), desc=f'Extracting frames from video {path_video}'):
            if random.random() < percent:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no-1)
                res, frame = cap.read()
                frame_name = ''.join([case_name, '_', name_video, '_frame_', str(frame_no).zfill(4), '.png'])
                resized = cv2.resize(frame, (300, 300))
                print(frame_name)
                img_dir = os.path.join(folder_dir, name_video, frame_name)
                cv2.imwrite(img_dir, resized)



if __name__ == "__main__":
    path_dir = ''
    reshape_data_blocks(path_dir)
    pass

