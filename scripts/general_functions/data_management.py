import os
import scipy.io
from matplotlib import pyplot as plt
import numpy as np
import cv2
import glob
import tqdm

def read_mat_files(file_dir):
    """
    Read a .mat file
    :param file_dir: directory where the .mat file is located
    :return: the mat file in dictionary format
    """

    mat = scipy.io.loadmat(file_dir)
    return mat



def analysze_dataset(path_dataset):
    """
    Chek all the original image sizes inside a directory and its subdirectoreis
    :param path_dataset: path of the dataset to analyze
    :return: list with all the unique image sizes
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

    def compare_imgs(image_1, image_2):
        difference = cv2.subtract(image_1, image_2)
        b, g, r = cv2.split(difference)
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            print('asies')
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

    #plt.figure()
    #plt.subplot(121)
    #init_x = 575
    #init_y = 155
    #plt.imshow(cv2.resize(big_img[init_y:init_y+704, init_x:init_x+719], (l_simage, w_simage)))
    #plt.subplot(122)
    #plt.imshow(small_img)
    #plt.show()
    for i in range(0, l_bimage-l_simage):
        for j in range(0, w_bimage-w_simage):
            print(i, j)
            #print(i, j)
            croped_image = big_img[j:j+round(719.7607), i:i+round(704.4466)]
            resized_img = cv2.resize(croped_image, (l_simage, w_simage))
            if compare_imgs(resized_img, small_img) is True:
                print(i, j)
                break





