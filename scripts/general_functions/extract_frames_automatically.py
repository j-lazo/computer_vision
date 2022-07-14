import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
import time
import csv
import copy
import copy
import os
import cv2
from absl import app, flags, logging
from absl.flags import FLAGS
import tqdm


def crop_image_and_label(path_image, path_mask):
    image_rgb = cv2.imread(path_image)
    label_rgb = cv2.imread(path_mask)
    # print(image_rgb.shape)
    limit_y = 0
    list_min_x = []
    list_max_x = []
    list_min_y = []
    for index_y in range(image_rgb.shape[0]):
        # print(image_rgb[index_y].shape)
        row = image_rgb[index_y]

        perimeter_points = [index_x for index_x, element in enumerate(row)
                            if element[0] != 0 and element[1] != 255 and element[2] != 0]

        if perimeter_points:
            list_min_y.append(index_y)
            limit_y = limit_y + 1
            list_min_x.append(min(perimeter_points))
            list_max_x.append(max(perimeter_points))

    min_x = min(list_min_x)
    min_y = min(list_min_y)
    max_x = max(list_max_x)

    crop_img = image_rgb[min_y:min_y + limit_y, min_x:max_x]
    crop_label = label_rgb[min_y:min_y + limit_y, min_x:max_x]

    return (crop_img, crop_label)


def crop_image(image_array, min_x, min_y, max_x, max_y):
    space_y = 10
    space_x = 10
    limit_x, limit_y, z = np.shape(image_array)

    init_y = min_y - space_y
    init_x = min_x - space_x

    if init_y < 0:
       init_y = 0

    if init_x < 0:
       init_x = 0

    end_y = max_y + space_y
    end_x = max_x + space_x

    if end_y > limit_y:
       end_y = limit_y

    if end_x > limit_x:
       end_x = limit_x

    crop_img = image_array[init_y:end_y, init_x:end_x]

    return crop_img


def detect_border_and_crop_image(path_image):

    image_rgb = cv2.imread(path_image)
    limit_y = 0
    list_min_x = []
    list_max_x = []
    list_min_y = []
    for index_y in range(image_rgb.shape[0]):
        row = image_rgb[index_y]

        perimeter_points = [index_x for index_x, element in enumerate(row)
                            if element[0] != 0 and element[1] != 255 and element[2]
                            != 0]

        if perimeter_points:
            list_min_y.append(index_y)
            limit_y = limit_y + 1
            list_min_x.append(min(perimeter_points))
            list_max_x.append(max(perimeter_points))

    min_x = min(list_min_x)
    min_y = min(list_min_y)
    max_x = max(list_max_x)

    crop_img = image_rgb[min_y:min_y + limit_y, min_x:max_x]
    return crop_img


def calculate_elipse_hough(image_gray, high_threshold=0.6, sigma=3.0):
    edges = canny(image_gray, sigma=sigma,
                  low_threshold=0.05, high_threshold=high_threshold)
    init_time = time.time()
    plt.figure()
    plt.imshow(edges)
    plt.show()
    min_size = int(min(np.shape(edges))/3)
    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    result = hough_ellipse(edges, accuracy=5, threshold=50,
                           min_size=min_size)
    delta = time.time() - init_time
    print(delta)
    return result, delta


def detect_edge(image_rgb, high_threshold=0.6, sigma=3.0):
    """#

    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    image_gray = color.rgb2gray(image_rgb)

    min_radius = int(min(np.shape(image_gray))/4)

    image_gray = color.rgb2gray(image_rgb)
    edges = canny(image_gray, sigma=2.0,
                  low_threshold=0.01, high_threshold=0.2)

    plt.figure()
    plt.imshow(edges)
    plt.show()
    print(np.shape(edges))
    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    result = hough_ellipse(edges, accuracy=20, threshold=20,
                           min_size=min_radius)
    result.sort(order='accumulator')
    print(result)
    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    image_rgb[cy, cx] = (0, 0, 255)
    # Draw the edge (white) and the resulting ellipse (red)
    edges = color.gray2rgb(img_as_ubyte(edges))
    edges[cy, cx] = (250, 0, 0)

    plt.figure()
    plt.imshow(edges)
    plt.show()"""



    # this is in case you want to detect a circle
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)  # color.rgb2gray(image_rgb)
    min_radius = int(min(np.shape(image_gray))/3)

    circles = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, 1, min_radius,
                               param1=80, param2=35,
                               minRadius=min_radius)
    #print(circles)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(image_rgb, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(image_rgb, center, radius, (255, 0, 255), 3)

    plt.figure()
    plt.imshow(image_rgb)
    plt.show()

    counter = 0
    # iterates in case no edge was obtained
    #while len(result) == 0 and counter < 3:
    #    print(f' iteration {counter}, sigma:  {sigma}, h. threshold {high_threshold}')
    #    sigma = sigma + 0.05
    #    high_threshold = high_threshold - 0.05
    #   circles = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, 1, 20,
    #                              param1=60, param2=40,
    #                              minRadius=1)
    #    #result, delta = calculate_elipse_hough(image_gray, sigma=sigma, high_threshold=high_threshold)
    #    counter = counter + 1
    """
    if counter == 3:
        # I don't understant this part???
        image_rgb = np.zeros([2, 2, 2])
        cropped_image = image_rgb
        delta = 999

    else:

        # Estimated parameters for the ellipse
        best = list(result[-1])
        yc, xc, a, b = [int(round(x)) for x in best[1:5]]
        orientation = best[5]

        # Draw the ellipse on the original image
        cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
        copy_image_rgb = copy.copy(image_rgb)
        image_rgb[cy, cx] = (0, 255, 0)
        cropped_image = crop_image(copy_image_rgb, np.amin(cx), np.amin(cy),
                                np.amax(cx), np.amax(cy))"""

    cropped_image = 0
    delta = 0
    return image_rgb, cropped_image, delta


def extract_frames_video_auto(path_video_file, extracted_frames_dir, output_dir=os.getcwd()):
    path_video_file = os.path.normpath(path_video_file)
    extracted_frames_dir = os.path.normpath(extracted_frames_dir)
    output_dir = os.path.normpath(output_dir)

    if not (os.path.isfile(path_video_file)):
        raise f'Video path:{path_video_file} not found'

    image_list = sorted([f for f in os.listdir(extracted_frames_dir)
                  if os.path.isfile(os.path.join(extracted_frames_dir, f))])

    list_frames = [name.split('frame_')[-1].replace('.png', '') for name in image_list]
    print(list_frames)
    name_video = os.path.split(path_video_file)[-1]
    file_extension = name_video.split('.')[-1]
    case_id = name_video.replace('.' + file_extension, '')

    cap = cv2.VideoCapture(path_video_file)
    for j, frame_number in enumerate(list_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number) - 1)
        image_path = os.path.join(extracted_frames_dir, image_list[j])
        ret, img = cap.read()
        image = cv2.imread(image_path)
        edge_detected, cropped_img, time = detect_edge(img)


        #plt.figure()
        #plt.subplot(121)
        #plt.imshow(img)
        #plt.subplot(122)
        #plt.imshow(image)
        #plt.show()

   #     count += 1
    #for image in image_list:
    #    path_image = os.path.join(extracted_frames_dir, image)
    #    edge_detected, cropped_img, time = detect_edge(path_image)
    #    plt.imshow(cropped_img)
    #    if not(np.any(edge_detected)):
    #        continue
    #    else:
            # save the detected edges image
    #        cv2.imwrite(''.join([output_dir_edges, image]), edge_detected)
            # save the croped images
    #        cv2.imwrite(''.join([cropped_img, image]), edge_detected)


    #csv_file_name = path_images_folder + 'time_test.csv'

    #with open(csv_file_name, mode='w') as results_file:
    #    results_file_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)


def main(_argv):
    video_path = FLAGS.video_path
    save_dir = FLAGS.save_dir
    extracted_frames_dir = FLAGS.extracted_frames_dir
    extract_frames_video_auto(video_path, extracted_frames_dir, output_dir=save_dir)


if __name__ == '__main__':
    flags.DEFINE_string('video_path', '', 'path of the video')
    flags.DEFINE_string('save_dir', os.getcwd(), 'path where to save the images extracted')
    flags.DEFINE_string('extracted_frames_dir', '', 'path where frames were previously extracted')

    try:
        app.run(main)
    except SystemExit:
        pass


