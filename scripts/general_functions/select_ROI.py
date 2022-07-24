import cv2
import os
import pandas as pd
import copy
import os
import cv2
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
from matplotlib import pyplot as plt
import pyautogui


coords = []
drawing = False
drawing_lines = True


def click_and_select_roi(event, x, y, flag, image):
    """
    Callback function, called by OpenCV when the user interacts
    with the window using the mouse. This function will be called
    repeatedly as the user interacts.
    (https://www.timpoulsen.com/2018/handling-mouse-events-in-opencv.html)
    """
    # get access to a couple of global variables we'll need
    global coords, drawing, drawing_lines, image_name, imgs_rois
    img_width, img_height, _ = np.shape(image)
    if img_width >= sc_width or img_height >= sc_height:
        image = image_resize(image, width=int(0.7 * sc_width))

    clone = image.copy()
    if imgs_rois[image_name]:
        cords_frame = imgs_rois[image_name]

        cv2.rectangle(clone, cords_frame[0], cords_frame[1], (0, 255, 0), 2)
        cv2.imshow(image_name, clone)
        if event == cv2.EVENT_RBUTTONDOWN:
            imgs_rois[image_name] = []
            cv2.imshow(image_name, image)
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            # user has clicked the mouse's left button
            drawing = True
            # save those starting coordinates
            coords = [(x, y)]

        elif event == cv2.EVENT_MOUSEMOVE and drawing is True:
            # user is moving the mouse within the window
            if drawing is True:
                # if we're in drawing mode, we'll draw a green rectangle
                # from the starting x,y coords to our current coords
                cv2.rectangle(clone, coords[0], (x, y), (0, 255, 0), 2)
                cv2.imshow(image_name, clone)

        elif event == cv2.EVENT_MOUSEMOVE and drawing_lines is True:
            clone = image.copy()

            cv2.line(clone, (x, 0), (x, img_width), (0, 255, 0), 2)
            cv2.line(clone, (0, y), (img_height, y), (0, 255, 0), 2)
            cv2.imshow(image_name, clone)

        elif event == cv2.EVENT_LBUTTONUP:
            # user has released the mouse button, leave drawing mode
            # and crop the photo
            drawing = False
            # save our ending coordinates
            coords.append((x, y))
            cv2.rectangle(clone, coords[0], (x, y), (0, 255, 0), 2)
            cv2.imshow(image_name, clone)
            drawing_lines = False
            imgs_rois[image_name] = coords


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def detect_edge(image_rgb, high_threshold=0.6, sigma=3.0):
    roi_coordinates = list()
    # this is in case you want to detect a circle
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)  # color.rgb2gray(image_rgb)
    min_radius = int(min(np.shape(image_gray))/3)

    circles = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, 1, min_radius,
                               param1=80, param2=35,
                               minRadius=min_radius)
    (h, w) = image_rgb.shape[:2]
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle outline
            radius = i[2]
            #cv2.circle(image_rgb, center, 15, (0, 255, 0), 3)
            #cv2.circle(image_rgb, center, radius, (255, 0, 255), 3)
            if np.shape(circles)[1] == 1:
                # draw a rectangle
                # represents the top left corner of rectangle
                left_x = int(center[0]) - radius
                left_y = int(center[1]) - radius
                #if left_x-5 > 0:
                #    left_x = left_x - 5
                #else:
                #    left_x = 0
                #if left_y-5 > 0:
                #    left_y = left_y - 5
                #else:
                #    left_y = 0
                start_point = (left_x, left_y)
                # represents the bottom right corner of rectangle
                right_x = int(center[0] + radius)
                right_y = int(center[1] + radius)
                #if right_x+5 < w:
                #    right_x = right_x + 20
                #else:
                #    right_x = w
                #if right_y+5 < h:
                #    right_y = right_y + 20
                #else:
                #    right_y = h
                end_point = (right_x, right_y)
                # Using cv2.rectangle() method
                # Draw a rectangle with blue line borders of thickness of 2 px
                image_rgb = cv2.rectangle(image_rgb, start_point, end_point, (0, 255, 0), 2)
                roi_width = right_x - left_x
                roi_height = right_y - left_y
                roi_coordinates = [center[0]/w, center[1]/h, roi_width/w, roi_height/h]
            else:
                roi_coordinates = list()

    return image_rgb, roi_coordinates


def annotate_roi(path_video_file, extracted_frames_dir, output_dir=''):
    global imgs_rois
    global sc_width, sc_height
    global image_name

    path_video_file = os.path.normpath(path_video_file)
    extracted_frames_dir = os.path.normpath(extracted_frames_dir)
    output_dir = os.path.normpath(output_dir)
    sc_width, sc_height = pyautogui.size()

    if not (os.path.isfile(path_video_file)):
        raise f'Video path:{path_video_file} not found'

    list_imgs = [f for f in os.listdir(extracted_frames_dir) if f.endswith('.png')]
    list_frames = [name.split('frame_')[-1].replace('.png', '') for name in list_imgs]
    name_video = os.path.split(path_video_file)[-1]
    file_extension = name_video.split('.')[-1]
    case_id = name_video.replace('.' + file_extension, '')
    cap = cv2.VideoCapture(path_video_file)

    width_video = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height_video = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    imgs_rois = dict.fromkeys(list_imgs, [])

    # wait for Esc or q key and then exit
    i = 0
    while True:
        roi_save = None
        if i == len(list_imgs):
            print(imgs_rois)
            break
        print(f'img: {i+1}/{len(list_imgs)}')
        image_name = list_imgs[i]
        img_dir = os.path.join(extracted_frames_dir, image_name)
        frame_number = list_frames[i]
        # read the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ret, image = cap.read()
        # make a copy of the image
        img_copy = copy.copy(image)
        copy_original_img = copy.copy(image)
        edge_detected, edges = detect_edge(img_copy)
        if edges:
            img_width, img_height, depth = np.shape(image)
            center_x, center_y, width, height = edges
            img_copy = cv2.circle(img_copy, (int(center_x * img_height), int(center_y * img_width)),
                                      radius=10, color=(0, 0, 255), thickness=-1)
            start_point = (int((center_x - (width / 2)) * img_height), int((center_y - (height / 2)) * img_width))
            # bottom right corner
            end_point = int((center_x + (width / 2)) * img_height), int((center_y + (height / 2)) * img_width)
            img_copy = cv2.rectangle(img_copy, start_point, end_point, (255, 255, 0), 2)

        if width_video >= sc_width or height_video >= sc_height:
            disp_img = image_resize(img_copy, width=int(0.7*sc_width))
        else:
            disp_img = img_copy

        cv2.imshow(image_name, disp_img)

        if imgs_rois[list_imgs[i-1]]:
            coords_frame = imgs_rois[list_imgs[i-1]]
            cv2.rectangle(img_copy, coords_frame[0], coords_frame[1], (0, 255, 0), 2)
            cv2.imshow(image_name, img_copy)
        """if imgs_rois[image_name]:
            print(imgs_rois)
            coords_frame = imgs_rois[image_name]
            #clone = image.copy()
            #cv2.rectangle(clone, coords_frame[0], coords_frame[1], (0, 255, 0), 2)
            cv2.imshow(image_name, disp_img)
        else:
            cv2.imshow(image_name, disp_img)
        """

        cv2.setMouseCallback(image_name, click_and_select_roi, image)
        new_key = cv2.waitKey(0) & 0xFF
        # wait for Esc or q key and then exit
        # previous frame
        if new_key == ord('a'):
            cv2.destroyWindow(image_name)
            i -= 1

        # next frame
        if new_key == ord('s'):
            #imgs_rois[image_name] == coords_frame
            cv2.destroyWindow(image_name)
            i += 1

        # repeat frame
        if new_key == ord('d'):
            imgs_rois[image_name] = list()
            image = copy_original_img
            cv2.destroyWindow(image_name)

        if new_key == ord('q'):
            cv2.destroyWindow(image_name)
            print(imgs_rois)
            i = len(list_imgs)
            break

        if new_key == 27 or new_key == ord("k"):
            pd.DataFrame.from_dict(imgs_rois)
            cv2.destroyAllWindows()
            break


def main(_argv):
    video_path = FLAGS.video_path
    save_dir = FLAGS.save_dir
    extracted_frames = FLAGS.extracted_frames_dir
    annotate_roi(video_path, extracted_frames, output_dir=save_dir)


if __name__ == '__main__':
    flags.DEFINE_string('video_path', '', 'path of the video')
    flags.DEFINE_string('extracted_frames_dir', '', 'path where frames were previously extracted')
    flags.DEFINE_string('save_dir', '', 'path where to save the images extracted')

    try:
        app.run(main)
    except SystemExit:
        pass