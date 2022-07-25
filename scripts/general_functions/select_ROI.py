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
    global coords, drawing, drawing_lines, image_name, imgs_rois, previous_coords
    img_width, img_height, _ = np.shape(image)
    if img_width >= sc_width or img_height >= sc_height:
        image = image_resize(image, width=int(0.7 * sc_width))

    clone = image.copy()
    if imgs_rois[image_name]:
        cords_frame = imgs_rois[image_name]
        # convert them
        cx, cy, w, d = cords_frame
        clone = draw_rectanlge(clone, cx, cy, w, d)
        #cv2.rectangle(clone, cords_frame[0], cords_frame[1], (0, 255, 0), 2)
        cv2.imshow(image_name, clone)
        previous_coords = cords_frame
        if event == cv2.EVENT_RBUTTONDOWN:
            imgs_rois[image_name] = list()
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
            left_x, left_y = coords[0]
            right_x, right_y = coords[1]
            roi_width = right_x - left_x
            roi_height = right_y - left_y
            new_center_x = int(roi_width / 2) + left_x
            new_center_y = int(roi_height / 2) + left_y
            imgs_rois[image_name] = normalize_roi(clone, new_center_x, new_center_y, roi_width, roi_height)
            previous_coords = normalize_roi(clone, new_center_x, new_center_y, roi_width, roi_height)

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


def detect_edge(image_rgb, draw_shapes=False):
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
            if draw_shapes:
                cv2.circle(image_rgb, center, 15, (0, 255, 0), 3)
                cv2.circle(image_rgb, center, radius, (255, 0, 255), 3)
            if np.shape(circles)[1] == 1:
                # represents the top left corner of rectangle
                left_x = int(center[0]) - radius
                left_y = int(center[1]) - radius
                left_x = np.max([left_x-5, 0])
                left_y = np.max([left_y-5, 0])

                right_x = int(center[0] + radius)
                right_y = int(center[1] + radius)
                right_x = np.min([right_x + 5, w])
                right_y = np.min([right_y + 5, h])

                start_point = (left_x, left_y)
                end_point = (right_x, right_y)
                # Using cv2.rectangle() method
                image_rgb = cv2.rectangle(image_rgb, start_point, end_point, (255, 255, 0), 5)
                roi_width = right_x - left_x
                roi_height = right_y - left_y
                new_center_x = int(roi_width/2) + left_x
                new_center_y = int(roi_height/2) + left_y
                roi_coordinates = [new_center_x, new_center_y, roi_width, roi_height]
            else:
                roi_coordinates = list()

    return image_rgb, roi_coordinates


def draw_rectanlge(img, cx, cy, width, height):
    copy_image = img.copy()
    img_height, img_width, depth = np.shape(copy_image)
    w = int(width * img_width)
    h = int(height * img_height)
    # left corner
    center_x = int(cx * img_width)
    center_y = int(cy * img_height)
    left_x = center_x - int(w/2)
    left_y = center_y - int(h/2)
    right_x = center_x + int(w/2)
    right_y = center_y + int(h/2)
    start_point = left_x, left_y
    # bottom right corner
    end_point = right_x, right_y
    copy_image = cv2.circle(copy_image, (int(center_x * img_height), int(center_y * img_width)),
                          radius=10, color=(0, 0, 255), thickness=-1)
    copy_image = cv2.rectangle(copy_image, start_point, end_point, (0, 255, 255), 3)

    return copy_image


def normalize_roi(image, center_x, center_y, width, height):
    img_height, img_width, _ = np.shape(image)
    norm_center_x = center_x/img_width
    norm_center_y = center_y/img_height
    norm_width = width/img_width
    norm_height = height/img_height

    return norm_center_x, norm_center_y, norm_width, norm_height


def save_rois(dict_rois, df, output_file_csv):
    new_df = df.copy()
    list_annotated_imgs = df['image_name'].tolist()
    for x in dict_rois:
        if x in list_annotated_imgs:
            index = list_annotated_imgs.index(x)

            new_row = df.iloc[index].copy()
            new_row['ROI'] = dict_rois[x]
            new_df.loc[index] = new_row

    new_df.to_csv(output_file_csv, index=False)


def crop_image(image, edges, output_dir, scale=True):
    # crop the image using array slicing
    copy_image = image.copy()
    if scale is True:
        cx, cy, width, height = normalize_roi(image, *edges)
    else:
        cx, cy, width, height = edges

    img_height, img_width, depth = np.shape(copy_image)
    w = int(width * img_width)
    h = int(height * img_height)
    # left corner
    center_x = int(cx * img_width)
    center_y = int(cy * img_height)
    left_x = center_x - int(w / 2)
    left_y = center_y - int(h / 2)
    right_x = center_x + int(w / 2)
    right_y = center_y + int(h / 2)
    left_x = np.max([0, left_x])
    left_y = np.max([0, left_y])
    cropped_img = copy_image[left_y:right_y, left_x:right_x]
    cv2.imwrite(output_dir, cropped_img)


def annotate_roi(path_video_file, extracted_frames_dir, output_dir=''):
    global imgs_rois
    global sc_width, sc_height
    global image_name
    global previous_coords
    previous_coords = None
    path_video_file = os.path.normpath(path_video_file)
    extracted_frames_dir = os.path.normpath(extracted_frames_dir)
    output_dir = os.path.normpath(output_dir)
    if 'annotations_all.csv' in os.listdir(output_dir):
        df_annotations = pd.read_csv(os.path.join(output_dir, 'annotations_all.csv'))
        output_file_csv = os.path.join(output_dir, 'annotations_all.csv')

    else:
        list_annotated_imgs = list()
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

        if i == len(list_imgs):
            save_rois(imgs_rois, df_annotations, output_file_csv)
            break
        image_name = list_imgs[i]
        img_dir = os.path.join(extracted_frames_dir, image_name)
        frame_number = list_frames[i]
        # read the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ret, image = cap.read()


        # make a copy of the image
        img_copy = copy.copy(image)
        if len(np.shape(img_copy)):
            if previous_coords:
                coords_frame = previous_coords
                cx, cy, nw, nd = coords_frame
                img_copy = draw_rectanlge(img_copy, cx, cy, nw, nd)
                cv2.imshow(image_name, img_copy)
            else:
                coords_frame = None

            print(f'img: {i+1}/{len(list_imgs)}')

            edge_detected, edges = detect_edge(img_copy)
            if edges:
                center_x, center_y, width, height = edges
                norm_center_x, norm_center_y, norm_width, norm_height = normalize_roi(img_copy, center_x, center_y, width, height)
                coords_frame = norm_center_x, norm_center_y, norm_width, norm_height
                img_width, img_height, _ = np.shape(image)
                img_copy = draw_rectanlge(img_copy, norm_center_x, norm_center_y, norm_width, norm_height)
                cv2.imshow(image_name, img_copy)

            if width_video >= sc_width or height_video >= sc_height:
                disp_img = image_resize(img_copy, width=int(0.7*sc_width))
            else:
                disp_img = img_copy

            cv2.imshow(image_name, disp_img)

            cv2.setMouseCallback(image_name, click_and_select_roi, image)
            new_key = cv2.waitKey(0) & 0xFF
            # wait for Esc or q key and then exit
            # previous frame

            # save the edge detected area and next!'
            if new_key == ord('s') and edges:
                center_x, center_y, width, height = edges
                imgs_rois[image_name] = list(normalize_roi(image, center_x, center_y, width, height))
                save_dir_img = os.path.join(output_dir, image_name)
                crop_image(image, edges, save_dir_img, scale=True)
                cv2.destroyWindow(image_name)
                i += 1

            # repeat frame
            if new_key == ord('d'):
                imgs_rois[image_name] = list()
                cv2.destroyWindow(image_name)

            # kill the program
            if new_key == ord('q'):
                cv2.destroyWindow(image_name)
                break

            # repeat the previous proposed area
            if new_key == ord('r'):
                if not imgs_rois[image_name]:
                    imgs_rois[image_name] = imgs_rois[list_imgs[i - 1]]
                save_dir_img = os.path.join(output_dir, image_name)
                imgs_rois[image_name] = list(imgs_rois[image_name])
                crop_image(image, imgs_rois[image_name], save_dir_img, scale=False)
                cv2.destroyWindow(image_name)
                previous_coords = imgs_rois[list_imgs[i - 1]]
                i += 1

            # exit and save the ROIs labeled until that point
            if new_key == 27 or new_key == ord("k"):
                save_rois(imgs_rois, df_annotations, output_file_csv)
                cv2.destroyAllWindows()
                break
        else:
            i += 1


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