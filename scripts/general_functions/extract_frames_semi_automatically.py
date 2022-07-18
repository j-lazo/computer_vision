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

def do_nothing(emp):
    # this function does nothing :P
    pass


def click_and_crop(event, x, y, flag, user_data):
    image = user_data[0]
    string = user_data[1]
    print(string)

    """
    Callback function, called by OpenCV when the user interacts
    with the window using the mouse. This function will be called
    repeatedly as the user interacts.
    """
    # get access to a couple of global variables we'll need
    global coords, drawing, final_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        # user has clicked the mouse's left button
        drawing = True
        # save those starting coordinates
        coords = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        # user is moving the mouse within the window
        if drawing is True:
            # if we're in drawing mode, we'll draw a green rectangle
            # from the starting x,y coords to our current coords
            clone = image.copy()
            cv2.rectangle(clone, coords[0], (x, y), (0, 255, 255), 2)
            cv2.imshow(string, clone)
    elif event == cv2.EVENT_LBUTTONUP:
        # user has released the mouse button, leave drawing mode
        # and crop the photo
        drawing = False
        # save our ending coordinates
        coords.append((x, y))
        if len(coords) == 2:
            # calculate the four corners of our region of interest
            ty, by, tx, bx = coords[0][1], coords[1][1], coords[0][0], coords[1][0]
            # crop the image using array slicing
            roi = image[ty:by, tx:bx]
            height, width = roi.shape[:2]
            if width > 0 and height > 0:
                while True:
                    second_key = cv2.waitKey(1) & 0xFF
                    if second_key == ord('q'):
                        cv2.imshow(string, image)
                        break
                    elif second_key == ord('k'):
                        print('oki doki')
                        final_coords = coords
                        break



def extract_frames_video(video_path, save_dir='', target_size=(350, 350)):
    """
    Given a video can extract selected frames. To advance press (->) right arrow,
    to return to the previous frame, (<-) left arrow. If you want to extract the current
    frame from the video press (s). Then select the ROI with he mouse (green square).
    It then will show the cropped image in an extra window. Press q to close the window.
    If you want to save the cropped image press (s) otherwise press any other key.
    You can keep drawing the ROI untill you have the ROI you want.
    Once you select the ROI for the first time it will show it as default in the
    following selected frames but you still can change it.

    Parameters
    ----------
    video_path : (str)
    save_dir : (str)
    target_size : (tuple)

    Returns
    -------

    """
    if not(os.path.isfile(video_path)):
        raise f'Video path:{video_path} not found'

    name_video = os.path.split(video_path)[-1]
    file_extension = name_video.split('.')[-1]
    case_id = name_video.replace('.' + file_extension, '')

    if save_dir == '':
        save_dir = os.path.join(os.getcwd(), case_id)
    else:
        save_dir = save_dir + case_id

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    cv2.namedWindow(name_video, 0)
    if width > 750:
        cv2.resizeWindow(name_video, int(width/3), int(height/3))
    else:
        cv2.resizeWindow(name_video, 600, 600)
    # get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    loop_flag = 0
    pos = 0
    cv2.createTrackbar('time', name_video, 0, total_frames, do_nothing)

    ret, img = cap.read()
    while cap.isOpened():
        key = cv2.waitKey(1) & 0xFF

        if loop_flag < 0:
            loop_flag = 0

        if loop_flag == total_frames:
            break

        if loop_flag == pos:
            # here we update the position of the bar
            cv2.setTrackbarPos('time', name_video, loop_flag)
        else:
            # here we read the position of the Tracker Bar
            pos = cv2.getTrackbarPos('time', name_video)
            loop_flag = pos

        #ret, img = cap.read()
        cv2.imshow(name_video, img)

        # (right arrow)
        if key == 83:
            ret, img = cap.read()
            loop_flag = loop_flag + 1
            cv2.setTrackbarPos('time', name_video, loop_flag)

        # (left arrow)
        if key == 81:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, img = cap.read()
            loop_flag = loop_flag - 1
            cv2.setTrackbarPos('time', name_video, loop_flag)

        # (s) stop the video in that frame
        if key == ord('s'):

            selected_frame = pos
            name_frame = 'frame_' + str(selected_frame).zfill(4)
            name_window = 'Selected Frame'

            cv2.namedWindow(name_window, 2)
            if width > 750:
                cv2.resizeWindow(name_window, int(width / 2), int(height / 2))

            else:
                cv2.resizeWindow(name_window, 600, 600)

            selected_frame = copy.copy(img)
            if coords:
                cv2.rectangle(selected_frame, coords[0], coords[1], (255, 0, 0), 2)

            cv2.imshow(name_window, selected_frame)
            cv2.setMouseCallback(name_window, click_and_crop, img)
            new_key = cv2.waitKey(0) & 0xFF

            if new_key == ord('s'):
                # save the current frame
                filename = ''.join([save_dir, '/', case_id, '_', name_frame, '.jpg'])
                if len(coords) == 2:
                    # calculate the four corners of our region of interest
                    ty, by, tx, bx = coords[0][1], coords[1][1], coords[0][0], coords[1][0]
                    # crop the image using array slicing
                    roi = img[ty:by, tx:bx]
                    reshape_img = cv2.resize(roi, target_size)
                    cv2.imwrite(filename, reshape_img)
                    cv2.destroyWindow(name_window)
            else:
                # Ignore and continue displaying video
                cv2.destroyWindow(name_window)

        # (q) quit
        if key == ord('q') or loop_flag == total_frames:
            break

    cap.release()
    cv2.destroyAllWindows()


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
            cv2.circle(image_rgb, center, radius, (255, 0, 255), 3)
            if np.shape(circles)[1] == 1:
                # draw a rectangle
                # represents the top left corner of rectangle
                left_x = int(center[0]) - radius
                left_y = int(center[1]) - radius
                if left_x-5 > 0:
                    left_x = left_x - 5
                else:
                    left_x = 0
                if left_y-5 > 0:
                    left_y = left_y - 5
                else:
                    left_y = 0
                start_point = (left_x, left_y)
                # represents the bottom right corner of rectangle
                right_x = int(center[0] + radius)
                right_y = int(center[1] + radius)
                if right_x+5 < w:
                    right_x = right_x + 20
                else:
                    right_x = w
                if right_y+5 < h:
                    right_y = right_y + 20
                else:
                    right_y = h
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


def extract_frames_video_auto(path_video_file, extracted_frames_dir, output_dir=os.getcwd()):
    final_coords = list()
    sc_width, sc_height = pyautogui.size()
    print(sc_width, sc_height)
    path_video_file = os.path.normpath(path_video_file)
    extracted_frames_dir = os.path.normpath(extracted_frames_dir)
    output_dir = os.path.normpath(output_dir)

    if not (os.path.isfile(path_video_file)):
        raise f'Video path:{path_video_file} not found'

    image_list = sorted([f for f in os.listdir(extracted_frames_dir)
                  if os.path.isfile(os.path.join(extracted_frames_dir, f))])

    list_frames = [name.split('frame_')[-1].replace('.png', '') for name in image_list]
    name_video = os.path.split(path_video_file)[-1]
    file_extension = name_video.split('.')[-1]
    case_id = name_video.replace('.' + file_extension, '')
    cap = cv2.VideoCapture(path_video_file)
    for j, frame_number in enumerate(list_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number) - 1)
        image_path = os.path.join(extracted_frames_dir, image_list[j])
        ret, img = cap.read()
        original_img = copy.copy(img)
        copy_copy = copy.copy(img)
        image = cv2.imread(image_path)
        edge_detected, edges = detect_edge(img)
        keep_reading = True
        if edges:
            img_width, img_height, depth = np.shape(img)

            center_x, center_y, width, height = edges
            original_img = cv2.circle(original_img, (int(center_x*img_height), int(center_y*img_width)),
                                      radius=10, color=(0, 0, 255), thickness=-1)
            start_point = (int((center_x - (width/2))*img_height), int((center_y - (height/2))*img_width))
            # bottom right corner
            end_point = int((center_x + (width/2))*img_height), int((center_y + (height/2))*img_width)
            print(start_point, end_point)
            disp_img = cv2.rectangle(original_img, start_point, end_point, (0, 255, 0), 2)

        if img_width >= sc_width or img_height >= sc_height:
            disp_img = image_resize(original_img, width=int(0.7*sc_width))

        while keep_reading is True:
            key = cv2.waitKey(1) & 0xFF
            name_frame = frame_number

            cv2.imshow(name_frame, disp_img)
            frame_name = name_frame
            cv2.setMouseCallback(name_frame, click_and_crop, [copy_copy, frame_name])
            if len(final_coords) == 2:
                print(final_coords)
                break
                #cv2.rectangle(copy_copy, coords[0], coords[1], (255, 0, 0), 2)
                #cv2.destroyWindow(name_frame)
                #cv2.imshow(name_frame, copy_copy)

            # (s) stop the video in that frame
            if key == ord('k'):
                cv2.destroyWindow(name_frame)
            # (q) quit
            if key == ord('q') or len(list_frames) == j:
                cv2.destroyWindow(name_frame)
                break


def main(_argv):
    video_path = FLAGS.video_path
    save_dir = FLAGS.save_dir
    extracted_frames = FLAGS.extracted_frames_dir
    extract_frames_video_auto(video_path, extracted_frames, output_dir=save_dir)


if __name__ == '__main__':
    flags.DEFINE_string('video_path', '', 'path of the video')
    flags.DEFINE_string('extracted_frames_dir', '', 'path where frames were previously extracted')
    flags.DEFINE_string('save_dir', '', 'path where to save the images extracted')

    try:
        app.run(main)
    except SystemExit:
        pass

