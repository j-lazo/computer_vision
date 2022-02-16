import copy
import os
import cv2
from absl import app, flags, logging
from absl.flags import FLAGS

coords = []
drawing = False

def do_nothing(emp):
    # this function does nothing :P
    pass


def click_and_crop(event, x, y, flag, image):
    """
    Callback function, called by OpenCV when the user interacts
    with the window using the mouse. This function will be called
    repeatedly as the user interacts.
    """
    # get access to a couple of global variables we'll need
    global coords, drawing
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
            cv2.rectangle(clone, coords[0], (x, y), (0, 255, 0), 2)
            cv2.imshow('Selected Frame', clone)
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
                # make sure roi has height/width to prevent imshow error
                # and show the cropped image in a new window
                cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
                cv2.imshow("ROI", roi)
                while True:
                    second_key = cv2.waitKey(1) & 0xFF
                    if second_key == ord('q'):
                        cv2.destroyWindow('ROI')
                        break


def extract_frames_video(video_path, save_dir='', target_size=(350, 350)):

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
    print(width)
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
                filename = ''.join([save_dir, '/', case_id, '_', name_frame, '.png'])
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


def run_program(_argv):
    video_path = FLAGS.video_path
    save_dir = FLAGS.save_dir
    extract_frames_video(video_path, save_dir=save_dir)


if __name__ == '__main__':
    flags.DEFINE_string('video_path', '', 'path of the video')
    flags.DEFINE_string('save_dir', '', 'path where to save the images extracted')

    try:
        app.run(run_program)
    except SystemExit:
        pass

