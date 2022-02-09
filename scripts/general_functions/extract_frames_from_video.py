import os
import cv2
from matplotlib import pyplot as plt
from matplotlib.widgets import RectangleSelector


def do_nothing(emp):
    # this function does nothing :P
    pass


def extract_frames_video(video_path, save_dir='', target_size=(350, 350)):

    name_video = os.path.split(video_path)[-1]
    file_extension = name_video.split('.')[-1]
    case_id = name_video.replace('.' + file_extension, '')

    if save_dir == '':
        save_dir = os.path.join(os.getcwd(), case_id)

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
    cap = cv2.VideoCapture(video_path)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    cv2.namedWindow(name_video, 0)
    cv2.resizeWindow(name_video, int(width/3), int(height/3))
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

            cv2.namedWindow(name_frame, 2)
            cv2.resizeWindow(name_frame, int(width/2), int(height/2))
            cv2.imshow(name_frame, img)

            new_key = cv2.waitKey(0) & 0xFF

            if new_key == ord('s'):
                # save the current frame
                filename = ''.join([save_dir, '/', case_id, '_', name_frame, '.png'])
                reshape_img = cv2.resize(img, target_size)
                cv2.imwrite(filename, reshape_img)
                cv2.destroyWindow(name_frame)
            else:
                # Ignore and continue displaying video
                cv2.destroyWindow(name_frame)

        # (q) quit
        if key == ord('q') or loop_flag == total_frames:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = "case_004_pt_001.mp4"
    extract_frames_video(video_path)