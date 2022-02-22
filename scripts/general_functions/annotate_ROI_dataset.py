import argparse
import cv2

coords = []
drawing = False
drawing_lines = True
import os
import pandas as pd


def click_and_select_roi(event, x, y, flag, image):
    """
    Callback function, called by OpenCV when the user interacts
    with the window using the mouse. This function will be called
    repeatedly as the user interacts.
    (https://www.timpoulsen.com/2018/handling-mouse-events-in-opencv.html)
    """
    # get access to a couple of global variables we'll need
    global coords, drawing, drawing_lines, image_name, imgs_rois
    if imgs_rois[image_name]:
        cords_frame = imgs_rois[image_name]
        clone = image.copy()
        cv2.rectangle(clone, cords_frame[0], cords_frame[1], (0, 255, 0), 2)
        cv2.imshow(image_name, clone)
        if event == cv2.EVENT_RBUTTONDOWN:
            imgs_rois[image_name] = []
            cv2.imshow(image_name, image)
    else:
        clone = image.copy()
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
            cv2.line(clone, (x, 0), (x, 511), (0, 255, 0), 2)
            cv2.line(clone, (0, y), (511, y), (0, 255, 0), 2)
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



def UI_annotate_roi(dir_dataset, output_dir=''):
    """
    Creates a dictionary of the ROI of the images in a dataset
    Parameters
    ----------
    dir_dataset : (str) Directory of the image dataset

    Returns
    -------

    """
    global imgs_rois
    list_imgs = [f for f in os.listdir(dir_dataset) if f.endswith('.png')]
    imgs_rois = dict.fromkeys(list_imgs, [])
    # wait for Esc or q key and then exit
    i = 0
    total_frames = len(list_imgs)
    while True:
        if i < 0:
            i = 0
        if i > total_frames - 1:
            i = total_frames - 1

        global image_name
        image_name = list_imgs[i]
        img_dir = dir_dataset + image_name
        image = cv2.imread(img_dir)

        # show the captured image in a window
        cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
        if imgs_rois[image_name]:
            coords_frame = imgs_rois[image_name]
            clone = image.copy()
            cv2.rectangle(clone, coords_frame[0], coords_frame[1], (0, 255, 0), 2)
            cv2.imshow(image_name, clone)
        else:
            cv2.imshow(image_name, image)
        cv2.setMouseCallback(image_name, click_and_select_roi, image)
        new_key = cv2.waitKey(0) & 0xFF
        # wait for Esc or q key and then exit
        # previous frame
        if new_key == ord('a'):
            cv2.destroyWindow(image_name)
            i -= 1

        # next frame
        if new_key == ord('s'):
            cv2.destroyWindow(image_name)
            i += 1

        if new_key == 27 or new_key == ord("k"):
            print(imgs_rois)
            pd.DataFrame.from_dict(imgs_rois)
            #out_put_file_name = '.json'
            #pd.to_json(out_put_file_name)
            cv2.destroyAllWindows()
            break


def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--path_dataset", help="Path to the dataset")
    args = vars(ap.parse_args())
    dir_dataset = args["path_dataset"] if args["path_dataset"] else 0
    # now let-s label the ROI of the dataset
    UI_annotate_roi(dir_dataset)


if __name__ == "__main__":
    main()