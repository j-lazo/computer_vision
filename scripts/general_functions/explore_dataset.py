import os
import cv2
import pandas as pd
from absl import app, flags, logging
from absl.flags import FLAGS
import data_management as dam


coords = []
drawing = False

def do_nothing(emp):
    # this function does nothing :P
    pass


def explore_dataset(dataset_dir, annotations_dir=''):

    list_imgs = sorted(os.listdir(dataset_dir))
    data_frame = pd.read_csv(annotations_dir)
    list_names_imgs = data_frame['image_name'].tolist()
    dictionary = data_frame.set_index('image_name').to_dict()
    print(data_frame)
    i = 0
    while i < len(list_imgs): #for i, image in enumerate(list_imgs):
        image = list_imgs[i]
        if image in list_names_imgs:
            index = list_names_imgs.index(image)
            print(f'{i}/{len(list_imgs)}', data_frame.at[index, 'image_name'], data_frame.at[index, 'imaging type'])
            img_dir = dataset_dir + image
            print(img_dir)
            img = cv2.imread(img_dir)
            while True:
                key = cv2.waitKey(1) & 0xFF
                cv2.imshow(image, img)

                # (right arrow)
                if key == 83:
                    data_frame.loc[index, 'imaging type'] = 'WLI'
                    print(image, data_frame.at[index, 'image_name'], data_frame.at[index, 'imaging type'])
                    i += 1
                    break
                # (left arrow)
                if key == 81:
                    data_frame.loc[index, 'imaging type'] = 'NBI'
                    print(image, data_frame.at[index, 'image_name'], data_frame.at[index, 'imaging type'])
                    i += 1
                    break
                # (s) stop the video in that frame
                if key == ord('r'):
                    i -= 1
                    break
            cv2.destroyAllWindows()

    data_frame.to_csv(annotations_dir)


def run_program(_argv):
    dataset_dir = FLAGS.dataset_dir
    annotations_dir = FLAGS.annotations_dir
    explore_dataset(dataset_dir, annotations_dir=annotations_dir)


if __name__ == '__main__':
    flags.DEFINE_string('dataset_dir', '', 'path of the video')
    flags.DEFINE_string('annotations_dir', '', 'path where to save the images extracted')

    try:
        app.run(run_program)
    except SystemExit:
        pass

