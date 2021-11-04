from absl import app, flags, logging
from absl.flags import FLAGS
from general_functions import data_management as dam
from matplotlib import pyplot as plt
import cv2
import classification as img_class

flags.DEFINE_string('path_dir', '', 'path to directory')


def run_experiment(_argv):
    if FLAGS.path_dir:
        dam.analysze_dataset(FLAGS.path_dir)


if __name__ == '__main__':
    try:
        app.run(run_experiment)
    except SystemExit:
        pass