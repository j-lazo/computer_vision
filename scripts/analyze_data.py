import os
from absl import app, flags, logging
from absl.flags import FLAGS
from general_functions import data_management as dam
from general_functions import data_management as daa
from matplotlib import pyplot as plt
import cv2
import classification as imgclass
from segmentation import call_models as imseg


flags.DEFINE_string('path_dir', '', 'path to directory')


def run_experiment(_argv):

    if FLAGS.path_dir:
        pass

if __name__ == '__main__':
    try:
        app.run(run_experiment)
    except SystemExit:
        pass