import os
from absl import app, flags, logging
from absl.flags import FLAGS
from general_functions import data_management as dam
from general_functions import data_analysis as daa
from matplotlib import pyplot as plt
import pandas as pd
import cv2
#import classification as imgclass
#from segmentation import call_models as imseg


flags.DEFINE_string('path_dir', '', 'path to directory')
flags.DEFINE_string('csv_dir', '', 'path to directory')

def run_experiment(_argv):

    if FLAGS.path_dir:
        path_dir = FLAGS.path_dir
        csv_file_dir = FLAGS.csv_dir
    #    list_subdirs = os.listdir(path_dir)
    #    list_csv_fiels = list()
    #    for sub_dir in list_subdirs:
    #        csv_name = [f for f in os.listdir(os.path.join(path_dir, sub_dir)) if f.endswith('.csv')].pop()
    #        dir_csv_files = ''.join([path_dir, sub_dir, '/', csv_name])
    #        list_csv_fiels.append(dir_csv_files)

    #    print(list_csv_fiels)
        daa.analyze_multiclass_experiment(csv_file_dir, path_dir)


if __name__ == '__main__':
    try:
        app.run(run_experiment)
    except SystemExit:
        pass