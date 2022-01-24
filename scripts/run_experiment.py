import os
from absl import app, flags, logging
from absl.flags import FLAGS
from general_functions import data_management as dam
from matplotlib import pyplot as plt
import cv2
from classification import call_models as img_class
flags.DEFINE_string('data_dir', '', 'path to directory')


def run_experiment(_argv):

    dataset = 'tissue_classification'
    name_model = 'fc_3layers'
    backbone_model = 'inception_v3'
    mode = 'train_backbone'
    batch_zie = 8
    learning_rate = 0.001
    epochs = 15
    base_dir = os.path.normpath(os.getcwd())
    path_list = base_dir.split(os.sep)
    data_dir = ''.join([os.getcwd(), '/datasets/', dataset, '/'])
    #data_dir = os.path.join(*path_list[:-1], 'datasets/tissue_classification/')
    test_data = data_dir + 'test/'
    results_dir = data_dir + 'results/'
    analyze_data = True
    trainable_layers = - 7
    fine_tune_epochs = 5

    if mode == 'train_backbone':
        learning_rate = [1e-6, 0.001]

    print('EXPERIMENT INFORMATION:', name_model, backbone_model, mode)
    img_class.call_models(name_model, mode, data_dir=data_dir,
                          backbone_model=backbone_model, batch_size=batch_zie,
                          epochs=epochs, test_data=test_data,
                          analyze_data=analyze_data,
                          learning_rate=learning_rate,
                          trainable_layers=trainable_layers,
                          results_dir=results_dir,
                          fine_tune_epochs=fine_tune_epochs)


if __name__ == '__main__':
    try:
        app.run(run_experiment)
    except SystemExit:
        pass