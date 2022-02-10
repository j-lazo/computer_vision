import os
from absl import app, flags, logging
from absl.flags import FLAGS
from general_functions import data_management as dam
from matplotlib import pyplot as plt
import cv2
from classification import call_models as img_class
from segmentation import call_models as segm

DATASETS = ['tissue_classification']


def run_experiment(_argv):
    project_folder = '/media/benoit/DATA/Jorge/temp_delete/dataset-20220209T103810Z-001/new_dataset/'
    name_models = ['Transpose_ResUnet']
    # Hyper-parameters:
    batches = [4]
    learing_rates = [1e-3]
    for name_model in name_models:
        for batch in batches:
            for lr in learing_rates:
                segm.call_model('train', project_folder, name_model, batch=batch, lr=lr)


def run_experiment_classification(_argv):
    #base_dir = os.path.normpath(os.getcwd())
    #path_list = base_dir.split(os.sep)
    # data_dir = os.path.join(*path_list[:-1], 'datasets/tissue_classification/')

    dataset = 'kvasir_image_classification'
    data_dir = ''.join([os.getcwd(), '/datasets/', dataset, '/'])
    test_data = data_dir + 'test/'
    results_dir = data_dir + 'results/'

    name_model = FLAGS.name_model
    mode = FLAGS.mode
    backbone_model = FLAGS.backbone
    batch_zie = FLAGS.batch_size
    epochs = FLAGS.epochs
    trainable_layers = FLAGS.trainable_layers
    fine_tune_epochs = FLAGS.fine_tune_epochs
    analyze_data = FLAGS.analyze_data

    if mode == 'train_backbone':
        learning_rate = [1e-5, 0.001]

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

    flags.DEFINE_string('name_model', 'fc_3layers', 'name of the model')
    flags.DEFINE_string('mode', 'train_backbone', 'train, predict, train_backbone')
    flags.DEFINE_string('backbone', 'resnet50', 'backbone network')
    flags.DEFINE_integer('batch_size', 8, 'batch size')
    flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
    flags.DEFINE_integer('epochs', 15, 'number of epochs')
    flags.DEFINE_integer('trainable_layers', -7, 'Trainable layers in case backbone is trained')
    flags.DEFINE_bool('analyze_data', True, 'select if analyze data or not')
    flags.DEFINE_integer('fine_tune_epochs', 6, 'epochs to fine tune the model')

    flags.DEFINE_string('dataset_dir', os.getcwd() + 'data/', 'path to dataset')
    flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
    flags.DEFINE_string('test_dataset', '', 'path to test dataset')
    flags.DEFINE_string('results_dir', os.getcwd() + 'results/', 'path to dataset')
    flags.DEFINE_string('weights', '', 'path to weights file')
    flags.DEFINE_string('directory_model', '', 'indicate the path to the directory')
    flags.DEFINE_float('validation_split', 0.2, 'iif not validation dir but needed')
    flags.DEFINE_string('file_to_predic', '', 'Directory or file where to perform predictions if predict mode selected')

    try:
        app.run(run_experiment)
    except SystemExit:
        pass