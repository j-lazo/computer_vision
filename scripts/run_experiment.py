import os
import argparse
from absl import app, flags, logging
from absl.flags import FLAGS
from general_functions import data_management as dam
from matplotlib import pyplot as plt
import cv2
from classification import call_models as img_class
from segmentation import call_models as segm
from classification import grad_cam as gc
from classification import call_models_tf as img_class_tf


DATASETS = ['tissue_classification', 'kvasir_image_classification', 'bladder_tissue_classification',
            'bladder_tissue_classification_v2', 'bladder_tissue_classification_v2_augmented',
            'bladder_tissue_classification_gan', 'bladder_tissue_classification_gan_augmented',
            'bladder_tissue_classification_gan_original', 'bladder_tissue_classification_gan_converted',
            'bladder_tissue_classification_gan_reconverted', 'bladder_tissue_classification_npz',
            'bladder_tissue_classification_npy']

flags.DEFINE_string('experiment_type', '', 'experiment type')
flags.DEFINE_string('name_model', 'gan_merge_features', 'name of the model: gan_merge_features, gan_merge_predictions_v1')
flags.DEFINE_string('mode', 'train_backbone', 'train, predict, train_backbone')
flags.DEFINE_string('backbone', 'resnet50', 'backbone network')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_integer('epochs', 15, 'number of epochs')
flags.DEFINE_integer('trainable_layers', -7, 'Trainable layers in case backbone is trained')
flags.DEFINE_bool('analyze_data', True, 'select if analyze data or not')
flags.DEFINE_bool('dropout', False, 'select if drop out included between the FC layers or not')
flags.DEFINE_integer('fine_tune_epochs', 5, 'epochs to fine tune the model')
flags.DEFINE_string('dataset_dir', os.getcwd() + 'data/', 'path to dataset')
flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
flags.DEFINE_string('test_dataset', '', 'path to test dataset')
flags.DEFINE_string('results_dir', os.getcwd() + 'results/', 'path to dataset')
flags.DEFINE_string('weights', '', 'path to weights file')
flags.DEFINE_string('directory_model', '', 'indicate the path to the directory')
flags.DEFINE_float('validation_split', 0.2, 'iif not validation dir but needed')
flags.DEFINE_string('after_concat', 'globalpooling', 'layer after concatenation: Global average Pooling or Flatten')
flags.DEFINE_string('file_to_predic', '', 'Directory or file where to perform predictions if predict mode selected')
flags.DEFINE_list('backbones', ['resnet50', 'resnet101'], 'A list of the nets used as backbones: resnet101, resnet50, densenet121, vgg19')

def grad_cam_experiment(_argv):

    #name_model = 'fc_3layers+resnet101_train_backbone_lr_1e-05_bs_16_16_02_2022_23_52'
    #dataset_dir = os.getcwd() + '/datasets/bladder_tissue_classification/'
    # dataset_to_analyze = dataset_dir + 'test/'
    name_model = FLAGS.name_model
    dataset_dir = FLAGS.dataset_dir
    dataset_to_analyze = FLAGS.test_dataset

    if dataset_dir in DATASETS:
        dataset_directory = ''.join([os.getcwd(), '/datasets/', dataset_dir, '/'])
    else:
        dataset_directory = dataset_dir

    gc.analyze_data_gradcam(name_model, dataset_directory, dataset_to_analyze, plot=False)


def run_segmentation_experiment(_argv):
    project_folder = ''
    name_models = ['Transpose_ResUnet']
    # Hyper-parameters:
    batches = [4]
    learing_rates = [1e-3]
    for name_model in name_models:
        for batch in batches:
            for lr in learing_rates:
                segm.call_model('train', project_folder, name_model, batch=batch, lr=lr)


def run_experiment_classification(_argv):

    name_model = FLAGS.name_model
    mode = FLAGS.mode
    backbone_model = FLAGS.backbone
    batch_zie = FLAGS.batch_size
    epochs = FLAGS.epochs
    trainable_layers = FLAGS.trainable_layers
    fine_tune_epochs = FLAGS.fine_tune_epochs
    analyze_data = FLAGS.analyze_data
    dataset_dir = FLAGS.dataset_dir

    if dataset_dir in DATASETS:
        data_dir = ''.join([os.getcwd(), '/datasets/', dataset_dir, '/'])
        test_data = data_dir + 'test/'
        results_dir = data_dir + 'results/'
    else:
        data_dir = dataset_dir
        test_data = FLAGS.test_dataset

    if mode == 'train_backbone':
        learning_rate = [1e-5, 1e-5]

        print('EXPERIMENT INFORMATION:', name_model, backbone_model, mode)
        img_class.call_models(name_model, mode, data_dir=data_dir,
                              backbone_model=backbone_model, batch_size=batch_zie,
                              epochs=epochs, test_data=test_data,
                              analyze_data=analyze_data,
                              learning_rate=learning_rate,
                              trainable_layers=trainable_layers,
                              results_dir=results_dir,
                              fine_tune_epochs=fine_tune_epochs)

    elif mode == 'train':
        learning_rate = FLAGS.learning_rate

        print('EXPERIMENT INFORMATION:', name_model, backbone_model, mode)
        img_class.call_models(name_model, mode, data_dir=data_dir,
                              backbone_model=backbone_model, batch_size=batch_zie,
                              epochs=epochs, test_data=test_data,
                              analyze_data=analyze_data,
                              learning_rate=learning_rate,
                              trainable_layers=trainable_layers,
                              results_dir=results_dir,
                              fine_tune_epochs=fine_tune_epochs)


def run_inference_classification(_argv):

    directory_model = FLAGS.directory_model
    file_to_predic = FLAGS.file_to_predic
    mode = 'predict'

    print('EXPERIMENT INFORMATION:', mode, file_to_predic)
    img_class.call_models('', mode, directory_model=directory_model, file_to_predic=file_to_predic)


def run_experiment_classification_tf(_arv):
    mode = FLAGS.mode
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    dataset_dir = FLAGS.dataset_dir
    learning_rate = FLAGS.learning_rate
    dropout = FLAGS.dropout
    backbones = FLAGS.backbones
    after_concat = FLAGS.after_concat
    name_model = FLAGS.name_model
    print(backbones)
    if dataset_dir in DATASETS:
        data_dir = ''.join([os.getcwd(), '/datasets/', dataset_dir, '/'])
        test_data = data_dir + 'test/'
        results_dir = data_dir + 'results/'
    else:
        data_dir = dataset_dir
        test_data = FLAGS.test_dataset
        results_dir = ''

    if mode == 'fit':
        img_class_tf.fit_model(name_model, data_dir, epochs=epochs, results_dir=results_dir,
                               learning_rate=learning_rate, batch_size=batch_size,
                               dropout=dropout,
                               backbones=backbones,
                               after_concat=after_concat)


def main(_argv):
    FUNCTION_MAP = {'grad_cam': grad_cam_experiment,
                    'classification': run_experiment_classification,
                    'segmentation': run_segmentation_experiment,
                    'inference_classification': run_inference_classification,
                    'classification_tf': run_experiment_classification_tf}
    func = FUNCTION_MAP[FLAGS.experiment_type]
    try:
        app.run(func)
    except SystemExit:
        pass


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass