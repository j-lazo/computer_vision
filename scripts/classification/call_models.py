
import numpy as np
import tensorflow as tf
import os
import sys
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import datetime
import shutil
import glob

from absl import app, flags, logging
from absl.flags import FLAGS

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Nadam
from tensorflow.keras import applications
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard

from sklearn.metrics import roc_curve, auc
from tensorflow import keras
sys.path.append(os.getcwd() + '/scripts/general_functions/')

import data_management as dam
import data_analysis as daa
from classification import classification_models as cms

class DataGenerator(tf.keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, list_IDs, labels, batch_size=32, dim=(64, 64), n_channels=1,
                 n_classes=10, shuffle=True):
        # Initialization
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_IDs_temp)

        return x, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.empty((self.batch_size, self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            if ID.endswith('.csv'):
                x[i,] = np.load(ID)
            elif ID.endswith('.png') or ID.endswith('.jpg') or ID.endswith('.jpeg'):
                img = cv2.imread(ID)/255
                reshaped = cv2.resize(img, self.dim)
                x[i,] = reshaped

            # Store class
            y[i] = self.labels[i]

        return x, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)


def make_predictions(model, innput_frame, output_size=(300, 300)):
    # get the input size of the network
    input_layer_shape = model.layers[0].input_shape
    shape_input = np.shape(input_layer_shape)
    if shape_input[-1] == 4:
        input_size_x = input_layer_shape[0][1]
        input_size_y = input_layer_shape[0][2]

    # reshape the input frame to be compatible with the input of the network
    reshaped_img = cv2.resize(innput_frame, (input_size_x, input_size_y),
                              interpolation=cv2.INTER_AREA)
    # apply blur to the image and normalize the image
    resized = (cv2.blur(reshaped_img, (7, 7)))/255
    # make a prediction of the mask
    mask = cvf.predict_mask(model, resized)
    # resize the image for a size to show to the user
    output_imgage = cv2.resize(reshaped_img, output_size, interpolation=cv2.INTER_AREA)
    w, h, d = np.shape(output_imgage)
    # calculate the center points of the lumen according to the detected mask
    point_x, point_y = cvf.detect_dark_region(mask, output_imgage)

    return output_imgage, point_x, point_y


def generate_experiment_ID(name_model='', learning_rate='na', batch_size='na', backbone_model='',
                           prediction_model='', mode=''):
    """
    Generate a ID name for the experiment considering the name of the model, the learning rate,
    the batch size, and the date of the experiment

    :param name_model: (str)
    :param learning_rate: (float)
    :param batch_size: (int)
    :param backbone_model: (str)
    :return: (str) id name
    """
    if type(learning_rate) == list:
        lr = learning_rate[0]
    else:
        lr = learning_rate

    if prediction_model == '':
        training_date_time = datetime.datetime.now()
        if backbone_model != '':
            name_mod = ''.join([name_model, '+', backbone_model])
        else:
            name_mod = name_model
        id_name = ''.join([name_mod, '_', mode, '_lr_', str(lr),
                                  '_bs_', str(batch_size), '_',
                                  training_date_time.strftime("%d_%m_%Y_%H_%M")
                                  ])
    else:
        predictions_date_time = datetime.datetime.now()
        id_name = ''.join([prediction_model, '_predictions_', predictions_date_time.strftime("%d_%m_%Y_%H_%M")])

    return id_name


def load_pretrained_model(name_model, weights='imagenet', include_top=False, trainable=False):
    base_dir_weights = ''.join([os.getcwd(), '/scripts/classification/weights_pretrained_models/'])
    """
    Loads a pretrained model given a name
    :param name_model: (str) name of the model
    :param weights: (str) weights names (default imagenet)
    :return: sequential model with the selected weights
    """
    if name_model == 'vgg16':
        weights_dir = base_dir_weights + 'vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.vgg16.VGG16(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable
        input_size = (224, 224, 3)

    elif name_model == 'vgg19':
        weights_dir = base_dir_weights + 'vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.vgg19.VGG19(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable
        input_size = (224, 224, 3)

    elif name_model == 'inception_v3':
        weights_dir = base_dir_weights + 'inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.inception_v3.InceptionV3(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable
        input_size = (299, 299, 3)

    elif name_model == 'resnet50':
        weights_dir = base_dir_weights + 'resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.resnet50.ResNet50(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable
        input_size = (224, 224, 3)

    elif name_model == 'resnet101':
        weights_dir = base_dir_weights + 'resnet101/resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.resnet.ResNet101(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable
        input_size = (224, 224, 3)

    elif name_model == 'mobilenet':
        weights_dir = base_dir_weights + 'mobilenet/mobilenet_1_0_224_tf_no_top.h5'
        base_model = applications.mobilenet.MobileNet(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable
        input_size = (224, 224, 3)

    elif name_model == 'densenet':
        weights_dir = base_dir_weights + 'densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.densenet.DenseNet121(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable
        input_size = (224, 224, 3)

    elif name_model == 'xception':
        weights_dir = base_dir_weights + 'xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.xception.Xception(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable
        input_size = (299, 299, 3)

    else:
        raise ValueError(f' MODEL: {name_model} not found')

    print('PRETRAINED Model')

    base_model.summary()
    return base_model, input_size


def load_cap_models(name_model, num_classes):

    if name_model == 'simple_fc':
        cap_model = cms.simple_fc(num_classes)
    elif name_model =='fc_3layers':
        cap_model = cms.fc_3layers(num_classes)
    else:
        print(f'mode {name_model} not found')

    return cap_model


def build_model(name_model, learning_rate, backbone_model='', num_classes=1,
                include_top=False, train_backbone=False, trainable_layers=-1,
                optimizer='adam', loss='categorical_crossentropy',
                metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]):

    # initialize model
    model = Sequential()

    # load the cap
    cap_model = load_cap_models(name_model, num_classes)

    if type(learning_rate) == list:
        lr = learning_rate[0]
    else:
        lr = learning_rate

    if train_backbone is True:
        # load the backbone
        base_model, input_shape_backbone = load_pretrained_model(backbone_model, include_top=include_top,
                                                                 trainable=True)

        for i, layer in enumerate(base_model.layers):
            if i in range(len(base_model.layers) + trainable_layers, len(base_model.layers)):
                layer.trainable = True
            else:
                layer.trainable = False
    else:
        base_model, input_shape_backbone = load_pretrained_model(backbone_model, include_top=include_top,
                                                                 trainable=False)
        base_model.trainable = False

    base_model.compile()
    base_model.summary()
    model.add(base_model)
    model.add(cap_model)

    if optimizer == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif optimizer == 'sgd':
        optimizer = SGD(learning_rate=lr, momentum=0.9)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()

    return model


def load_model(directory_model):
    if directory_model.endswith('.h5'):
        model_path = directory_model
    else:
        files_dir = [f for f in os.listdir(directory_model) if f.endswith('.h5')]
        if files_dir:
            model_path = files_dir.pop()
        else:
            files_dir = [f for f in os.listdir(directory_model) if f.endswith('.pb')]
            if files_dir:
                model_path = ''
                print(f'Tensorflow model found at {directory_model}')
            else:
                print(f'No model found in {directory_model}')

    print('MODEL USED:')
    print(model_path)
    #model = tf.keras.models.load_model(model_path, compile=False)
    print(f'Model path: {directory_model + model_path}')
    model = keras.models.load_model(directory_model + model_path)
    model.summary()
    input_size = (len(model.layers[0].output_shape[:]))

    return model, input_size


def fine_tune_backbone(model, training_generator, validation_generator, epochs,
                batch_size, results_directory, new_results_id, shuffle=1, verbose=1,
                       learning_rate=0.001, trainable_layers=-1, optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]):

    print('Fine-tuning model')
    temp_name_model = results_directory + new_results_id + "_fine_tuned_model.h5"
    callbacks = [
        ModelCheckpoint(temp_name_model,
                        monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', patience=25),
        CSVLogger(results_directory + 'fine_tune_train_history_' + new_results_id + "_.csv"),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]

    model.fit(training_generator,
              epochs=epochs,
              shuffle=shuffle,
              batch_size=batch_size,
              validation_data=validation_generator,
              verbose=verbose,
              callbacks=callbacks)

    if type(learning_rate) == list:
        lr = learning_rate[1]
    else:
        lr = learning_rate

    # Freeze the weights of the backbone
    model.layers[0].trainable = False
    # Now just train the cap-model
    model.layers[-1].trainable = True

    print('LR:', lr)
    if optimizer == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif optimizer == 'sgd':
        optimizer = SGD(learning_rate=lr, momentum=0.9)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def generate_dict_x_y(general_dict):

    dict_x = {}
    dict_y = {}
    unique_values = []
    for i, element in enumerate(general_dict):
        dict_x[i] = general_dict[element]['image_dir']
        if general_dict[element]['classification'] not in unique_values:
            unique_values.append(general_dict[element]['classification'])
        dict_y[i] = int(unique_values.index(general_dict[element]['classification']))

    return dict_x, dict_y, unique_values


def load_data(data_dir, annotations_file='', backbone_model='',
              img_size=(255, 255), batch_size=8, prediction_mode=False):
    # If using a pre-trained backbone model, then use the img data generator from the pretrained model
    if backbone_model != '':
        if backbone_model == 'vgg16':
            data_idg = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)
            img_width, img_height = 224, 224

        elif backbone_model == 'vgg19':
            data_idg = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg19.preprocess_input)
            img_width, img_height = 224, 224

        elif backbone_model == 'inception_v3':
            data_idg = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_v3.preprocess_input)
            img_width, img_height = 299, 299

        elif backbone_model == 'resnet50':
            data_idg = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
            img_width, img_height = 224, 224

        elif backbone_model == 'resnet101':
            data_idg = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet.preprocess_input)
            img_width, img_height = 224, 224

        elif backbone_model == 'mobilenet':
            data_idg = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)
            img_width, img_height = 224, 224

        elif backbone_model == 'densenet121':
            data_idg = ImageDataGenerator(preprocessing_function=tf.keras.applications.densenet.preprocess_input)
            img_width, img_height = 224, 224

        elif backbone_model == 'xception':
            data_idg = ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input)
            img_width, img_height = 299, 299
        else:
            raise ValueError(f' MODEL: {backbone_model} not found')

    else:
        pass

    if annotations_file == '':
        # determine if the structure of the directory is divided by classes or if there is an annotation file
        print(f'Loading data from dir: {data_dir}')
        files_dir = [f for f in os.listdir(data_dir) if os.path.isdir(data_dir+f)]
        num_classes = len(files_dir)
        # if the number of sub-folders is less than two then it supposes that
        # there is an annotation file in .csv format and looks for it
        if num_classes < 2:
            list_csv_files = [f for f in os.listdir(data_dir) if os.path.isdir(f)]
            if not list_csv_files:
                print('No annotation files found or directory with sub-classes found')
            else:
                csv_annotations_file = list_csv_files.pop()
                dataframe = pd.read_csv(data_dir + csv_annotations_file)

        else:
            if prediction_mode is True:
                subdirs = [f for f in os.listdir(data_dir) if os.path.isdir(data_dir + f)]
                total_all_imgs = 0
                for subdir in subdirs:
                    all_imgs = os.listdir(''.join([data_dir, subdir, '/']))
                    total_all_imgs = total_all_imgs + len(all_imgs)
                    data_generator = data_idg.flow_from_directory(data_dir,
                                                                  batch_size=15,
                                                                  class_mode='categorical',
                                                                  target_size=(img_width, img_height),
                                                                  shuffle=False)

            else:
                data_generator = data_idg.flow_from_directory(data_dir,
                                                          batch_size=batch_size,
                                                          class_mode='categorical',
                                                          target_size=(img_width, img_height))

            num_classes = len(data_generator.class_indices)
    else:
        # read the annotations from a csv file
        dataframe = pd.read_csv(data_dir + annotations_file)
        data_dictionary = dam.build_dictionary_data_labels(data_dir)
        # Parameters
        data, labels, num_classes = generate_dict_x_y(data_dictionary)
        params = {'dim': img_size,
                  'batch_size': 8,
                  'n_classes': num_classes,
                  'n_channels': 3,
                  'shuffle': True}
        data_generator = DataGenerator(data, labels, params, annotations_file)

    return data_generator, num_classes


def train_model(model, training_generator, validation_generator, epochs,
                batch_size, results_directory, new_results_id, shuffle=1, verbose=1):
    temp_name_model = results_directory + new_results_id + "_model.h5"
    callbacks = [
        ModelCheckpoint(temp_name_model,
                        monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', patience=25),
        CSVLogger(results_directory + 'train_history_' + new_results_id + "_.csv"),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]

    trained_model = model.fit(training_generator,
              epochs=epochs,
              shuffle=shuffle,
              batch_size=batch_size,
              validation_data=validation_generator,
              verbose=verbose,
              callbacks=callbacks)

    return trained_model


def evaluate_and_predict(model, directory_to_evaluate, results_directory,
                         output_name='', results_id='', backbone_model='', batch_size=1,
                         analyze_data=False, output_dir=''):
    print(f'Evaluation of: {directory_to_evaluate}')

    # load the data to evaluate and predict
    data_gen, _ = load_data(directory_to_evaluate, backbone_model=backbone_model,
                                  batch_size=13, prediction_mode=True)

    evaluation = model.evaluate(data_gen, verbose=True)
    print('Evaluation results:')
    print(evaluation)
    predictions = model.predict(data_gen, verbose=True)

    # determine the top-1 prediction class
    predicts = np.argmax(predictions, axis=1)

    x_p = x_p = [[] for _ in range(len(np.unique(predicts)))]
    for x in predictions:
        for i in range(len(np.unique(predicts))):
            x_p[i].append(x[i])

    label_index = {v: k for k, v in data_gen.class_indices.items()}
    predicts = [label_index[p] for p in predicts]
    header_column = ['class_' + str(i+1) for i in range(len(np.unique(predicts)))]
    header_column.insert(0, 'fname')
    header_column.append('over all')
    df = pd.DataFrame(columns=header_column)
    df['fname'] = [os.path.basename(x) for x in data_gen.filenames]

    for i in range(len(np.unique(predicts))):
        class_name = 'class_' + str(i+1)
        df[class_name] = x_p[i]

    df['over all'] = predicts
    # save the predictions  of each case
    results_csv_file = ''.join([results_directory, 'predictions_', output_name, '_', results_id, '_.csv'])
    df.to_csv(results_csv_file, index=False)

    if analyze_data is True:
        list_files = [f for f in os.listdir(directory_to_evaluate) if f.endswith('.csv')]
        if list_files:
            annotations_csv_data = list_files.pop()
            dir_annotations_csv = directory_to_evaluate + annotations_csv_data

            if len(np.unique(predicts)) == 2:
                auc = daa.calculate_auc_and_roc(results_csv_file, dir_annotations_csv, output_name, plot=False,
                                                results_directory=results_directory, results_id=results_id, save_plot=True)
                print(f'AUC: {auc}')
        else:
            print(f'No annotation file found in {directory_to_evaluate}')

    return results_csv_file


def evalute_test_directory(model, test_data, results_directory, new_results_id, backbone_model, analyze_data=True):

    # determine if there are sub_folders or if it's the absolute path of the dataset
    sub_dirs = [f for f in os.listdir(test_data) if os.path.isdir(test_data + f)]
    if sub_dirs:
        for sub_dir in sub_dirs:
            sub_sub_dirs = [f for f in os.listdir(test_data + sub_dir) if
                            os.path.isdir(''.join([test_data, sub_dir, '/', f]))]
            if sub_sub_dirs:
                print(f'Sub-directories:{sub_dirs} found in {test_data}')
                # this means that inside each sub-dir there is more directories so we can iterate over the previous one
                name_file = evaluate_and_predict(model, ''.join([test_data, sub_dir, '/']), results_directory,
                                                 results_id=new_results_id, output_name=sub_dir,
                                                 backbone_model=backbone_model, analyze_data=analyze_data)

                print(f'Evaluation results saved at {name_file}')
            else:
                name_file = evaluate_and_predict(model, test_data, results_directory,
                                                 results_id=new_results_id, output_name='test',
                                                 backbone_model=backbone_model, analyze_data=analyze_data)
                print(f'Evaluation results saved at {name_file}')
                break

    else:
        name_file = evaluate_and_predict(model, test_data, results_directory,
                                         results_id=new_results_id, output_name='test',
                                         backbone_model=backbone_model, analyze_data=analyze_data)
        print(f'Evaluation results saved at {name_file}')


def call_models(name_model, mode, data_dir=os.getcwd() + '/data/', validation_data_dir='',
                test_data='', results_dir=os.getcwd() + '/results/', epochs=2, batch_size=4, learning_rate=0.001,
                backbone_model='', eval_val_set=False, eval_train_set=False, analyze_data=False, directory_model='',
                file_to_predic='', trainable_layers=-1, fine_tune_epochs=1):


    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # Decide how to act according to the mode (train/predict/train-backbone... )
    if mode == 'train' or mode == 'train_backbone':
        # Determine what is the structure of the data directory,
        # if the directory contains train/val datasets
        if validation_data_dir == '':
            sub_dirs = os.listdir(data_dir)
            if 'train' in sub_dirs:
                train_data_dir = data_dir + 'train/'

            if 'val' in sub_dirs:
                validation_data_dir = data_dir + 'val/'
            else:
                print(f'substructure found in {data_dir} not recognized  , please indicate the validation(val) dataset')
        else:
            train_data_dir = data_dir

        # Define Generators
        training_generator, num_classes = load_data(train_data_dir, backbone_model=backbone_model,
                                                    batch_size=batch_size)
        validation_generator, num_classes = load_data(validation_data_dir, backbone_model=backbone_model,
                                                      batch_size=batch_size)

        # define a dir to save the results and Checkpoints
        # if results directory doesn't exist create it
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)

        # ID name for the folder and results
        new_results_id = generate_experiment_ID(name_model=name_model, learning_rate=learning_rate,
                                                batch_size=batch_size, backbone_model=backbone_model,
                                                mode=mode)

        results_directory = ''.join([results_dir, new_results_id, '/'])
        # if results experiment doesn't exists create it
        if not os.path.isdir(results_directory):
            os.mkdir(results_directory)

        # Build the model
        if mode == 'train_backbone':
            train_backbone = True
        else:
            train_backbone = False

        model = build_model(name_model, learning_rate, backbone_model, num_classes,
                            train_backbone=train_backbone, trainable_layers=trainable_layers)

        if train_backbone is True:

            model = fine_tune_backbone(model, training_generator, validation_generator, fine_tune_epochs,
                    batch_size, results_directory, new_results_id, trainable_layers=trainable_layers,
                                       learning_rate=learning_rate)

            if test_data != '':
                evalute_test_directory(model, test_data, results_directory, new_results_id + '(_pre)', backbone_model,
                                       analyze_data=True)

        # track time
        start_time = datetime.datetime.now()
        # Train the model

        trained_model = train_model(model, training_generator, validation_generator, epochs,
                    batch_size, results_directory, new_results_id)

        model.save(''.join([results_directory, 'model_', new_results_id]))

        print('Total Training TIME:', (datetime.datetime.now() - start_time))
        print('History Model Keys:')
        print(trained_model.history.keys())
        # in case evaluate val dataset is True
        if eval_val_set is True:
            evaluate_and_predict(model, validation_data_dir, results_directory,
                                 results_id=new_results_id, output_name='val',
                                 backbone_model=backbone_model)

        if eval_train_set is True:
            evaluate_and_predict(model, train_data_dir, results_directory,
                                 results_id=new_results_id, output_name='train',
                                 backbone_model=backbone_model)
        if test_data != '':
            evalute_test_directory(model, test_data, results_directory, new_results_id, backbone_model,
                                   analyze_data=True)

    elif mode == 'predict':
        model, _ = load_model(directory_model)
        backbone_model = model.get_layer(index=0).name
        print(f'Backbone identified: {backbone_model}')
        if daa.check_file_isvid(file_to_predic):
            pass
        elif os.path.isdir(file_to_predic):
            new_results_id = generate_experiment_ID(prediction_model=os.path.basename(os.path.normpath(directory_model)))
            results_directory = directory_model

            evalute_test_directory(model, file_to_predic, results_directory, new_results_id, backbone_model,
                                   analyze_data=True)

        elif file_to_predic == 'webcam':
            pass

        else:
            print(f'Format or dir {file_to_predic} not understood')

        pass

    else:
        print(f'Mode:{mode} not understood')


def main(_argv):

    name_model = FLAGS.name_model
    mode = FLAGS.mode
    backbone_model = FLAGS.backbone
    data_dir = os.path.join(FLAGS.dataset_dir, '')
    val_data = os.path.join(FLAGS.val_dataset, '')
    batch_zie = FLAGS.batch_size
    epochs = FLAGS.epochs
    test_data = os.path.join(FLAGS.test_dataset, '')
    analyze_data = FLAGS.analyze_data
    directory_model = FLAGS.directory_model
    file_to_predic = FLAGS.file_to_predic
    trainable_layers = FLAGS.trainable_layers
    if mode == 'predict':
        if directory_model == '':
            raise ValueError('No directory of the model indicated')
        else:
            if file_to_predic == '':
                raise ValueError('No test dataset, image or video indicated')
    if mode == 'train_backbone':
        if trainable_layers == -1:
            raise ValueError(f'Mode: {mode} selected, please indicate the layers of the backbone to train')

    """
    e.g: 
    call_models.py --name_model=simple_fc --backbone=VGG19 --mode=train --batch_size=4
    --dataset_dir=directory/to/train/data/ --batch_size=16  --epochs=5 
    """

    print('INFORMATION:', name_model, backbone_model, mode)
    call_models(name_model, mode, data_dir=data_dir, backbone_model=backbone_model,
                batch_size=batch_zie, epochs=epochs, test_data=test_data,
                analyze_data=analyze_data, directory_model=directory_model,
                file_to_predic=file_to_predic,
                trainable_layers=trainable_layers)


if __name__ == '__main__':

    flags.DEFINE_string('name_model', '', 'name of the model')
    flags.DEFINE_string('mode', '', 'train, predict, train_backbone')
    flags.DEFINE_string('backbone', '', 'backbone network')
    flags.DEFINE_string('dataset_dir', os.getcwd() + 'data/', 'path to dataset')
    flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
    flags.DEFINE_string('test_dataset', '', 'path to test dataset')
    flags.DEFINE_string('results_dir', os.getcwd() + 'results/', 'path to dataset')
    flags.DEFINE_integer('epochs', 1, 'number of epochs')
    flags.DEFINE_integer('batch_size', 4, 'batch size')
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    flags.DEFINE_string('weights', '', 'path to weights file')
    flags.DEFINE_bool('analyze_data', False, 'select if analyze data or not')
    flags.DEFINE_string('directory_model', '', 'indicate the path to the directory')
    flags.DEFINE_float('validation_split', 0.2, 'iif not validation dir but needed')
    flags.DEFINE_string('file_to_predic', '', 'Directory or file where to perform predictions if predict mode selected')
    flags.DEFINE_integer('trainable_layers', -1, 'Trainable layers in case backbone is trained')

    """
    flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                      'fit: model.fit, '
                      'eager_fit: model.fit(run_eagerly=True), '
                      'eager_tf: custom GradientTape')
    flags.DEFINE_enum('transfer', 'none',
                      ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                      'none: Training from scratch, '
                      'darknet: Transfer darknet, '
                      'no_output: Transfer all but output, '
                      'frozen: Transfer and freeze all, '
                      'fine_tune: Transfer all and freeze darknet only')
    flags.DEFINE_integer('size', '', 'image size')
    """

    try:
        app.run(main)
    except SystemExit:
        pass
