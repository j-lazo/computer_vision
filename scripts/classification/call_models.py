from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
import os
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Nadam
from tensorflow.keras import applications
from keras.preprocessing.image import ImageDataGenerator
sys.path.append(os.getcwd() + '/scripts/general_functions/')
import data_management as dam
import classification_models
import cv2

import pandas as pd
import csv
from matplotlib import pyplot as plt
import datetime
import shutil

from sklearn.metrics import roc_curve, auc

flags.DEFINE_string('name_model', '', 'name of the model')
flags.DEFINE_string('mode', '', 'train or predict')
flags.DEFINE_string('backbone', '', 'backbone network')
flags.DEFINE_string('train_dataset', os.getcwd() + 'data/', 'path to dataset')
flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
flags.DEFINE_string('test_dataset', '', 'path to test dataset')
flags.DEFINE_string('results_dir', os.getcwd() + 'results/', 'path to dataset')
flags.DEFINE_integer('epochs', 1, 'number of epochs')
flags.DEFINE_integer('batch_size', 4, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf', 'path to weights file')


"""flags.DEFINE_string('train_dataset', '', 'path to dataset')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf', 'path to weights file')
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
flags.DEFINE_integer('epochs', 2, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')
flags.DEFINE_boolean('multi_gpu', False, 'Use if wishing to train with more than 1 GPU.')"""

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
        x = np.empty((self.batch_size, * self.dim, self.n_channels))
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



def load_pretrained_model(name_model, weights='imagenet'):

    """
    Loads a pretrained model given a name
    :param name_model: (str) name of the model
    :param weights: (str) weights names (default imagenet)
    :return: sequential model with the selected weights
    """

    if name_model == 'VGG16':
        base_model = applications.vgg16.VGG16(include_top=False, weights=weights)
        base_model.trainable = False
        base_model.summary()

    elif name_model == 'VGG19':
        base_model = applications.vgg19.VGG19(include_top=False, weights=weights)
        base_model.trainable = False
        base_model.summary()

    elif name_model == 'InceptionV3':
        base_model = applications.inception_v3.InceptionV3(include_top=False, weights=weights)
        base_model.trainable = False
        base_model.summary()

    elif name_model == 'ResNet50':
        base_model = applications.resnet50.ResNet50(include_top=False, weights=weights)
        base_model.trainable = False
        base_model.summary()

    elif name_model == 'ResNet101':
        base_model = applications.resnet.ResNet101(include_top=False, weights=weights)
        base_model.trainable = False
        base_model.summary()

    elif name_model == 'MobileNet':
        base_model = applications.mobilenet.MobileNet(include_top=False, weights=weights)
        base_model.trainable = False
        base_model.summary()

    elif name_model == 'DenseNet121':
        base_model = applications.densenet.DenseNet121(include_top=False, weights=weights)
        base_model.trainable = False
        base_model.summary()

    elif name_model == 'Xception':
        base_model = applications.xception.Xception(include_top=False, weights=weights)
        base_model.trainable = False
        base_model.summary()

    return base_model


def load_model(name_model, backbone_model=''):
    print(backbone_model)
    model = Sequential()
    num_classes = 2
    if backbone_model != '':
        base_model = load_pretrained_model(backbone_model)
        cap_model = getattr(classification_models, name_model)(num_classes)
        model.add(base_model)
        model.add(cap_model)
    else:
        cap_model = getattr(classification_models, name_model)(num_classes)
        model = model.add(cap_model)
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


def load_data(data_dir, backbone_model):

    data_dictionary = dam.build_dictionary_data_labels(data_dir)
    # ------generators to feed the model----------------

    if backbone_model != '':

        list_files_path = []
        class_types = []
        unique_values = []

        for element in data_dictionary:
            list_files_path.append(data_dictionary[element]['image_dir'])
            if data_dictionary[element]['classification'] not in unique_values:
                unique_values.append(data_dictionary[element]['classification'])
            class_types.append(int(unique_values.index(data_dictionary[element]['classification'])))

        dataframe = pd.DataFrame({'filename':list_files_path, 'class': class_types})
        print(dataframe)
        print(dataframe)
        if backbone_model == 'VGG16':
            data_idg = ImageDataGenerator(preprocessing_function= tf.keras.applications.vgg16.preprocess_input)
            img_width, img_height = 224, 224

        elif backbone_model == 'VGG19':
            data_idg = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg19.preprocess_input)
            img_width, img_height = 224, 224

        elif backbone_model == 'InceptionV3':
            data_idg = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_v3.preprocess_input)
            img_width, img_height = 224, 224

        elif backbone_model == 'ResNet50':
            data_idg = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
            img_width, img_height = 224, 224

        elif backbone_model == 'ResNet101':
            data_idg = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet.preprocess_input)
            img_width, img_height = 224, 224

        elif backbone_model == 'MobileNet':
            data_idg = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)
            img_width, img_height = 224, 224

        elif backbone_model == 'DenseNet121':
            data_idg = ImageDataGenerator(preprocessing_function=tf.keras.applications.densenet.preprocess_input)
            img_width, img_height = 224, 224

        elif backbone_model == 'Xception':
            data_idg = ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input)
            img_width, img_height = 224, 224

        data_generator = data_idg.flow_from_dataframe(dataframe,
                                                  target_size=(img_width, img_height),
                                                  batch_size=20,
                                                  class_mode='binary')
        num_classes = len(data_generator.class_indices)

    else:
        # Parameters
        data, labels, num_classes = generate_dict_x_y(data_dictionary)
        img_width, img_height = 300, 300
        params = {'dim': (img_width, img_height),
                  'batch_size': 8,
                  'n_classes': num_classes,
                  'n_channels': 3,
                  'shuffle': True}
        data_generator = DataGenerator(data, labels, params)

    return data_generator, num_classes


def train_model(model):

    return model



def call_models(name_model, mode, train_data_dir=os.getcwd() + 'data/', validation_data_dir='',
                test_data='', results_dir=os.getcwd() + 'results/', epochs=2, batch_size=4, backbone_model=''):

    # load the data
    """train_data_dict = dam.build_dictionary_data_labels(train_data_dir)
    val_data_dict = dam.build_dictionary_data_labels(validation_data_dir)
    train_data, train_labels, train_classes = generate_dict_x_y(train_data_dict)
    val_data, val_labels, val_classes = generate_dict_x_y(val_data_dict)
    num_classes = max([len(train_classes), len(val_classes)])

    img_width, img_height = 224, 224
    # Parameters
    params = {'dim': (img_width, img_height),
              'batch_size': 8,
              'n_classes': num_classes,
              'n_channels': 3,
              'shuffle': True}

    # Generators
    training_generator = DataGenerator(train_data, train_labels, **params)
    validation_generator = DataGenerator(val_data, val_labels, **params)"""

    training_generator = load_data(train_data_dir, backbone_model)
    validation_generator = load_data(validation_data_dir, backbone_model)

    # load the model
    model = load_model(name_model, backbone_model)
    adam = Adam(lr=0.0001)
    sgd = SGD(lr=0.001, momentum=0.9)
    rms = 'rmsprop'
    metrics = ["accuracy", tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=metrics)
    #model.summary()

    # then decide how to act according to the mode
    if mode == 'train':

        """train_idg = ImageDataGenerator(preprocessing_function=preprocess_input)
        val_idg = ImageDataGenerator(preprocessing_function=preprocess_input)

        # ------generators to feed the model----------------

        train_gen = train_idg.flow_from_directory(train_data_dir,
                                                  target_size=(img_width, img_height),
                                                  batch_size=20)

        validation_gen = val_idg.flow_from_directory(validation_data_dir,
                                                     target_size=(img_width, img_height),
                                                     batch_size=20)"""
        # train the model
        model.fit(training_generator,
                  epochs=epochs,
                  shuffle=1,
                  batch_size=batch_size,
                  validation_data=validation_generator,
                  validation_batch_size=batch_size,
                  verbose=1)
    elif mode == 'predict':
        test_idg = ImageDataGenerator(preprocessing_function=preprocess_input)
        test_gen = test_idg.flow_from_directory(validation_data_dir,
                                                     target_size=(img_width, img_height),
                                                     batch_size=50)

        evaluation = model.evaluate(test_gen, verbose=True, steps=10)
        predictions = model.predict(test_gen, verbose=True, steps=1)

        # top print the names in each batch of the generator
        # for i in test_gen[0]:
        #    idx = (test_gen.batch_index - 1) * test_gen.batch_size
        #    print(test_gen.filenames[idx: idx + test_gen.batch_size])
        """
        x_0 = [x[0] for x in predicts]
        x_1 = [x[1] for x in predicts]
        names = [os.path.basename(x) for x in test_gen.filenames]
        print(len(x_0), len(names))

        predicts = np.argmax(predicts, axis=1)
        label_index = {v: k for k, v in train_gen.class_indices.items()}
        predicts = [label_index[p] for p in predicts]

        df = pd.DataFrame(columns=['class_1', 'class_2', 'fname', 'over all'])
        df['fname'] = [os.path.basename(x) for x in test_gen.filenames]
        df['class_1'] = x_0
        df['class_2'] = x_1
        df['over all'] = predicts

    else:
        print('No mode defined')
    # ------generators to feed the model----------------

    train_gen = train_idg.flow_from_directory(train_data_dir,
                                              target_size=(img_width, img_height),
                                              batch_size=20)

    validation_gen = val_idg.flow_from_directory(validation_data_dir,
                                                 target_size=(img_width, img_height),
                                                 batch_size=20)

    # build the ResNet50 network
    base_model = applications.ResNet50(include_top=False, weights='imagenet')
    base_model.trainable = False
    base_model.summary()

    for layer in base_model.layers[:-4]:
        layer.trainable = False

    for layer in base_model.layers[-4:]:
        layer.trainable = True

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=metrics)

    # model.summary()
    epochs = 25
    history = model.fit(train_gen,
                        epochs=epochs,
                        shuffle=1,
                        batch_size=25,
                        validation_batch_size=25,
                        validation_data=validation_gen,
                        verbose=1)

    # --------------- evaluate the model -----------------

    validation_gen = val_idg.flow_from_directory(validation_data_dir,
                                                 target_size=(img_width, img_height),
                                                 batch_size=50)

    evaluation = model.evaluate(validation_gen, verbose=True, steps=10)
    print('VALIDATION dataset')
    print(metrics)
    print(evaluation)"""

    # -------------------------predictions on the validation set --------------------------
    """
    va_gen2 = test_idg2.flow_from_directory(validation_data_dir,
                                            target_size=(img_width, img_height),
                                            shuffle=False,
                                            batch_size=10)

    predict3 = model.predict_generator(va_gen2, verbose=True, steps=1)
    x_0 = [x[0] for x in predict3]
    x_1 = [x[1] for x in predict3]
    names = [os.path.basename(x) for x in va_gen2.filenames[:]]
    print(len(x_0), len(names))

    predict3 = np.argmax(predict3, axis=1)
    label_index = {v: k for k, v in va_gen2.class_indices.items()}
    predict3 = [label_index[p] for p in predict3]
    
    ## ------------- save the weights ------------------
    date = datetime.datetime.strftime(datetime.datetime.today(), '%Y%m%d-%Hh%mm')

    results_directory = ''.join([results_dir, date, '/'])

    if not (os.path.isdir(results_directory)):
        os.mkdir(results_directory)

    model.save_weights(
        ''.join([results_directory, 'weights_resnet50_', data, '_', str(epochs), 'e_', fold, '_.h5']), True)

    # ----------------- save results ---------------------------

    with open(''.join([results_directory, 'resume_training_', data, '_', fold, '_', '_.csv']), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Acc', 'Val_acc', 'Loss', 'Val_Loss'])
        for i, num in enumerate(history.history['accuracy']):
            writer.writerow(
                [num, history.history['val_accuracy'][i], history.history['loss'][i], history.history['val_loss'][i]])

    for case in test_data:

        if data == 'all':
            if case[:3] == 'cys':
                all_cases_dir = all_cases_dir.replace('urs/', 'cys/')
            elif case[:3] == 'urs':
                all_cases_dir = all_cases_dir.replace('all/', 'urs/')

        test_data_dir = all_cases_dir + case
        subdirs = os.listdir(test_data_dir)
        total_all_imgs = 0
        for subdir in subdirs:
            all_imgs = os.listdir(''.join([test_data_dir, '/', subdir, '/']))
            total_all_imgs = total_all_imgs + len(all_imgs)

        # if total_all_imgs > 350:
        #    total_all_imgs = 350

        test_gen = test_idg.flow_from_directory(test_data_dir,
                                                target_size=(img_width, img_height),
                                                shuffle=False,
                                                batch_size=total_all_imgs)

        evaluation_test = model.evaluate(test_gen, verbose=True, steps=1)
        print('TEST ', case)
        print(['loss', 'acc', 'prec', 'rec'])
        print(evaluation_test)

        ###-----------------------lets make predictions-------------------
        predicts = model.predict(test_gen, verbose=True, steps=1)

        # top print the names in each batch of the generator
        # for i in test_gen[0]:
        #    idx = (test_gen.batch_index - 1) * test_gen.batch_size
        #    print(test_gen.filenames[idx: idx + test_gen.batch_size])

        x_0 = [x[0] for x in predicts]
        x_1 = [x[1] for x in predicts]
        names = [os.path.basename(x) for x in test_gen.filenames]
        print(len(x_0), len(names))

        predicts = np.argmax(predicts, axis=1)
        label_index = {v: k for k, v in train_gen.class_indices.items()}
        predicts = [label_index[p] for p in predicts]

        df = pd.DataFrame(columns=['class_1', 'class_2', 'fname', 'over all'])
        df['fname'] = [os.path.basename(x) for x in test_gen.filenames]
        df['class_1'] = x_0
        df['class_2'] = x_1
        df['over all'] = predicts
        # save the predictions  of each case
        name_save_predictions_1 = ''.join(
            [results_directory, 'predictions_ResNet50_', data, '_', case, '.csv'])
        df.to_csv(name_save_predictions_1, index=False)

        # -----------now lets calculate the AUC---------------------------------

        ground_truth_directory = ''
        case_test = case
        real_test = ''.join([ground_truth_directory, case_test, '.csv'])
        print('test file csv', real_test)
        auc = calculate_auc_and_roc(name_save_predictions_1, real_test, case_test, plot=True)
        print(case_test, 'AUC:', auc)

    if test_data_2 != []:

        for case in test_data_2:
            if data == 'cys':
                all_cases_dir = all_cases_dir.replace('cys/', 'urs/')
            elif data == 'urs':
                all_cases_dir = all_cases_dir.replace('urs/', 'cys/')

            test_data_dir = all_cases_dir + case
            subdirs = os.listdir(test_data_dir)
            total_all_imgs = 0
            for subdir in subdirs:
                all_imgs = os.listdir(''.join([test_data_dir, '/', subdir, '/']))
                total_all_imgs = total_all_imgs + len(all_imgs)

            # if total_all_imgs > 350:
            #    total_all_imgs = 350

            test_gen = test_idg.flow_from_directory(test_data_dir,
                                                    target_size=(img_width, img_height),
                                                    shuffle=False,
                                                    batch_size=total_all_imgs)

            evaluation_test = model.evaluate(test_gen, verbose=True, steps=1)
            print('TEST ', case)
            print(['loss', 'acc', 'prec', 'rec'])
            print(evaluation_test)

            ###-----------------------lets make predictions-------------------
            predicts = model.predict(test_gen, verbose=True, steps=1)

            # top print the names in each batch of the generator
            # for i in test_gen[0]:
            #    idx = (test_gen.batch_index - 1) * test_gen.batch_size
            #    print(test_gen.filenames[idx: idx + test_gen.batch_size])

            x_0 = [x[0] for x in predicts]
            x_1 = [x[1] for x in predicts]
            names = [os.path.basename(x) for x in test_gen.filenames]
            print(len(x_0), len(names))

            predicts = np.argmax(predicts, axis=1)
            label_index = {v: k for k, v in train_gen.class_indices.items()}
            predicts = [label_index[p] for p in predicts]

            df = pd.DataFrame(columns=['class_1', 'class_2', 'fname', 'over all'])
            df['fname'] = [os.path.basename(x) for x in test_gen.filenames]
            df['class_1'] = x_0
            df['class_2'] = x_1
            df['over all'] = predicts
            # save the predictions  of each case
            name_save_predictions_1 = ''.join(
                [results_directory, 'predictions_ResNet50_', data, '_',case])
            df.to_csv(name_save_predictions_1, index=False)

            # -----------now lets calculate the AUC---------------------------------

            ground_truth_directory = ''
            case_test = case
            real_test = ''.join([ground_truth_directory, case_test, '.csv'])
            print('test file csv', real_test)
            auc = calculate_auc_and_roc(name_save_predictions_1, real_test, case_test, plot=True)
            print(case_test, 'AUC:', auc)

        plt.show()

    print('results dir:', results_directory)
    plt.show()"""


def main(_argv):

    name_model = FLAGS.name_model
    mode = FLAGS.mode
    backbone_model = FLAGS.backbone
    dataset_dir = FLAGS.train_dataset
    val_dataset_dir = FLAGS.val_dataset

    print('INFORMATION:', name_model, backbone_model, mode)
    call_models(name_model, mode, train_data_dir=dataset_dir, backbone_model=backbone_model,
                validation_data_dir=val_dataset_dir)
    """base_dir = ''
    results_dir = ''
    data_dir = ''
    current_wroking_directory = ''.join([base_dir, data, '/', 'all_cases/', fold, '/'])

    train_dir = ''.join([current_wroking_directory, 'train/'])
    val_dir = ''.join([current_wroking_directory, 'val/'])
    results_directory = ''
    test_data_2 = []
    

    call_models(train_dir, val_dir, data, test_data, all_cases_dir, results_directory, test_data_2, fold=fold)

    all_cases_dir = ''.join([base_dir, data, '/'])

    data = 'all'
    fold = 'fold_3'
    if data == 'cys':
        if fold == 'fold_1':
            test_data = ['cys_case_012', 'cys_case_001', 'cys_case_005']
            test_data_2 = ['urs_case_005', 'urs_case_001', 'urs_case_009']
        if fold == 'fold_2':
            test_data = ['cys_case_010', 'cys_case_002', 'cys_case_008']
            test_data_2 = ['urs_case_007', 'urs_case_012', 'urs_case_011', 'urs_case_006']
        if fold == 'fold_3':
            test_data = ['cys_case_000', 'cys_case_006', 'cys_case_011']
            test_data_2 = ['urs_case_014', 'urs_case_004', 'urs_case_015', 'urs_case_010', 'urs_case_001']

    if data == 'urs':
        if fold == 'fold_1':
            test_data = ['urs_case_005', 'urs_case_001', 'urs_case_009']
            test_data_2 = ['cys_case_012', 'cys_case_001', 'cys_case_005']
        if fold == 'fold_2':
            test_data = ['urs_case_007', 'urs_case_012', 'urs_case_011', 'urs_case_006']
            test_data_2 = ['cys_case_010', 'cys_case_002', 'cys_case_008']
        if fold == 'fold_3':
            test_data = ['urs_case_014', 'urs_case_004', 'urs_case_015', 'urs_case_010', 'urs_case_001']
            test_data_2 = ['cys_case_000', 'cys_case_006', 'cys_case_011']

    if data == 'all':
        if fold == 'fold_1':
            test_data = ['urs_case_005', 'urs_case_001', 'urs_case_009', 'cys_case_012', 'cys_case_001', 'cys_case_005']
        if fold == 'fold_2':
            test_data = ['urs_case_007', 'urs_case_012', 'urs_case_011', 'urs_case_006', 'cys_case_010', 'cys_case_002',
                         'cys_case_008']
        if fold == 'fold_3':
            test_data = ['urs_case_014', 'urs_case_004', 'urs_case_015', 'urs_case_010', 'urs_case_001', 'cys_case_000',
                         'cys_case_006', 'cys_case_011']"""


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
