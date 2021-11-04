from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential
from keras import applications
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from matplotlib import pyplot as plt
import datetime
import os
import pandas as pd
import classification_models
import shutil

from sklearn.metrics import roc_curve, auc
from keras import regularizers

flags.DEFINE_string('train_dataset', '', 'path to dataset')
flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
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
flags.DEFINE_boolean('multi_gpu', False, 'Use if wishing to train with more than 1 GPU.')


def load_pretrained_model(name_model, weights='imagenet'):

    """
    Loads a pretrained model given a name
    :param name_model: (str) name of the model
    :param weights: (str) weights names (default imagenet)
    :return: sequential model with the selected weights
    """

    if name_model == 'VGG16':
        base_model = applications.VGG16(include_top=False, weights=weights)
        base_model.trainable = False
        base_model.summary()

    elif name_model == 'VGG19':
        base_model = applications.VGG19(include_top=False, weights=weights)
        base_model.trainable = False
        base_model.summary()

    elif name_model == 'InceptionV3':
        base_model = applications.InceptionV3(include_top=False, weights=weights)
        base_model.trainable = False
        base_model.summary()

    elif name_model == 'ResNet50':
        base_model = applications.ResNet50(include_top=False, weights=weights)
        base_model.trainable = False
        base_model.summary()

    elif name_model == 'ResNet101':
        base_model = applications.ResNet101(include_top=False, weights=weights)
        base_model.trainable = False
        base_model.summary()

    elif name_model == 'MobileNet':
        base_model = applications.MobileNet(include_top=False, weights=weights)
        base_model.trainable = False
        base_model.summary()

    elif name_model == 'DenseNet121':
        base_model = applications.DenseNet121(include_top=False, weights=weights)
        base_model.trainable = False
        base_model.summary()

    return base_model

def load_model(name_model, backbone=False, backbone_model=''):
    model = Sequential()

    if backbone is True:
        base_model = load_pretrained_model(backbone_model)
        model.add(base_model)
    else:
        ehm_model? = classification_models.name_model()
        model = model.add(ehm_model?)
    return model


def call_models(name_model, mode, backbone_model = '', train_data_dir, validation_data_dir, test_data, all_cases_dir, results_dir):

    # first load model
    if backbone_model != '':
        backbone = True
    model = load_model(name_model, backbone, backbone_model)

    adam = Adam(lr=0.0001)
    sgd = SGD(lr=0.001, momentum=0.9)
    rms = 'rmsprop'
    metrics = ["accuracy", tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=metrics)
    model.summary()
    img_width, img_height = 224, 224


    if mode == 'train':

        # train the model
        train_gen =
        validation_gen =
        train_idg = ImageDataGenerator(preprocessing_function=preprocess_input)
        val_idg = ImageDataGenerator(preprocessing_function=preprocess_input)

        model.fit(train_gen,
                  epochs=3,
                  shuffle=1,
                  batch_size=20,
                  validation_batch_size=20,
                  validation_data=validation_gen,
                  verbose=1)
    elif mode == 'predict':

        predictions = model.predict(test_gen, verbose=True, steps=1)

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
    print(evaluation)

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
    """
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
    plt.show()


def main(_argv):
    name_model = FLAGS.name_model
    mode = FLAGS.mode

    base_dir = ''
    results_dir = ''
    data_dir = ''
    current_wroking_directory = ''.join([base_dir, data, '/', 'all_cases/', fold, '/'])

    train_dir = ''.join([current_wroking_directory, 'train/'])
    val_dir = ''.join([current_wroking_directory, 'val/'])
    results_directory = ''
    test_data_2 = []
    call_models(name_model, mode, train_data_dir, validation_data_dir, test_data, all_cases_dir, results_dir,
                test_data_2, fold='')

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
                         'cys_case_006', 'cys_case_011']


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
