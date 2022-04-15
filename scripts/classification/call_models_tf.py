import copy
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import tensorflow as tf
from absl import app, flags
from absl.flags import FLAGS
from tensorflow.keras import applications
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Nadam
import pandas as pd
import tqdm
import time


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


def load_data_from_directory(path_data):
    """
    Give a path, creates two lists with the
    Parameters
    ----------
    path_data :

    Returns
    -------

    """
    images_path = list()
    labels = list()
    dictionary_labels = {}

    list_files = os.listdir(path_data)
    list_unique_classes = np.unique([f for f in list_files if os.path.isdir(os.path.join(path_data, f))])
    for j, unique_class in enumerate(list_unique_classes):
        path_images = ''.join([path_data, '/', unique_class, '/*'])
        added_images = sorted(glob(path_images))
        new_dictionary_labels = {image_name: unique_class for image_name in added_images}

        images_path = images_path + added_images
        added_labels = [j] * len(added_images)
        labels = labels + added_labels
        dictionary_labels = {**dictionary_labels, **new_dictionary_labels}

    print(f'Found {len(images_path)} images corresponding to {len(list_unique_classes)} classes at: {path_data}')

    return images_path, labels, dictionary_labels


def read_stacked_images_npy(path_data):
    """

    Parameters
    ----------
    path_data : (bytes) path to the data
    preprocessing_input : Pr-processing input unit to be used in case some backbone is used in the classifier

    Returns
    -------

    """
    path_data = path_data.decode()
    if path_data.endswith('.npz'):
        img_array = np.load(path_data)
        img = img_array['arr_0']
    else:
        img = np.load(path_data)

    img = img.astype(np.float64)
    return img


def tf_parser_npy(x, y):

    def _parse(x, y):
        x = read_stacked_images_npy(x)
        out_y = np.zeros(NUM_CLASSES)
        out_y[y] = 1.
        out_y[y] = out_y[y].astype(np.float64)
        return x, out_y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([3, 256, 256, 3])
    y.set_shape([NUM_CLASSES])
    return x, y


def generate_tf_dataset(x, y, batch_size=1, shuffle=False, buffer_size=10, preprocess_function=None,
                        input_size=(256, 256)):

    """
    Generates a tf dataset asd described in https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    Parameters
    ----------
    x : (list of strings) input data
    y : (list of int) target labels
    batch_size : int
    shuffle : (bool)

    Returns
    -------
    tensorflow Dataset
    """
    global PREPROCESS_FUNCTION
    global INPUT_SIZE
    global NUM_CLASSES

    NUM_CLASSES = len(np.unique(y))
    PREPROCESS_FUNCTION = preprocess_function
    INPUT_SIZE = input_size

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size * batch_size)

    dataset = dataset.map(tf_parser_npy)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()

    return dataset


def analyze_tf_dataset(dataset_dir, plot=True):

    """
    Given the directory to a classification directory, it generates a tf dataset and iterates over it to
    visualize it
    Parameters
    ----------
    dataset_dir : (str) Path to the dataset

    Returns
    -------

    """

    test_x, test_y, dataset_dictionary = load_data_from_directory(dataset_dir)
    test_dataset = generate_tf_dataset(test_x, test_y, batch_size=8, shuffle=True,
                                       buffer_size=500)

    unique_labels = np.unique([f for f in os.listdir(dataset_dir) if os.path.isdir(dataset_dir + f)])
    for j, element in enumerate(test_dataset):
        x, y = element
        x_array = x.numpy()
        y_array = y.numpy()
        print('x:', np.shape(x_array), np.amin(x_array), np.amax(x_array))
        print('y:', np.shape(y_array), np.unique(y_array))
        if plot is True:

            plt.figure()
            plt.subplot(3, 4, 1)
            plt.imshow(x_array[0][0])
            plt.axis('off')
            plt.gca().set_title(str(unique_labels[list(y_array[0]).index(1)]))
            plt.subplot(3, 4, 2)
            plt.imshow(x_array[1][0])
            plt.axis('off')
            plt.gca().set_title(str(unique_labels[list(y_array[1]).index(1)]))
            plt.subplot(3, 4, 3)
            plt.imshow(x_array[2][0])
            plt.axis('off')
            plt.gca().set_title(str(unique_labels[list(y_array[2]).index(1)]))
            plt.subplot(3, 4, 4)
            plt.imshow(x_array[3][0])
            plt.axis('off')
            plt.gca().set_title(str(unique_labels[list(y_array[3]).index(1)]))

            plt.subplot(3, 4, 5)
            plt.imshow(x_array[0][1])
            plt.axis('off')
            plt.subplot(3, 4, 6)
            plt.imshow(x_array[1][1])
            plt.axis('off')
            plt.subplot(3, 4, 7)
            plt.imshow(x_array[2][1])
            plt.axis('off')
            plt.subplot(3, 4, 8)
            plt.imshow(x_array[3][1])
            plt.axis('off')

            plt.subplot(3, 4, 9)
            plt.imshow(x_array[0][2])
            plt.axis('off')
            plt.subplot(3, 4, 10)
            plt.imshow(x_array[1][2])
            plt.axis('off')
            plt.subplot(3, 4, 11)
            plt.imshow(x_array[2][2])
            plt.axis('off')
            plt.subplot(3, 4, 12)
            plt.imshow(x_array[3][2])
            plt.axis('off')
            plt.show()


def load_pretrained_backbones(name_model, weights='imagenet', include_top=False, trainable=False, new_name=None):
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
        base_model.trainable = True
        base_model.name = new_name
        input_size = (224, 224, 3)

    elif name_model == 'resnet101':
        weights_dir = base_dir_weights + 'resnet101/resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.resnet.ResNet101(include_top=include_top, weights=weights_dir)
        base_model.trainable = True
        layer1 = base_model.layers[2]
        global weights_1
        weights_1 = layer1.weights

        new_base_model = tf.keras.models.clone_model(base_model)
        new_base_model.set_weights(base_model.get_weights())
        layer2 = new_base_model.layers[2]
        global weights_2
        weights_2 = layer2.weights
        print(np.array_equal(weights_1[0], weights_2[0]))

        #base_model.name = new_name

        input_size = (224, 224, 3)

    elif name_model == 'mobilenet':
        weights_dir = base_dir_weights + 'mobilenet/mobilenet_1_0_224_tf_no_top.h5'
        base_model = applications.mobilenet.MobileNet(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable
        input_size = (224, 224, 3)

    elif name_model == 'densenet121':
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

    layer3 = new_base_model.layers[2]
    global weights3
    weights3 = layer3.weights
    print(np.array_equal(weights3[0], weights_2[0]))
    return new_base_model


#def build_model(model_type, list_backbone_models):
#    input_sizes_models = {'vgg16': (224, 224), 'vgg19': (224, 224), 'inception_v3': (299, 299),
#                          'resnet50': (224, 224), 'resnet101': (224, 244), 'mobilenet': (224, 224),
#                          'densenet121': (224, 224), 'xception': (299, 299)}

def build_model():
    input_model = Input((3, 256, 256, 3))

    x1, x2, x3 = tf.split(input_model, 3, axis=1)
    input_backbone_1 = tf.squeeze(x1, axis=1)
    input_backbone_2 = tf.squeeze(x2, axis=1)
    input_backbone_3 = tf.squeeze(x3, axis=1)

    b1 = tf.image.resize(input_backbone_1, (224, 224), method='bilinear')
    b2 = tf.image.resize(input_backbone_2, (224, 224), method='bilinear')
    b3 = tf.image.resize(input_backbone_3, (224, 224), method='bilinear')

    b1 = tf.keras.applications.resnet.preprocess_input(b1)
    b2 = tf.keras.applications.resnet.preprocess_input(b2)
    b3 = tf.keras.applications.resnet.preprocess_input(b3)

    backbone_model_1 = load_pretrained_backbones('resnet101')
    backbone_model_2 = load_pretrained_backbones('resnet101')
    backbone_model_3 = load_pretrained_backbones('resnet101')

    backbone_model_1._name = 'backbone_1'
    for layer in backbone_model_1.layers:
        layer.trainable = False

    backbone_model_2._name = 'backbone_2'
    for layer in backbone_model_2.layers:
        layer.trainable = False
    backbone_model_3._name = 'backbone_3'
    for layer in backbone_model_3.layers:
        layer.trainable = False

    """layer4 = backbone_model_1.layers[2]
    weights4 = layer4.weights
    print(np.array_equal(weights4[0], weights3[0]))

    layer5 = backbone_model_2.layers[2]
    weights5 = layer5.weights
    print(np.array_equal(weights4[0], weights5[0]))

    layer6 = backbone_model_3.layers[2]
    weights6 = layer6.weights
    print(np.array_equal(weights5[0], weights6[0]))"""

    b1 = backbone_model_1(b1)
    b2 = backbone_model_2(b2)
    b3 = backbone_model_3(b3)

    x = Concatenate()([b1, b2, b3])
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    #x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    #x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Flatten()(x)
    output_layer = Dense(5, activation='softmax')(x)

    ensemble = Model(inputs=input_model,
                     outputs=output_layer, name='Multi_Imaging_Classification')

    return ensemble

"""def build_model():

    
    ## Model
    # check here: https://github.com/tensorflow/tensorflow/issues/20698
    #inputs = layers.Concatenate()([model_data_a.output, model_data_b.output]); ### Get the outputs of 'model_data_a' , 'model_data_b' and combine the outputs
    #outputs = layers.Dense(1, activation=relu)(inputs);
    #model = tf.keras.Model([model_data_a.input, model_data_b.input], outputs); ## Add both inputs


    num_classes = 5
    #input_model = Input((3, 256, 256, 3))
    input_original = Input((256, 256, 3))
    input_convert = Input((256, 256, 3))
    input_reconvert = Input((256, 256, 3))
    #print('General input:', input_model)
    #x1, _, _ = tf.split(input_model[0, :, :, :, :], num_or_size_splits=3, axis=0)
    #print('AFTER split:', x1)
    #x1 = input_model[:, 0, :, :, :]
    input_backbone_1 = input_original
    print('Input backbone 1:', np.shape(input_backbone_1))
    input_backbone_2 = input_convert
    input_backbone_3 = input_reconvert

    b1 = tf.image.resize(input_backbone_1, (224, 224), method='bilinear')
    b2 = tf.image.resize(input_backbone_2, (224, 224), method='bilinear')
    b3 = tf.image.resize(input_backbone_3, (224, 224), method='bilinear')

    b1 = tf.keras.applications.resnet.preprocess_input(b1)
    b2 = tf.keras.applications.resnet.preprocess_input(b2)
    b3 = tf.keras.applications.resnet.preprocess_input(b3)

    backbone_model_1 = load_pretrained_backbones('resnet101')
    backbone_model_2 = load_pretrained_backbones('resnet101')
    backbone_model_3 = load_pretrained_backbones('resnet101')

    backbone_model_1._name = 'backbone_1'
    for layer in backbone_model_1.layers:
        layer.trainable = False
    backbone_model_2._name = 'backbone_2'
    for layer in backbone_model_2.layers:
        layer.trainable = False
    backbone_model_3._name = 'backbone_3'
    for layer in backbone_model_3.layers:
        layer.trainable = False

    b1 = backbone_model_1(b1)
    b2 = backbone_model_2(b2)
    b3 = backbone_model_3(b3)
    print(np.shape(b1))
    print(np.shape(b2))
    print(np.shape(b3))
    x = Concatenate()([b1, b2, b3])
    print(np.shape(x))
    x = GlobalAveragePooling2D()(x)
    print(np.shape(x))
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Flatten()(x)
    output_layer = Dense(5, activation='softmax')(x)
    print('output shape', np.shape(output_layer))

    ensemble = Model(inputs=[input_original, input_convert, input_reconvert],
                     outputs=[output_layer], name='Ensemble_Classification')

    return ensemble"""


def evaluate_and_predict(model, directory_to_evaluate, results_directory,
                         output_name='', results_id='', batch_size=1,
                         analyze_data=False, output_dir=''):
    print(f'Evaluation of: {directory_to_evaluate}')

    # load the data to evaluate and predict

    batch_size = 8
    test_x, test_y, dataset_dictionary = load_data_from_directory(directory_to_evaluate)
    test_dataset = generate_tf_dataset(test_x, test_y, batch_size=batch_size)
    test_steps = (len(test_x) // batch_size)

    if len(test_x) % batch_size != 0:
        test_steps += 1

    model.evaluate(test_dataset, steps=test_steps)
    inference_times = []
    evaluation = model.evaluate(test_dataset, verbose=True)
    print('Evaluation results:')
    print(evaluation)

    for i, (x, y) in tqdm.tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        init_time = time.time()
        x = read_stacked_images_npy(x)
        x = np.expand_dims(x, axis=0)
        y_pred = model.predict(x)
        print(y_pred, y)
        inference_times.append(time.time() - init_time)
        # determine the top-1 prediction class
        predicts = np.argmax(y_pred, axis=1)

    print('End')
    """    for x in predictions:
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

    return results_csv_file"""


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
                                                 analyze_data=analyze_data)

                print(f'Evaluation results saved at {name_file}')
            else:
                name_file = evaluate_and_predict(model, test_data, results_directory,
                                                 results_id=new_results_id, output_name='test',
                                                 analyze_data=analyze_data)
                print(f'Evaluation results saved at {name_file}')
                break

    else:
        name_file = evaluate_and_predict(model, test_data, results_directory,
                                         results_id=new_results_id, output_name='test',
                                         analyze_data=analyze_data)
        print(f'Evaluation results saved at {name_file}')


def compile_model(model, learning_rate, optimizer='adam', loss='categorical_crossentropy',
                metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]):

    if optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()

    return model


def fit_model(name_model, dataset_dir, epochs=50, learning_rate=0.0001, results_dir=os.getcwd() + '/results/', backbone_model=None,
              val_dataset=None, eval_val_set=None, eval_train_set=False, test_data=None, batch_size=16, buffer_size=50):
    mode = 'fit'
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # Decide how to act according to the mode (train/predict/train-backbone... )
    files_dataset_directory = [f for f in os.listdir(dataset_dir)]
    if 'train' in files_dataset_directory:
        path_train_dataset = os.path.join(dataset_dir, 'train')
    else:
        path_train_dataset = dataset_dir

    if 'val' in files_dataset_directory:
        path_val_dataset = os.path.join(dataset_dir, 'val')
    elif val_dataset:
        path_val_dataset = val_dataset
    else:
        raise 'Validation directory not found'

    print(f'train directory found at: {path_train_dataset}')
    print(f'validation directory found at: {path_train_dataset}')

    train_x, train_y, dictionary_train = load_data_from_directory(path_train_dataset)
    train_dataset = generate_tf_dataset(train_x, train_y, batch_size=batch_size, shuffle=True,
                                       buffer_size=buffer_size)

    val_x, val_y, dictionary_val = load_data_from_directory(path_val_dataset)
    val_dataset = generate_tf_dataset(val_x, val_y, batch_size=batch_size, shuffle=True,
                                       buffer_size=buffer_size)

    train_steps = len(train_x) // batch_size
    val_steps = len(val_x) // batch_size

    if len(train_x) % batch_size != 0:
        train_steps += 1
    if len(val_x) % batch_size != 0:
        val_steps += 1

    # define a dir to save the results and Checkpoints
    # if results directory doesn't exist create it
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    # ID name for the folder and results
    name_model = 'gan_multi_input'
    backbone_model = 'resnet101'
    new_results_id = generate_experiment_ID(name_model=name_model, learning_rate=learning_rate,
                                            batch_size=batch_size, backbone_model=backbone_model,
                                            mode=mode)

    results_directory = ''.join([results_dir, new_results_id, '/'])
    # if results experiment doesn't exists create it
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    # Build the model

    model = build_model()
    model = compile_model(model, learning_rate)

    temp_name_model = results_directory + new_results_id + "_model.h5"
    callbacks = [
        ModelCheckpoint(temp_name_model,
                        monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', patience=25),
        CSVLogger(results_directory + 'train_history_' + new_results_id + "_.csv"),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]

    # track time
    start_time = datetime.datetime.now()
    # Train the model

    trained_model = model.fit(train_dataset,
                              epochs=epochs,
                              shuffle=True,
                              batch_size=batch_size,
                              validation_data=val_dataset,
                              steps_per_epoch=train_steps,
                              validation_steps=val_steps,
                              verbose=True,
                              callbacks=callbacks)

    model.save(''.join([results_directory, 'model_', new_results_id]))

    print('Total Training TIME:', (datetime.datetime.now() - start_time))
    print('History Model Keys:')
    print(trained_model.history.keys())
    # in case evaluate val dataset is True
    if eval_val_set is True:
        evaluate_and_predict(model, val_dataset, results_directory,
                             results_id=new_results_id, output_name='val',
                             backbone_model=backbone_model)

    if eval_train_set is True:
        evaluate_and_predict(model, train_dataset, results_directory,
                             results_id=new_results_id, output_name='train',
                             backbone_model=backbone_model)

    if 'test' in files_dataset_directory:
        path_test_dataset = os.path.join(dataset_dir, 'test')
        list_sub_dirs = os.listdir(path_test_dataset)
        if list_sub_dirs:
            for folder in list_sub_dirs:
                sub_path_test_dataset = os.path.join(path_test_dataset, folder)
                evalute_test_directory(model, sub_path_test_dataset + '/', results_directory, new_results_id, backbone_model, analyze_data=True)
        else:
            evalute_test_directory(model, path_test_dataset, results_directory, new_results_id, backbone_model,
                                   analyze_data=True)
    if test_data != '':
        evalute_test_directory(model, path_test_dataset, results_directory, new_results_id, backbone_model,
                               analyze_data=True)


def main(_argv):

    name_model = FLAGS.name_model
    mode = FLAGS.mode
    train_dataset = FLAGS.dataset_dir
    test_dataset = FLAGS.test_dataset
    val_dataset = FLAGS.val_dataset
    backbone_model = FLAGS.backbone
    batch_size = FLAGS.batch_size
    buffer_size = FLAGS.buffer_size
    epochs = FLAGS.epochs

    if mode == 'analyze_dataset':
        analyze_tf_dataset(test_dataset)

    elif mode == 'fit':
        fit_model('initial_test', train_dataset, val_dataset=val_dataset, epochs=epochs)
        #fit_model(name_model, train_dataset, backbone_model, val_dataset=val_dataset, batch_size=batch_size,
        #          buffer_size=buffer_size)


if __name__ == '__main__':

    flags.DEFINE_string('name_model', '', 'name of the model')
    flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf', 'analyze_dataset'],
                      'fit: model.fit, '
                      'eager_fit: model.fit(run_eagerly=True), '
                      'eager_tf: custom GradientTape,'
                      'analyze_dataset: analyze_dataset')
    flags.DEFINE_string('backbone', None, 'backbone network')
    flags.DEFINE_string('dataset_dir', os.getcwd() + 'data/', 'path to dataset')
    flags.DEFINE_string('val_dataset', None, 'path to validation dataset')
    flags.DEFINE_string('test_dataset', '', 'path to test dataset')
    flags.DEFINE_string('results_dir', os.getcwd() + 'results/', 'path to dataset')
    flags.DEFINE_integer('epochs', 1, 'number of epochs')
    flags.DEFINE_integer('batch_size', 4, 'batch size')
    flags.DEFINE_integer('buffer_size', 500, 'buffer size when shuffle dataset')
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    flags.DEFINE_string('weights', '', 'path to weights file')
    flags.DEFINE_bool('analyze_data', False, 'select if analyze data or not')
    flags.DEFINE_string('directory_model', '', 'indicate the path to the directory')
    flags.DEFINE_float('validation_split', 0.2, 'iif not validation dir but needed')
    flags.DEFINE_string('file_to_predic', '',
                        'Directory or file where to perform predictions if predict mode selected')
    flags.DEFINE_integer('trainable_layers', -1, 'Trainable layers in case backbone is trained')


    try:
        app.run(main)
    except SystemExit:
        pass