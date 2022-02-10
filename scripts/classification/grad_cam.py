import os
import numpy as np
import tensorflow as tf
import random
import cv2
from classification import call_models as img_class
from classification import grad_cam as gc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime
import tqdm

from absl import app, flags, logging
from absl.flags import FLAGS
from general_functions import data_management as dam


def get_img_array(img_path, size):

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def load_preprocess_input(backbone_model):

    if backbone_model == 'vgg16':
        preprocessing_function = tf.keras.applications.vgg16.preprocess_input
        size = (224, 224)

    elif backbone_model == 'vgg19':
        preprocessing_function = tf.keras.applications.vgg19.preprocess_input
        size = (224, 224)

    elif backbone_model == 'inception_v3':
        preprocessing_function = tf.keras.applications.inception_v3.preprocess_input
        size = (299, 299)

    elif backbone_model == 'resnet50':
        preprocessing_function = tf.keras.applications.resnet50.preprocess_input
        size = (224, 224)

    elif backbone_model == 'resnet101':
        preprocessing_function = tf.keras.applications.resnet.preprocess_input
        size = (224, 224)

    elif backbone_model == 'mobilenet':
        preprocessing_function = tf.keras.applications.mobilenet.preprocess_input
        size = (224, 224)

    elif backbone_model == 'densenet121':
        preprocessing_function = tf.keras.applications.densenet.preprocess_input
        size = (224, 224)

    elif backbone_model == 'xception':
        preprocessing_function = tf.keras.applications.xception.preprocess_input
        size = (299, 299)
    else:
        raise ValueError(f' MODEL: {backbone_model} not found')

    return preprocessing_function, size


def create_auxiliar_networks(model, last_conv_layer_name, classifier_layer_names, cap_network=[]):

    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output, name='Features_last_layer_network')
    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input

    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)

    if cap_network:
        x = cap_network(x)

    classifier_model = tf.keras.Model(classifier_input, x, name='Classifier_Model')

    return last_conv_layer_model, classifier_model


def make_gradcam_heatmap(img_array, last_conv_layer_model, classifier_model):

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


def analyze_data_gradcam(name_model, dataset_dir, dataset_to_analyze, output_dir='', plot=False):

    if output_dir == '':
        output_dir = ''.join([dataset_dir, 'results/', name_model,
                              '/gradcam_predictions/'])

        heat_maps_dir = output_dir + 'heatmaps/'
        binary_masks_dir = output_dir + 'predicted_masks/'
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        if not os.path.isdir(heat_maps_dir):
            os.mkdir(heat_maps_dir)

        if not os.path.isdir(binary_masks_dir):
            os.mkdir(binary_masks_dir)

    directory_model = ''.join([dataset_dir, 'results/', name_model,
                               '/model_', name_model])
    model, _ = img_class.load_model(directory_model)
    backbone_model = model.get_layer(index=0)
    classifier_cap = model.get_layer(index=1)

    print(f'Backbone identified: {backbone_model.name}')
    backbone_model.summary()
    classifier_layer_names = list()
    for l in reversed(backbone_model.layers):
        classifier_layer_names.append(l.name)

        if l.__class__.__name__ == 'Concatenate':
            last_conv_layer_name = l.name
            break

        if l.__class__.__name__ == 'Add':
            last_conv_layer_name = l.name
            break

        if l.__class__.__name__ == 'Conv2D':
            last_conv_layer_name = l.name
            break

    classifier_layer_names.remove(last_conv_layer_name)
    classifier_layer_names.reverse()
    print(last_conv_layer_name)
    print(classifier_layer_names)
    last_conv_layer_model, classifier_model = gc.create_auxiliar_networks(backbone_model,
                                                        last_conv_layer_name,
                                                        classifier_layer_names,
                                                        cap_network=classifier_cap)

    # load the data
    test_dataset = os.listdir(dataset_to_analyze)
    test_dataset = [f for f in test_dataset if os.path.isdir(os.path.join(dataset_to_analyze, f))]
    list_imgs = list()

    for folder in test_dataset:
        dir_folder = ''.join([dataset_to_analyze, folder, '/'])
        imgs_subdir = [dir_folder + f for f in os.listdir(dir_folder) if f.endswith('.png')]
        list_imgs = list_imgs + imgs_subdir

    for i, img_path in enumerate(tqdm.tqdm(list_imgs, desc=f'Making mask predictions, {len(list_imgs)} images')):
        # if you want to pick random paths uncoment bewlloa and comment the previous one to have a counter
        #img_path = random.choice(list_imgs)
        preprocess_input, img_size = gc.load_preprocess_input(backbone_model.name)
        img_array = preprocess_input(gc.get_img_array(img_path, size=img_size))
        img_name = os.path.split(img_path)[-1]
        img = tf.keras.preprocessing.image.load_img(img_path)
        img = tf.keras.preprocessing.image.img_to_array(img)

        test_img = cv2.imread(img_path)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(test_img, img_size, interpolation=cv2.INTER_AREA)

        start_time = datetime.datetime.now()
        heatmap = gc.make_gradcam_heatmap(img_array, last_conv_layer_model, classifier_model)
        print('Inference TIME:', (datetime.datetime.now() - start_time))

        mask_heatmap = cv2.resize(heatmap, img_size)

        if np.isnan(np.sum(np.unique(mask_heatmap))):
            mask_heatmap = np.zeros(img_size)
        else:
            limit = 0.75 * np.amax(mask_heatmap)
            mask_heatmap[mask_heatmap >= limit] = 1
            mask_heatmap[mask_heatmap < limit] = 0

        # We rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)


        # We use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # We use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # We create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))

        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * 0.4 + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

        w, d = np.shape(mask_heatmap)
        binary_mask = np.zeros((w, d, 3))
        binary_mask[:, :, 0] = mask_heatmap * 255
        binary_mask[:, :, 1] = mask_heatmap * 255
        binary_mask[:, :, 2] = mask_heatmap * 255

        cv2.imwrite(binary_masks_dir + img_name, binary_mask)
        superimposed_img.save(heat_maps_dir + img_name)

        if plot is True:
            plt.figure()
            plt.subplot(131)
            plt.imshow(img_resized)
            plt.subplot(132)
            plt.imshow(superimposed_img)
            plt.subplot(133)
            plt.imshow(mask_heatmap)
            plt.show()




def main():
    pass

if __name__ == '__main__':
    #flags.DEFINE_string('name_model', '', 'name of the model')
    try:
        app.run(main)
    except SystemExit:
        pass
    pass