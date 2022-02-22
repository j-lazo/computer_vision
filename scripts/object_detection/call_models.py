import matplotlib
import matplotlib.pyplot as plt

import os
import random
import io
import imageio
import glob
import scipy.misc
import numpy as np
from PIL import Image, ImageDraw, ImageFont
#from IPython.display import display, Javascript
#from IPython.display import Image as IPyImage
#from six import BytesIO
import keras
import tensorflow as tf

#from object_detection.utils import label_map_util
#from object_detection.utils import config_util
#from object_detection.utils import visualization_utils as viz_utils
#from object_detection.utils import colab_utils
#from object_detection.builders import model_builder

tf.keras.backend.clear_session()

print('Building model and restoring weights for fine-tuning...', flush=True)
num_classes = 1
pipeline_config = 'models/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'
checkpoint_path = 'models/research/object_detection/test_data/checkpoint/ckpt-0'


def load_model(directory_model):
    if directory_model.endswith('.h5'):
        model_path = directory_model
    else:
        files_dir = [f for f in os.listdir(directory_model) if f.endswith('.h5')]
        if files_dir:
            model_path = files_dir.pop()
        else:
            files_dir = [f for f in os.listdir(directory_model + 'saved_model/') if f.endswith('.pb')]
            if files_dir:
                model_path = ''
                print(f'Tensorflow model found at {directory_model}')
            else:
                raise f'No model found in {directory_model}'

    print('MODEL USED:')
    #model = tf.keras.models.load_model(model_path, compile=False)
    print(f'Model path: {directory_model + model_path}')
    model = keras.models.load_model(directory_model + model_path)
    model.summary()
    input_size = (len(model.layers[0].output_shape[:]))

    return model, input_size

dir_model = ''
