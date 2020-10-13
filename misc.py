import pickle
import json
import skimage
from pathlib import Path
import datetime
import os
import numpy as np
import tensorflow as tf
import itertools


def get_base_path(training):
    base_path = str(Path(__file__).parent.parent.parent) + "/"

    if training:
        checkpoint_path = base_path + "ckp_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "/"
        os.mkdir(checkpoint_path)
        return base_path, checkpoint_path
    else:
        return base_path


def create_TF_records_folder(data_path, data_purposes, params):
    TF_records_path = data_path + "TF_records_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if params is not None:
        true_params = [key for key in params if params[key] is True]
        if len(true_params) > 0:
            TF_records_path += "-" + "-".join(true_params)
    TF_records_path += "/"

    os.mkdir(TF_records_path)
    for purpose in data_purposes:
        os.mkdir(TF_records_path + purpose)
    return TF_records_path


def save_pickle(path, array):
    with open(path, 'wb') as handle:
        pickle.dump(array, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def save_json(path, array):
    with open(path, 'w') as f:
        json.dump(array, f)


def load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def split_into_chunks(array, size):
    """
        Yield successive n-sized chunks from list.
        i.e. : list(split_into_chunks(list(range(10, 75)), 10))
    """
    for i in range(0, len(array), size):
        yield array[i:i + size]


def make_patches(volume, padding, patch_size):
    padded = np.pad(volume, padding, 'constant')
    blocks = skimage.util.shape.view_as_blocks(padded, (patch_size, patch_size, patch_size))

    patches = []
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            for k in range(blocks.shape[2]):
                patches.append(blocks[i, j, k, :, :, :])

    return np.array(patches)


def flip_data(data):
    dims = list(range(len(data.shape)))
    flips = []
    for L in range(0, len(dims) + 1):
        for subset in itertools.combinations(dims, L):
            flips.append(np.flip(data, axis=subset))

    return np.array(flips)


def get_argmax_prediction(logits):
    probs = tf.nn.softmax(logits)
    predictions = tf.math.argmax(probs, axis=-1)

    return tf.cast(predictions[..., tf.newaxis], tf.float32)
