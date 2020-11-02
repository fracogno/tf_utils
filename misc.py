import pickle
import json
import skimage
from pathlib import Path
import datetime
import os
import numpy as np
import tensorflow as tf
import itertools
import random


def get_base_path(training):
    base_path = str(Path(__file__).parent.parent.parent) + "/"

    if training:
        checkpoint_path = base_path + "ckp_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "/"
        os.mkdir(checkpoint_path)
        return base_path, checkpoint_path
    else:
        return base_path


def create_TF_records_folder(data_path, data_purposes):
    TF_records_path = data_path + "TF_records_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "/"

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


def get_cut_indices(volume, desired_shape):
    """
    :param volume: Given a volume, find indices where we could cut removing some useless background
    :param desired_shape: Chosen indices
    :return:
    """
    assert (len(volume.shape) == 3 and len(desired_shape) == 3)
    result = (volume.shape[0] - desired_shape[0], volume.shape[1] - desired_shape[1], volume.shape[2] - desired_shape[2])

    volume_non_zeros = np.count_nonzero(volume)
    range_x, range_y, range_z = list(range(result[0])), list(range(result[1])), list(range(result[2]))
    random.shuffle(range_x)
    random.shuffle(range_y)
    random.shuffle(range_z)
    
    checked = 0
    for i in range_x:
        for j in range_y:
            for k in range_z:
                if np.count_nonzero(volume[i:-(result[0] - i), j:-(result[1] - j), k:-(result[2] - k)]) == volume_non_zeros:
                    return (i, result[0] - i), (j, result[1] - j), (k, result[2] - k)

                checked += 1
                if checked > 1000:
                    break

    return (result[0] // 2, result[0] // 2), (result[1] // 2, result[1] // 2), (result[2] // 2, result[2] // 2)


def get_cut_volume(volume, x_cut, y_cut, z_cut):
    cut_volume = volume[x_cut[0]:-x_cut[1], y_cut[0]:-y_cut[1], z_cut[0]:-z_cut[1]]
    #assert (np.count_nonzero(volume) == np.count_nonzero(cut_volume))
    return cut_volume
