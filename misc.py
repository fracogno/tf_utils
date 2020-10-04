import pickle
import json
from pathlib import Path
import datetime
import os


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
