import tensorflow as tf
import os
import numpy as np

from . import misc


class TFRecordsManager:

    def __init__(self):
        pass

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def save_record(self, path, data):
        writer = tf.io.TFRecordWriter(path + ".tfrecord", options=tf.io.TFRecordOptions(compression_type="GZIP"))

        # Convert to NumPy array float32
        for key in data:
            data[key] = np.array(data[key], dtype=np.float32)

        keys = list(data.keys())
        # Iterate over each sample
        for index in range(len(data[keys[0]])):

            # Iterate over keys of data (X, Y, etc.)
            feature = {}
            for key in keys:
                feature[key] = self._bytes_feature(data[key][index].tostring())

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        writer.close()

    def get_records_filenames(self, path):
        filenames = os.listdir(path)
        for index in range(len(filenames)):
            filenames[index] = path + filenames[index]
        filenames.sort()
        return filenames

    def parser_TFRecord(self, record, data_purpose, params):
        features = {}
        for key in params["data_keys"]:
            features[key] = tf.io.FixedLenFeature([], tf.string, default_value="")

        parsed_record = tf.io.parse_single_example(record, features)
        data = {}
        for key in params["data_keys"]:
            data[key] = tf.reshape(tf.io.decode_raw(parsed_record[key], tf.float32), params["shapes"][data_purpose][key])

        return data

    def load_datasets(self, path, batch_size):
        params = misc.load_json(path + "params.json")

        datasets = {}
        for data_purpose in params["data_purposes"]:
            dataset = tf.data.TFRecordDataset(self.get_records_filenames(path + data_purpose + "/"),
                                              compression_type='GZIP').map(lambda record: self.parser_TFRecord(record, data_purpose, params))

            dataset = dataset.shuffle(100)
            dataset = dataset.batch(batch_size) if "train" in data_purpose else dataset.batch(1)
            datasets[data_purpose] = dataset
            del dataset

        return datasets
