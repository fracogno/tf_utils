import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, Concatenate, BatchNormalization

from tf_utils.layers.maxout import Maxout

class CompUnpoolBlock(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size=5):
        super(CompUnpoolBlock, self).__init__()
        self.concat1 = Concatenate()
        self.conv = Conv2D(num_filters, kernel_size=kernel_size, padding='same')
        self.relu = ReLU()
        self.batch_norm = BatchNormalization()
        self.concat2 = Concatenate()
        self.max_out = Maxout()
        pass

    def call(self, inputs, training=None, mask=None):

        if not isinstance(inputs, list):
            raise ValueError("CUB layer should be called on a list of inputs.")

        main_input = inputs[0]
        skip_input = inputs[1]

        x = self.concat1([main_input, skip_input])
        x = self.conv(x, training=training)
        x = self.relu(x, training=training)
        x = self.batch_norm(x, training=training)

        x = self.max_out([x, main_input])
        return x

    def get_config(self):
        pass


    def plot_summary(self, input_shapes):
        x = tf.keras.Input(shape=input_shapes[0])
        x2 = tf.keras.Input(shape=input_shapes[1])
        model = tf.keras.Model(inputs=[x, x2], outputs=self.call([x, x2], training=False))
        tf.keras.utils.plot_model(model, to_file='CUB.png', show_shapes=True, expand_nested=True)
        model.summary(line_length=200)


if __name__ == '__main__':
    block = CompUnpoolBlock(64)
    block.plot_summary([(16, 16, 64), (16, 16, 64)])



