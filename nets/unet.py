import tensorflow as tf

from tf_utils.blocks.cnn_block import CNN


class UNet(tf.keras.Model):
    def __init__(self, n_classes, n_layers, starting_filters, k_size, init, batch_norm, dropout, activation):
        super(UNet, self).__init__()

        self.encoder = []
        for i in range(n_layers):
            n_filters = 512 if starting_filters * (2 ** i) > 512 else starting_filters * (2 ** i)
            if i == 0:
                self.encoder.append(CNN(n_filters, k_size, strides=2, kernel_initializer=init, batch_norm=False, dropout=False, activation=activation))
            else:
                self.encoder.append(CNN(n_filters, k_size, strides=2, kernel_initializer=init, batch_norm=batch_norm, dropout=False, activation=activation))

        self.decoder = []
        for i in range(n_layers - 2, -1, -1):
            n_filters = 512 if starting_filters * (2 ** i) > 512 else starting_filters * (2 ** i)
            self.decoder.append(CNN(n_filters, k_size, strides=2, kernel_initializer=init, batch_norm=batch_norm, dropout=dropout, activation=activation, up=True))

        self.conv = CNN(starting_filters, 3, strides=2, kernel_initializer=init, batch_norm=batch_norm, dropout=0., activation=activation, up=True)
        self.last_conv = CNN(n_classes, 3, strides=1, kernel_initializer=init, batch_norm=None, dropout=0., activation=None)

    def call(self, x, training):

        skips = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x, training=training)
            skips.append(x)

        skips = list(reversed(skips[:-1]))
        for i in range(len(self.decoder)):
            x = self.decoder[i](x, training=training)
            x = tf.keras.layers.Concatenate()([x, skips[i]])

        x = self.conv(x)
        return self.last_conv(x)

    def summary(self, input_shape):
        """
        :param input_shape: (32, 32, 1)
        """
        x = tf.keras.Input(shape=input_shape)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x, training=False))
        tf.keras.utils.plot_model(model, to_file='UNet.png', show_shapes=True, expand_nested=True)
        model.summary(line_length=200)


if __name__ == "__main__":
    deepseacat = UNet(9, 5, 64, 3, "he_normal", False, 0., tf.keras.layers.LeakyReLU)
    deepseacat.summary((32, 32, 32, 2))
