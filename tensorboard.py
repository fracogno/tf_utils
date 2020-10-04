import tensorflow as tf


class Tensorboard:

    def __init__(self, ckp_path, scalars, modes):
        self.ckp_path = ckp_path
        self.scalars = scalars

        if ckp_path is not None:
            self.summary_writers = {mode: tf.summary.create_file_writer(self.ckp_path + 'logs/' + mode) for mode in modes}

    def write_tensorboard(self, mode, images, shape, epoch):
        """ Write stats in tensorboard file. """
        with self.summary_writers[mode].as_default():
            for key in self.scalars:
                # Check if list or not
                if isinstance(self.scalars[key], list):
                    for step in range(len(self.scalars[key])):
                        tf.summary.scalar(key + '-' + str(step), self.scalars[key][step].result(), step=epoch)
                else:
                    tf.summary.scalar(key, self.scalars[key].result(), step=epoch)

            # Tensorboard images
            if images is not None:
                for key in list(images.keys()):
                    tf.summary.image(key, images[key][:, :, :, int(shape // 2)], max_outputs=1, step=epoch)

        self.reset_states_tensorboard()

    def reset_states_tensorboard(self):
        """ After each epoch reset accumulating tensorboard metrics. """
        for key in self.scalars:
            if isinstance(self.scalars[key], list):
                for step in range(len(self.scalars[key])):
                    self.scalars[key][step].reset_states()
            else:
                self.scalars[key].reset_states()
