import tensorflow as tf


class MetricsManager:

    def __init__(self):
        self.metrics = {
            "GDICEL": self.generalize_dice_loss,
            "DICEL": self.dice_loss,
            "CE": self.cross_entropy,
            "WCE": self.weighted_cross_entropy
        }

    def generalize_dice_loss(self, one_hot, logits):
        w = tf.reduce_sum(one_hot, axis=[1, 2, 3])
        w = 1 / (w ** 2 + 1e-6)
        # w = w / tf.reduce_sum(w)  # Normalize weights

        probs = tf.nn.softmax(logits)

        multed = tf.reduce_sum(probs * one_hot, axis=[1, 2, 3])
        summed = tf.reduce_sum(probs, axis=[1, 2, 3]) + tf.reduce_sum(one_hot, axis=[1, 2, 3])

        numerator = w * multed
        denominator = w * summed

        dice_score = tf.reduce_mean(2. * numerator / (denominator + 1e-6), axis=0)

        return 1. - tf.reduce_mean(dice_score)

    def dice_score_from_logits(self, one_hot, logits):
        """
        Dice coefficient (F1 score) is between 0 and 1.
        :param labels: one hot encoding of target (num_samples, num_classes)
        :param logits: output of network (num_samples, num_classes)
        :return: Dice score by each class
        """
        probs = tf.nn.softmax(logits)

        intersect = tf.reduce_sum(probs * one_hot, axis=[1, 2, 3])
        denominator = tf.reduce_sum(probs, axis=[1, 2, 3]) + tf.reduce_sum(one_hot, axis=[1, 2, 3])

        dice_score = tf.reduce_mean(2. * intersect / (denominator + 1e-6), axis=0)

        return dice_score

    def dice_loss(self, one_hot, logits):
        return 1. - tf.reduce_mean(self.dice_score_from_logits(one_hot, logits))

    def cross_entropy(self, onehot, logits):
        ce = tf.nn.softmax_cross_entropy_with_logits(onehot, logits, axis=-1)
        return tf.reduce_mean(ce)

    def weighted_cross_entropy(self, one_hot, logits):
        w = tf.reduce_sum(one_hot, axis=[0, 1, 2, 3])
        w = 1 / (w + 1e-6)
        w = w / tf.reduce_sum(w)  # Normalize weights
        # w = tf.constant([0.01, 3., 2., 2., 20., 3., 20., 20., 4., 5.])

        ce = tf.nn.softmax_cross_entropy_with_logits(one_hot, logits, axis=-1)
        weights = tf.reduce_sum(w * one_hot, axis=-1)

        return tf.reduce_mean(weights * ce)
