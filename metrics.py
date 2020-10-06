import tensorflow as tf


class MetricsManager:

    def __init__(self):
        pass

    def tversky_loss(self, one_hot, logits):
        alpha = 0.5
        beta = 0.5

        probs = tf.nn.softmax(logits)

        ones = tf.ones_like(one_hot)
        p0 = probs
        p1 = ones - probs
        g0 = one_hot
        g1 = ones - one_hot

        numerator = tf.reduce_sum(p0 * g0, axis=[0, 1, 2, 3])
        denominator = numerator + alpha * tf.reduce_sum(p0 * g1, axis=[0, 1, 2, 3]) + beta * tf.reduce_sum(p1 * g0, axis=[0, 1, 2, 3])

        T = tf.reduce_sum(numerator / denominator)

        return 1. - T

    def generalize_dice_loss(self, one_hot, logits):
        w = tf.reduce_sum(one_hot, axis=[0, 1, 2, 3])
        w = 1 / (w ** 2 + 0.000001)

        # Dice coefficient
        probs = tf.nn.softmax(logits)
        numerator = w * tf.reduce_sum(probs * one_hot, axis=[0, 1, 2, 3])
        numerator = tf.reduce_sum(numerator)

        denominator = w * tf.reduce_sum(probs + one_hot, axis=[0, 1, 2, 3])
        denominator = tf.reduce_sum(denominator)

        loss = 1. - (2. * numerator / denominator)

        return loss

    def dice_score_from_logits(self, one_hot, logits):
        """
        Dice coefficient (F1 score) is between 0 and 1.
        :param labels: one hot encoding of target (num_samples, num_classes)
        :param logits: output of network (num_samples, num_classes)
        :return: Dice score by each class
        """
        probs = tf.nn.softmax(logits)

        intersect = tf.reduce_sum(probs * one_hot, axis=[0, 1, 2, 3])
        denominator = tf.reduce_sum(probs + one_hot, axis=[0, 1, 2, 3])

        dice_score = 2. * intersect / (denominator + 1e-6)

        return dice_score

    def weighted_cross_entropy(self, onehot_labels, logits):
        ce = tf.nn.softmax_cross_entropy_with_logits(onehot_labels, logits, axis=-1)

        # class_weights = tf.constant([0.01, 3., 2., 2., 20., 3., 12., 4., 5.])
        # weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)

        # return tf.reduce_mean(weights * ce)
        return tf.reduce_mean(ce)
