import tensorflow as tf


class MetricsManager:

    def __init__(self):
        pass

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
