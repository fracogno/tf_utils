import tensorflow as tf


class MetricsManager:

    def __init__(self):
        pass

    def dice_score_from_logits(self, labels, logits, num_classes):
        """
        Dice coefficient (F1 score) is between 0 and 1.
        :param labels: one hot encoding of target (num_samples, num_classes)
        :param logits: output of network (num_samples, num_classes)
        :return: Dice score for each class
        """
        probs = tf.nn.softmax(logits)
        one_hot = tf.keras.utils.to_categorical(labels, num_classes)

        intersect = tf.reduce_sum(probs * one_hot, axis=[0, 1, 2, 3])
        denominator = tf.reduce_sum(probs + one_hot, axis=[0, 1, 2, 3])

        dice_score = 2. * intersect / (denominator + 1e-6)

        return dice_score
