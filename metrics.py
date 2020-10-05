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

    def gen_dice(y_true, y_pred, eps=1e-6):
        """both tensors are [b, h, w, classes] and y_pred is in logit form"""

        # [b, h, w, classes]
        pred_tensor = tf.nn.softmax(y_pred)
        y_true_shape = tf.shape(y_true)

        # [b, h*w, classes]
        y_true = tf.reshape(y_true, [-1, y_true_shape[1] * y_true_shape[2], y_true_shape[3]])
        y_pred = tf.reshape(pred_tensor, [-1, y_true_shape[1] * y_true_shape[2], y_true_shape[3]])

        # [b, classes]
        # count how many of each class are present in
        # each image, if there are zero, then assign
        # them a fixed weight of eps
        counts = tf.reduce_sum(y_true, axis=1)
        weights = 1. / (counts ** 2)
        weights = tf.where(tf.math.is_finite(weights), weights, eps)

        multed = tf.reduce_sum(y_true * y_pred, axis=1)
        summed = tf.reduce_sum(y_true + y_pred, axis=1)

        # [b]
        numerators = tf.reduce_sum(weights * multed, axis=-1)
        denom = tf.reduce_sum(weights * summed, axis=-1)
        dices = 1. - 2. * numerators / denom
        dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))
        return tf.reduce_mean(dices)
