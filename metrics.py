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

    def generalized_dice_score(self, y_true, logits, eps=1e-6):
        """
        :param y_true: [b, h, w, d, classes]
        :param logits: [b, h, w, classes]
        :param eps:
        :return:
        """
        # [b, h, w, d, classes]
        pred_tensor = tf.nn.softmax(logits)
        y_true_shape = tf.shape(y_true)

        # [b, h*w*d, classes]
        y_true = tf.reshape(y_true, [-1, y_true_shape[1] * y_true_shape[2] * y_true_shape[3], y_true_shape[4]])
        y_pred = tf.reshape(pred_tensor, [-1, y_true_shape[1] * y_true_shape[2] * y_true_shape[3], y_true_shape[4]])

        # [b, classes]
        # count how many of each class are present in each image, if there are zero, then assign them a fixed weight of eps
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

    def weighted_cross_entropy(self, onehot_labels, logits):
        """# your class weights
        class_weights = tf.constant([[1.0, 2.0, 3.0]])
        # deduce weights for batch samples based on their true label
        weights = tf.reduce_sum(class_weights * onehot_labels, axis=1)
        # compute your (unweighted) softmax cross entropy loss
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(onehot_labels, logits)
        # apply the weights, relying on broadcasting of the multiplication
        weighted_losses = unweighted_losses * weights
        # reduce the result to get your final loss
        loss = tf.reduce_mean(weighted_losses)"""

        weights = [0.01, 3., 2., 2., 20., 3., 12., 4.]
        ce = tf.nn.softmax_cross_entropy_with_logits(onehot_labels, logits, axis=-1)

        return tf.reduce_mean(ce)
