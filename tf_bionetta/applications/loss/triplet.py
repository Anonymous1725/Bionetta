"""
Triplet Loss Function implementation.
"""

import tensorflow as tf


class TripletLoss(tf.keras.losses.Loss):
    """
    TripletLoss loss for contrastive learning using an anchor, positive pair, and negatives.

    Args:
        margin(float): Margin value for the triplet loss.
    """

    def __init__(
        self, embedding_size: int, margin: float = 0.3, name: str = "triplet_loss"
    ) -> None:
        """
        Triplet Loss Function initialization.

        Args:
            embedding_size (int): Size of the embedding space.
            margin (float): Margin value for the triplet loss.
            name (str): Name of the loss function.
        """

        super().__init__(name=name)
        self.n = embedding_size
        self.margin = margin

    def call(self, _, y_pred):
        """
        Computes the Triplet Loss.

        Args:
            y_true: Not used (placeholder for compatibility).
            y_pred: Tuple of (anchor, positive, negatives) tensors.

        Returns:
            tf.Tensor: Scalar loss value.
        """

        anchor = y_pred[:, : self.n]
        positive = y_pred[:, self.n : (2 * self.n)]
        negative = y_pred[:, (2 * self.n) :]

        positive_distances = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        negative_distances = tf.reduce_sum(tf.square(anchor - negative), axis=1)

        basic_loss = positive_distances - negative_distances + self.margin
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

        return loss
