"""
Triplet Loss Function implementation with the global
orthogonal regularization (GOR) for contrastive learning.
"""

import tensorflow as tf


class TripletGORLoss(tf.keras.losses.Loss):
    """
    TripletLoss loss for contrastive learning using an anchor, positive pair,
    and negatives. In addition to the triplet loss, the global orthogonal
    regularization (GOR) is applied to the embeddings.

    Args:
        margin(float): Margin value for the triplet loss.
    """

    def __init__(
        self,
        embedding_size: int,
        margin: float = 0.3,
        gor_weight: float = 0.1,
        name: str = "triplet_gor_loss",
    ) -> None:
        """
        Triplet Loss Function initialization.

        Args:
            embedding_size (int): Size of the embedding space.
            margin (float): Margin value for the triplet loss.
            gor_weight (float): Weight of the GOR regularization.
            name (str): Name of the loss function.
        """

        super().__init__(name=name)
        self.n = embedding_size
        self.margin = margin
        self.alpha = gor_weight

    def triplet_loss(
        self, anchor: tf.Tensor, positive: tf.Tensor, negative: tf.Tensor
    ) -> tf.Tensor:
        """
        Computes the triplet loss.

        Args:
            anchor (tf.Tensor): Anchor embeddings in shape (batch_size, embedding_size).
            positive (tf.Tensor): Positive embeddings in shape (batch_size, embedding_size).
            negative (tf.Tensor): Negative embeddings in shape (batch_size, embedding_size).

        Returns:
            tf.Tensor: Scalar triplet loss value.
        """

        positive_distances = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        negative_distances = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        basic_loss = positive_distances - negative_distances + self.margin

        triplet_loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
        return triplet_loss

    def gor_loss(self, anchor: tf.Tensor, negative: tf.Tensor) -> tf.Tensor:
        """
        Computes the GOR regularization loss.

        Args:
            anchor (tf.Tensor): Anchor embeddings in shape (batch_size, embedding_size).
            negative (tf.Tensor): Negative embeddings in shape (batch_size, embedding_size).

        Returns:
            tf.Tensor: Scalar GOR regularization loss value.
        """

        mean = tf.reduce_sum(anchor * negative, axis=1)  # (batch_size, 1)
        mean = tf.reduce_mean(mean, axis=0)  # (1) --- the result is a scalar

        moment = tf.reduce_sum(anchor * negative, axis=1)  # (batch_size, 1)
        moment = tf.square(moment)  # (batch_size, 1)
        moment = tf.reduce_mean(moment, axis=0)  # (1) --- the result is a scalar

        m1 = tf.square(mean)
        m2 = tf.math.maximum(0.0, moment - 1.0 / self.n)

        return m1 + m2

    def call(self, _, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes the Triplet Loss with GOR regularization.

        Args:
            y_true: Not used (placeholder for compatibility).
            y_pred: Tuple of (anchor, positive, negatives) tensors.

        Returns:
            tf.Tensor: Scalar loss value.
        """

        assert (
            y_pred.shape[1] % 3 == 0
        ), f"Expected 3 times the embedding size, got {y_pred.shape[1]}"
        assert (
            y_pred.shape[1] // 3 == self.n
        ), f"Expected 3 times the embedding size, got {y_pred.shape[1]}"

        # Extract the embeddings
        anchor = y_pred[:, : self.n]
        positive = y_pred[:, self.n : (2 * self.n)]
        negative = y_pred[:, (2 * self.n) :]

        loss = self.triplet_loss(
            anchor, positive, negative
        ) + self.alpha * self.gor_loss(anchor, negative)
        return loss
