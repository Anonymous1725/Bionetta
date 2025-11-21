from typing_extensions import override
import tensorflow as tf


class ShadowTripletLoss(tf.keras.losses.Loss):
    """
    ShadowTripletLoss loss for contrastive learning using an anchor, positive pair, and negatives
    with magnitudes between projections pairs.

    This class implements $L_{shadow}$(1):

    $$
                          a . p                a . p
    L_{shadow} = || |a| - _____ || - || |a| -  _____ ||
                           |a|                  |a|
    $$

    [1]: https://arxiv.org/pdf/2311.14012
    """

    def __init__(
        self,
        embedding_size: int,
        margin: float = 1.0,
        name: str = "shadow_triplet_loss",
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

    @override
    def call(self, y_true, y_pred):
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

        anchor_norm = tf.reduce_sum(tf.square(anchor), axis=1)

        anchor_positive_product = tf.divide(
            tf.reduce_sum(tf.multiply(anchor, positive), axis=1), anchor_norm
        )
        anchor_negative_product = tf.divide(
            tf.reduce_sum(tf.multiply(anchor, negative), axis=1), anchor_norm
        )

        delta_plus = tf.abs(anchor_norm - anchor_positive_product)
        delta_minus = tf.abs(anchor_norm - anchor_negative_product)

        loss = tf.reduce_mean(tf.maximum(delta_plus - delta_minus + self.margin, 0.0))

        return loss
