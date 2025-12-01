"""
Implementation of the last layer of the ArcFace model.

Taken from the following source:
https://github.com/yinguobing/arcface/blob/main/losses.py
"""

from typing import Dict

import tensorflow as tf
from math import pi


class ArcFaceLoss(tf.keras.losses.Loss):
    """
    Implementation of the ArcFaceLoss. See original paper for details:
    https://arxiv.org/abs/1801.07698
    """

    def __init__(self, margin: float = 0.5, scale: float = 64.0, **kwargs) -> None:
        """
        Initializes the loss function used in the ArcFace model.
        """

        super(ArcFaceLoss, self).__init__(name="arcloss", **kwargs)

        self.margin = margin  # m parameter in the paper
        self.scale = scale  # s parameter in the paper

        # Indirect Parameters
        self.cos_m = tf.math.cos(margin)  # cosine of the margin
        self.sin_m = tf.math.sin(margin)  # sine of the margin
        self.threshold = tf.math.cos(pi - margin)
        # Safe margin: https://github.com/deepinsight/insightface/issues/108
        self.safe_margin = self.sin_m * margin

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Executes the layer
        """

        cos_theta = y_pred
        sin_theta = tf.math.sqrt(1.0 - tf.math.square(cos_theta))

        cos_theta_margin = tf.where(
            cos_theta > self.threshold,
            cos_theta * self.cos_m - sin_theta * self.sin_m,
            cos_theta - self.safe_margin,
        )

        # Assume labels here had already been converted to one-hot encoding
        mask = y_true
        cos_theta_onehot = cos_theta * mask
        cos_theta_margin_onehot = cos_theta_margin * mask

        # Calculate the final scaled logits
        logits = (cos_theta + cos_theta_margin_onehot - cos_theta_onehot) * self.scale

        losses = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
        return losses

    def get_config(self) -> Dict:
        """
        Returns the configuration of the layer
        """

        config = super(ArcFaceLoss, self).get_config()
        config.update({"margin": self.margin, "scale": self.scale})
        return config
