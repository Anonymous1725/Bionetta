"""
In case we want to train the classification network,
we need to project the embeddings to the output
space of the classification network. For that reason,
we are going to use the matrix with orthogonal columns,
as suggested by numerous papers on the subject.
"""

from typing import Tuple, Dict

import tensorflow as tf
from keras import regularizers


class ClassProjectionLayer(tf.keras.layers.Layer):
    """
    Class Projection Layer. This layer is used to project the embeddings to the
    output space of the classification network. A common layer in Norm models such
    as ArcFace, CosFace, and SphereFace.
    """

    def __init__(
        self,
        num_classes: int,
        kernel_regularizer: regularizers.Regularizer | None = regularizers.l2(1e-4),
        **kwargs,
    ) -> None:
        """
        Initializes the Class Projection Layer.

        Args:
            - num_classes (int): Number of classes in the dataset.
            - kernel_regularizer (l2, optional): Regularizer for the kernel. Defaults to l2(1e-4).
        """

        super(ClassProjectionLayer, self).__init__(**kwargs)

        self.num_classes = num_classes
        self.kernel_regularizer = kernel_regularizer
        self.input_spec = tf.keras.layers.InputSpec(ndim=2)


    def build(self, input_shape: Tuple) -> None:
        """
        Builds the layer. This method is called once the input shape is known.
        """

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Class Projection layer "
                "should be defined. Found None. "
                f"Full input shape received: {input_shape}"
            )

        self.kernel = self.add_weight(
            "kernel",
            shape=[last_dim, self.num_classes],
            initializer="orthogonal",
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        super(ClassProjectionLayer, self).build(input_shape)


    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass of the layer.
        """

        # Normalize the columns of the kernel
        normed_kernel = tf.nn.l2_normalize(self.kernel, axis=0, name="normed_kernel")
        return tf.matmul(inputs, normed_kernel, name="logits")


    def compute_output_shape(self, input_shape: Tuple) -> tf.TensorShape:
        """
        Computes the output shape of the layer.
        """

        input_shape = tf.TensorShape(input_shape)
        return tf.TensorShape([input_shape[0], self.num_classes])


    def get_config(self) -> Dict:
        """
        Returns the configuration of the layer.
        """

        config = super(ClassProjectionLayer, self).get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "kernel_regularizer": self.kernel_regularizer,
            }
        )

        return config
