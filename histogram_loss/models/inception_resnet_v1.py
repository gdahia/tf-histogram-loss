from typing import List

import tensorflow as tf

from histogram_loss import layers
from histogram_loss.loss import histogram_loss


class InceptionResNetV1(tf.keras.Model):
  """Inception ResNet v1 model.

  This class implements the Inception ResNet v1 model, described in:
    ["Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning". C. Szegedy, S. Ioffe, V. Vanhoucke, A. Alemi](https://arxiv.org/abs/1602.07261)
  as a tf.keras.Model.

  Args:
    bottleneck_units: number of units of the fully connected layer that generates descriptors or logits.
    dropout_rate: rate to drop units in bottleneck layer input.
  """

  def __init__(self, bottleneck_units: int, dropout_rate: float = 0.2) -> None:
    super(InceptionResNetV1, self).__init__(name='inception_resnet_v1')

    self._bottleneck_units = bottleneck_units
    self._layers_ = []  # type: List[tf.keras.layers.Layer]
    self._training_kws = []  # type: List[bool]

    # 149 x 149 x 32
    layer = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=2,
        padding='valid',
        activation=tf.nn.relu)
    self._layers_.append(layer)
    self._training_kws.append(False)

    layer = tf.keras.layers.BatchNormalization()
    self._layers_.append(layer)
    self._training_kws.append(True)

    # 147 x 147 x 32
    layer = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding='valid',
        activation=tf.nn.relu)
    self._layers_.append(layer)
    self._training_kws.append(False)

    layer = tf.keras.layers.BatchNormalization()
    self._layers_.append(layer)
    self._training_kws.append(True)

    # 147 x 147 x 64
    layer = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding='valid',
        activation=tf.nn.relu)
    self._layers_.append(layer)
    self._training_kws.append(False)

    layer = tf.keras.layers.BatchNormalization()
    self._layers_.append(layer)
    self._training_kws.append(True)

    # 73 x 73 x 64
    layer = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid')
    self._layers_.append(layer)
    self._training_kws.append(False)

    # 73 x 73 x 80
    layer = tf.keras.layers.Conv2D(
        filters=80,
        kernel_size=1,
        strides=1,
        padding='valid',
        activation=tf.nn.relu)
    self._layers_.append(layer)
    self._training_kws.append(False)

    layer = tf.keras.layers.BatchNormalization()
    self._layers_.append(layer)
    self._training_kws.append(True)

    # 71 x 71 x 192
    layer = tf.keras.layers.Conv2D(
        filters=192,
        kernel_size=3,
        strides=1,
        padding='valid',
        activation=tf.nn.relu)
    self._layers_.append(layer)
    self._training_kws.append(False)

    layer = tf.keras.layers.BatchNormalization()
    self._layers_.append(layer)
    self._training_kws.append(True)

    # 35 x 35 x 256
    layer = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=3,
        strides=2,
        padding='valid',
        activation=tf.nn.relu)
    self._layers_.append(layer)
    self._training_kws.append(False)

    layer = tf.keras.layers.BatchNormalization()
    self._layers_.append(layer)
    self._training_kws.append(True)

    # 5 x Block35
    self._layers_.extend([layers.Block35(scale=0.17) for _ in range(5)])
    self._training_kws.extend([True for _ in range(5)])

    # ReductionA
    self._layers_.append(layers.ReductionA())
    self._training_kws.append(True)

    # 10 x Block17
    self._layers_.extend([layers.Block17(scale=0.10) for _ in range(10)])
    self._training_kws.extend([True for _ in range(10)])

    # ReductionB
    self._layers_.append(layers.ReductionB())
    self._training_kws.append(True)

    # 5 x Block8
    self._layers_.extend([layers.Block8(scale=0.20) for _ in range(5)])
    self._training_kws.extend([True for _ in range(5)])

    self._layers_.append(layers.Block8(activation=None))
    self._training_kws.append(True)

    # dropout
    self._dropout = tf.keras.layers.Dropout(dropout_rate)

    # bottleneck layer
    self._bottleneck = tf.keras.layers.Dense(
        units=bottleneck_units, activation=None)

    self._bottleneck_batchnorm = tf.keras.layers.BatchNormalization()

  def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
    """Performs forward propagation/inference on the model.

    Args:
      inputs: a `tf.Tensor` with the inputs in shape `[batch_size, height, width, channels]`.
      training: Either a Python boolean, or a TensorFlow boolean scalar tensor (e.g. a placeholder). Whether to perform forward propagation in training mode (use batch statistics in batch normalization, apply dropout) or in inference mode (use the running statiscs for batch normalization, return the input untouched).

    Returns:
      norm_outputs: a `tf.Tensor` matrix of shape `[batch_size, bottleneck_units]` with the L2 normalized descriptors of the given inputs.
      outputs: a `tf.Tensor` matrix of shape `[batch_size, bottleneck_units]` with the outputs of the given inputs.
    """
    # common layers
    outputs = tf.convert_to_tensor(inputs)
    for layer, training_kw in zip(self._layers_, self._training_kws):
      if training_kw:
        outputs = layer(outputs, training=training)
      else:
        outputs = layer(outputs)

    # adaptive pooling
    outputs = tf.reduce_mean(outputs, axis=[1, 2], keepdims=False)

    # dropout
    outputs = self._dropout(outputs, training=training)

    # bottleneck layer
    outputs = self._bottleneck(outputs)
    outputs = self._bottleneck_batchnorm(outputs, training=training)

    # l2 normalize
    norm_outputs = tf.nn.l2_normalize(outputs, axis=-1)

    return norm_outputs, outputs

  def compute_output_shape(self, input_shape: tf.TensorShape):
    # simulate forward propagation for output shape
    shape = tf.TensorShape(input_shape)
    for layer in self._layers_:
      shape = layer.compute_output_shape(shape)

    # dynamic average pooling
    shape = tf.TensorShape(shape).as_list()
    shape = tf.TensorShape([shape[0], shape[-1]])

    # last layer
    shape = self._dropout.compute_output_shape(shape)
    shape = self._bottleneck.compute_output_shape(shape)
    shape = self._bottleneck_batchnorm.compute_output_shape(shape)

    return tf.TensorShape(shape)

  def histogram_loss(self, outputs: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """Computes the histogram loss of `outputs` with given `labels`.

    More details can be found in the documentation for `histogram_loss.loss.histogram_loss`.

    Args:
      outputs: a `tf.Tensor` matrix of shape `[batch_size, desc_dims]` containing the L2 normalized descriptors output by the model.
      labels: a `tf.Tensor` array with `batch_size` elements corresponding to the labels for the given descriptors.

    Returns:
      the histogram loss.
    """
    return histogram_loss(descriptors=outputs, labels=labels)

  def triplet_loss(self, outputs: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """Computes the triplet loss with semi-hard negative mining.

    More details can be found in the documentation for `tf.contrib.losses.metric_learning.triplet_semihard_loss`.

    Args:
      outputs: a `tf.Tensor` matrix of shape `[batch_size, desc_dims]` containing the L2 normalized descriptors output by the model.
      labels: a `tf.Tensor` array with `batch_size` elements corresponding to the labels for the given descriptors.

    Returns:
      the triplet loss.
    """
    labels = tf.reshape(labels, [-1])

    return tf.contrib.losses.metric_learning.triplet_semihard_loss(
        embeddings=outputs, labels=labels)

  def softmax_loss(self, outputs: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """
    """

  def train(self, loss: tf.Tensor, learning_rate: float):
    """
    """

    optimizer = tf.train.AdamOptimizer(learning_rate)

    return optimizer.minimize(loss)
