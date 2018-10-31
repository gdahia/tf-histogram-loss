from typing import List
import tensorflow as tf

from histogram_loss import layers
from histogram_loss.loss import histogram_loss


class InceptionResNetV1:
  """Inception ResNet v1 model.

  This class implements the Inception ResNet v1 model, described in:
    ["Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning". C. Szegedy, S. Ioffe, V. Vanhoucke, A. Alemi](https://arxiv.org/abs/1602.07261)

  Args:
    bottleneck_units: number of units of the fully connected layer that generates descriptors or logits.
    dropout_rate: rate to drop units in bottleneck layer input.
  """

  def __init__(self, bottleneck_units: int, dropout_rate: float = 0.2) -> None:
    self._bottleneck_units = bottleneck_units
    self._variables = []  # type: List[tf.Variable]
    self._layers = []  # type: List[tf.keras.layers.Layer]
    self._training_kws = []  # type: List[bool]

    # 149 x 149 x 32
    layer = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=2,
        padding='valid',
        activation=tf.nn.relu)
    self._layers.append(layer)
    self._training_kws.append(False)
    self._variables += layer.variables

    layer = tf.keras.layers.BatchNormalization()
    self._layers.append(layer)
    self._training_kws.append(True)
    self._variables += layer.variables

    # 147 x 147 x 32
    layer = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding='valid',
        activation=tf.nn.relu)
    self._layers.append(layer)
    self._training_kws.append(False)
    self._variables += layer.variables

    layer = tf.keras.layers.BatchNormalization()
    self._layers.append(layer)
    self._training_kws.append(True)
    self._variables += layer.variables

    # 147 x 147 x 64
    layer = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding='valid',
        activation=tf.nn.relu)
    self._layers.append(layer)
    self._training_kws.append(False)
    self._variables += layer.variables

    layer = tf.keras.layers.BatchNormalization()
    self._layers.append(layer)
    self._training_kws.append(True)
    self._variables += layer.variables

    # 73 x 73 x 64
    layer = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid')
    self._layers.append(layer)
    self._training_kws.append(False)
    self._variables += layer.variables

    # 73 x 73 x 80
    layer = tf.keras.layers.Conv2D(
        filters=80,
        kernel_size=1,
        strides=1,
        padding='valid',
        activation=tf.nn.relu)
    self._layers.append(layer)
    self._training_kws.append(False)
    self._variables += layer.variables

    layer = tf.keras.layers.BatchNormalization()
    self._layers.append(layer)
    self._training_kws.append(True)
    self._variables += layer.variables

    # 71 x 71 x 192
    layer = tf.keras.layers.Conv2D(
        filters=192,
        kernel_size=3,
        strides=1,
        padding='valid',
        activation=tf.nn.relu)
    self._layers.append(layer)
    self._training_kws.append(False)
    self._variables += layer.variables

    layer = tf.keras.layers.BatchNormalization()
    self._layers.append(layer)
    self._training_kws.append(True)
    self._variables += layer.variables

    # 35 x 35 x 256
    layer = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=3,
        strides=2,
        padding='valid',
        activation=tf.nn.relu)
    self._training_kws.append(False)
    self._variables += layer.variables

    layer = tf.keras.layers.BatchNormalization()
    self._layers.append(layer)
    self._training_kws.append(True)
    self._variables += layer.variables

    # 5 x Block35
    self._layers += [layers.Block35(scale=0.17) for _ in range(5)]
    self._training_kws += [True for _ in range(5)]
    for layer in self._layers[-5:]:
      self._variables += layer.variables

    # ReductionA
    self._layers.append(layers.ReductionA())
    self._training_kws.append(True)
    self._variables += self._layers[-1].variables

    # 10 x Block17
    self._layers += [layers.Block17(scale=0.10) for _ in range(10)]
    self._training_kws += [True for _ in range(10)]
    for layer in self._layers[-10:]:
      self._variables += layer.variables

    # ReductionB
    self._layers.append(layers.ReductionB())
    self._training_kws.append(True)
    self._variables += self._layers[-1].variables

    # 5 x Block8
    self._layers += [layers.Block8(scale=0.20) for _ in range(5)]
    self._training_kws += [True for _ in range(5)]
    for layer in self._layers[-5:]:
      self._variables += layer.variables

    self._layers.append(layers.Block8(activation=None))
    self._training_kws.append(True)
    self._variables += self._layers[-1].variables

    # dropout
    self._dropout = tf.keras.layers.Dropout(dropout_rate)

    # bottleneck layer
    self._bottleneck = tf.keras.layers.Dense(
        units=bottleneck_units, activation=tf.nn.relu)
    self._variables += self._bottleneck.variables

    self._bottleneck_batchnorm = tf.keras.layers.BatchNormalization()
    self._variables += self._bottleneck_batchnorm.variables

    # create checkpoint saver
    self._checkpoint = tf.train.Checkpoint(variables=self._variables)

  def forward(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
    """Performs forward propagation/inference on the model.

    Args:
      inputs: a `tf.Tensor` with the inputs in shape `[batch_size, height, width, channels]`.
      training: Either a Python boolean, or a TensorFlow boolean scalar tensor (e.g. a placeholder). Whether to perform forward propagation in training mode (use batch statistics in batch normalization, apply dropout) or in inference mode (use the running statiscs for batch normalization, return the input untouched).

    Returns:
      norm_outputs: a `tf.Tensor` matrix of shape `[batch_size, bottleneck_units]` with the L2 normalized descriptors of the given inputs.
      outputs: a `tf.Tensor` matrix of shape `[batch_size, bottleneck_units]` with the outputs of the given inputs.
    """
    # common layers
    outputs = inputs
    for layer, training_kw in zip(self._layers, self._training_kws):
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

  def train(self, loss: tf.Tensor, learning_rate: float):
    """
    """

    optimizer = tf.train.AdamOptimizer(learning_rate)

    return optimizer.minimize(loss)

  def save(self, file_preffix: str, sess: tf.Session = None) -> str:
    """Saves the model variables with the given preffix.

    Args:
      file_prefix: a prefix to use for the checkpoint filenames (`/path/to/directory/and_a_prefix`). 
      session: the session to evaluate variables in. Ignored when executing eagerly. If not provided when graph building, the default session is used.

    Returns:
      the full path to the checkpoint in which the model variables were saved.
    """
    return self._checkpoint.save(file_preffix, sess)

  def restore(self, save_path: str):
    """Restores the model variables with the latest checkpoint in `save_path`.

    Args:
      save_path: refer to the documentation for `tf.train.Checkpoint.restore` for more details.

    Returns:
      a  load status object. Refer to the documentation for `tf.train.Checkpoint.restore` for more details.
    """
    return self._checkpoint.restore(save_path)
