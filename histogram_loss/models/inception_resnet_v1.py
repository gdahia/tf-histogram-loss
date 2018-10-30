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

    # create saver
    self._saver = tf.train.Saver(var_list=self._variables)
