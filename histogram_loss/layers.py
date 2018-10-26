from typing import List, Dict, Any

import tensorflow as tf


class Block35(tf.keras.layers.Layer):
  """Inception ResNet 35x35 block layer.
  
  This layer creates a layer to encapsulate a ResNet 35x35 block.

  Args:
    scale:
    activation: Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: `a(x) = x`).
  """

  def __init__(self,
               scale: float = 1.0,
               activation=tf.nn.relu,
               **kwargs: Dict[str, Any]) -> None:
    self._scale = scale
    self._activation = activation
    self._branches = []  # type: List[List[tf.keras.layers.Layer]]

    # branch 0
    branch0 = []  # tyep: List[tf.keras.layers.Layer]
    branch0_0 = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch0.append(branch0_0)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch0.append(batch_norm)
    self._branches.append(branch0)

    # branch 1
    branch1 = []  # tyep: List[tf.keras.layers.Layer]
    branch1_0 = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch1.append(branch1_0)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch1.append(batch_norm)

    branch1_1 = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch1.append(branch1_1)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch1.append(batch_norm)
    self._branches.append(branch1)

    # branch 2
    branch2 = []  # tyep: List[tf.keras.layers.Layer]
    branch2_0 = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch2.append(branch2_0)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch2.append(batch_norm)

    branch2_1 = tf.keras.layers.Conv2D(
        filters=48,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch2.append(branch2_1)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch2.append(batch_norm)

    branch2_2 = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch2.append(branch2_2)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch2.append(batch_norm)
    self._branches.append(branch2)

    # up
    self._up = tf.keras.layers.Conv2D(
        filters=128, kernel_size=1, strides=1, padding='same', activation=None)

    super(Block35, self).__init__(**kwargs)

  def build(self, input_shape: tf.TensorShape) -> None:
    # build each layer according to the
    # output shape of the previous layer
    for branch in self._branches:
      branch_input_shape = input_shape
      for layer in branch:
        layer.build(branch_input_shape)
        branch_input_shape = layer.compute_output_shape(branch_input_shape)

    super(Block35, self).build(input_shape)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    # forward through branches
    branch_outputs = []
    for branch in self._branches:
      branch_output = inputs
      for layer in branch:
        branch_output = layer.call(branch_output)
      branch_outputs.append(branch_output)

    # stack branches
    mixed = tf.concat(branch_outputs, 3)
    up_out = self._up.call(mixed)

    # residual connection
    inputs += self._scale * up_out

    # apply activation function, if any
    if self._activation:
      inputs = self._activation(inputs)

    return inputs

  def compute_output_shape(self,
                           input_shape: tf.TensorShape) -> tf.TensorShape:
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = 128
    return tf.TensorShape(shape)

  def get_config(self) -> Dict[str, Any]:
    base_config = super(Block35, self).get_config()
    base_config['scale'] = self._scale
    base_config['activation'] = self._activation
    return base_config

  @classmethod
  def from_config(cls, config: Dict[str, Any]):
    return cls(**config)
