from typing import List, Dict, Any
from abc import ABC, abstractmethod

import tensorflow as tf


class InceptionResNetBlock(tf.keras.layers.Layer, ABC):
  """Base Inception ResNet block keras layer."""

  def build(self, input_shape: tf.TensorShape) -> None:
    # build each layer of each branch according
    # to the output shape of the previous layer
    for branch in self._branches:
      branch_input_shape = input_shape
      for layer in branch:
        layer.build(branch_input_shape)
        branch_input_shape = layer.compute_output_shape(branch_input_shape)

    # build up
    self._up = tf.keras.layers.Conv2D(
        filters=input_shape[3],
        kernel_size=1,
        strides=1,
        padding='same',
        activation=None)
    self._up.build(input_shape)

    super(InceptionResNetBlock, self).build(input_shape)

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
    return tf.TensorShape(input_shape)

  def get_config(self) -> Dict[str, Any]:
    base_config = super(InceptionResNetBlock, self).get_config()
    base_config['scale'] = self._scale
    base_config['activation'] = self._activation
    return base_config

  @classmethod
  def from_config(cls, config: Dict[str, Any]):
    return cls(**config)


class Block35(InceptionResNetBlock):
  """Inception ResNet 35x35 block layer.
  
  This layer encapsulates a ResNet 35x35 block.

  Args:
    scale: Activation scaling constant.
    activation: Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: `a(x) = x`).
  """

  def __init__(self,
               scale: float = 1.0,
               activation=tf.nn.relu,
               **kwargs: Dict[str, Any]) -> None:
    self._scale = scale
    self._activation = activation
    self._branches = []  # type: List[List[tf.keras.layers.Layer]]
    super(InceptionResNetBlock, self).__init__(**kwargs)

  def build(self, input_shape: tf.TensorShape) -> None:
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

    # build branches, up, and internal keras layer structure
    super(Block35, self).build(input_shape)


class Block17(InceptionResNetBlock):
  """Inception ResNet 17x17 block layer.
  
  This layer encapsulates a ResNet 17x17 block.

  Args:
    scale: Activation scaling constant.
    activation: Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: `a(x) = x`).
  """

  def __init__(self,
               scale: float = 1.0,
               activation=tf.nn.relu,
               **kwargs: Dict[str, Any]) -> None:
    self._scale = scale
    self._activation = activation
    self._branches = []  # type: List[List[tf.keras.layers.Layer]]
    super(InceptionResNetBlock, self).__init__(**kwargs)

  def build(self, input_shape: tf.TensorShape) -> None:
    # branch 0
    branch0 = []  # tyep: List[tf.keras.layers.Layer]
    branch0_0 = tf.keras.layers.Conv2D(
        filters=128,
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
        filters=128,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch1.append(branch1_0)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch1.append(batch_norm)

    branch1_1 = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=[1, 7],
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch1.append(branch1_1)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch1.append(batch_norm)
    self._branches.append(branch1)

    branch1_2 = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=[7, 1],
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch1.append(branch1_2)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch1.append(batch_norm)
    self._branches.append(branch1)

    # build branches, up, and internal keras layer structure
    super(Block17, self).build(input_shape)


class Block8(InceptionResNetBlock):
  """Inception ResNet 8x8 block layer.
  
  This layer encapsulates a ResNet 8x8 block.

  Args:
    scale: Activation scaling constant.
    activation: Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: `a(x) = x`).
  """

  def __init__(self,
               scale: float = 1.0,
               activation=tf.nn.relu,
               **kwargs: Dict[str, Any]) -> None:
    self._scale = scale
    self._activation = activation
    self._branches = []  # type: List[List[tf.keras.layers.Layer]]
    super(InceptionResNetBlock, self).__init__(**kwargs)

  def build(self, input_shape: tf.TensorShape) -> None:
    # branch 0
    branch0 = []  # tyep: List[tf.keras.layers.Layer]
    branch0_0 = tf.keras.layers.Conv2D(
        filters=192,
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
        filters=192,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch1.append(branch1_0)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch1.append(batch_norm)

    branch1_1 = tf.keras.layers.Conv2D(
        filters=192,
        kernel_size=[1, 3],
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch1.append(branch1_1)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch1.append(batch_norm)
    self._branches.append(branch1)

    branch1_2 = tf.keras.layers.Conv2D(
        filters=192,
        kernel_size=[3, 1],
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch1.append(branch1_2)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch1.append(batch_norm)
    self._branches.append(branch1)

    # build branches, up, and internal keras layer structure
    super(Block8, self).build(input_shape)


class InceptionResNetReduction(tf.keras.layers.Layer, ABC):
  """Base Inception ResNet reduction keras layer."""

  def __init__(self, **kwargs: Dict[str, Any]) -> None:
    self._branches = []  # type: List[List[tf.keras.layers.Layer]]
    super(InceptionResNetReduction, self).__init__(**kwargs)

  def build(self, input_shape: tf.TensorShape) -> None:
    # build each layer of each branch according
    # to the output shape of the previous layer
    for branch in self._branches:
      branch_input_shape = input_shape
      for layer in branch:
        layer.build(branch_input_shape)
        branch_input_shape = layer.compute_output_shape(branch_input_shape)

    super(InceptionResNetReduction, self).build(input_shape)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    # forward through branches
    branch_outputs = []
    for branch in self._branches:
      branch_output = inputs
      for layer in branch:
        branch_output = layer.call(branch_output)
      branch_outputs.append(branch_output)

    # stack branches
    out = tf.concat(branch_outputs, 3)

    return out

  def compute_output_shape(self,
                           input_shape: tf.TensorShape) -> tf.TensorShape:
    shape = tf.TensorShape(input_shape).as_list()
    shape[1] = shape[1] // 2 - 1
    shape[2] = shape[2] // 2 - 1
    shape[3] += self.get_num_filters()
    return tf.TensorShape(shape)

  @abstractmethod
  def get_num_filters(self):
    """Number of added filters in the reduction."""

  def get_config(self) -> Dict[str, Any]:
    return super(InceptionResNetReduction, self).get_config()

  @classmethod
  def from_config(cls, config: Dict[str, Any]):
    return cls(**config)
