from typing import List, Dict, Any
from abc import ABC, abstractmethod

import tensorflow as tf


class InceptionResNetBlock(tf.keras.layers.Layer, ABC):
  """Base Inception ResNet block keras layer.

  This layer encapsulates a general Inception ResNet block.

  Args:
    scale: Activation scaling constant. Instead of doing the traditional residual connection `F(x) + x`, does `scale * F(x) + x`.
    activation: Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: `a(x) = x`).
  """

  def __init__(self,
               scale: float = 1.0,
               activation=tf.nn.relu,
               **kwargs: Dict[str, Any]) -> None:
    self._scale = scale
    self._activation = activation
    self._branches = []  # type: List[List[tf.keras.layers.Layer]]
    self._training_kws = []  # type: List[List[bool]]
    super(InceptionResNetBlock, self).__init__(**kwargs)

  def build(self, input_shape: tf.TensorShape) -> None:
    # build each layer of each branch according
    # to the output shape of the previous layer
    up_input_filters = 0
    for branch in self._branches:
      branch_input_shape = input_shape
      for layer in branch:
        layer.build(branch_input_shape)
        branch_input_shape = layer.compute_output_shape(branch_input_shape)
      up_input_filters += branch_input_shape.as_list()[-1]

    # build up
    self._up = tf.keras.layers.Conv2D(
        filters=int(input_shape[3]),
        kernel_size=1,
        strides=1,
        padding='same',
        activation=None)

    up_input_shape = branch_input_shape.as_list()
    up_input_shape[-1] = up_input_filters
    self._up.build(up_input_shape)

    super(InceptionResNetBlock, self).build(input_shape)

  def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
    # forward through branches
    branch_outputs = []
    for branch, training_kws in zip(self._branches, self._training_kws):
      branch_output = inputs
      for layer, training_kw in zip(branch, training_kws):
        if training_kw:
          branch_output = layer.call(branch_output, training=training)
        else:
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
  """Inception ResNet 35x35 block layer."""

  def build(self, input_shape: tf.TensorShape) -> None:
    # branch 0
    branch0 = []  # type: List[tf.keras.layers.Layer]
    branch0_training_kws = []  # type: List[bool]

    branch0_0 = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch0.append(branch0_0)
    branch0_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch0.append(batch_norm)
    branch0_training_kws.append(True)

    assert 2 == len(branch0) == len(branch0_training_kws)
    self._branches.append(branch0)
    self._training_kws.append(branch0_training_kws)

    # branch 1
    branch1 = []  # type: List[tf.keras.layers.Layer]
    branch1_training_kws = []  # type: List[bool]

    branch1_0 = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch1.append(branch1_0)
    branch1_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch1.append(batch_norm)
    branch1_training_kws.append(True)

    branch1_1 = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch1.append(branch1_1)
    branch1_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch1.append(batch_norm)
    branch1_training_kws.append(True)

    assert 4 == len(branch1) == len(branch1_training_kws)
    self._branches.append(branch1)
    self._training_kws.append(branch1_training_kws)

    # branch 2
    branch2 = []  # type: List[tf.keras.layers.Layer]
    branch2_training_kws = []  # type: List[bool]

    branch2_0 = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch2.append(branch2_0)
    branch2_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch2.append(batch_norm)
    branch2_training_kws.append(True)

    branch2_1 = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch2.append(branch2_1)
    branch2_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch2.append(batch_norm)
    branch2_training_kws.append(True)

    branch2_2 = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch2.append(branch2_2)
    branch2_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch2.append(batch_norm)
    branch2_training_kws.append(True)

    assert 6 == len(branch2) == len(branch2_training_kws)
    self._branches.append(branch2)
    self._training_kws.append(branch2_training_kws)

    # build branches, up, and internal keras layer structure
    assert 3 == len(self._branches) == len(self._training_kws)
    super(Block35, self).build(input_shape)


class Block17(InceptionResNetBlock):
  """Inception ResNet 17x17 block layer."""

  def build(self, input_shape: tf.TensorShape) -> None:
    # branch 0
    branch0 = []  # type: List[tf.keras.layers.Layer]
    branch0_training_kws = []  # type: List[bool]

    branch0_0 = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch0.append(branch0_0)
    branch0_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch0.append(batch_norm)
    branch0_training_kws.append(True)

    assert 2 == len(branch0) == len(branch0_training_kws)
    self._branches.append(branch0)
    self._training_kws.append(branch0_training_kws)

    # branch 1
    branch1 = []  # type: List[tf.keras.layers.Layer]
    branch1_training_kws = []  # type: List[bool]

    branch1_0 = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch1.append(branch1_0)
    branch1_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch1.append(batch_norm)
    branch1_training_kws.append(True)

    branch1_1 = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=[1, 7],
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch1.append(branch1_1)
    branch1_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch1.append(batch_norm)
    branch1_training_kws.append(True)

    branch1_2 = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=[7, 1],
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch1.append(branch1_2)
    branch1_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch1.append(batch_norm)
    branch1_training_kws.append(True)

    assert 6 == len(branch1) == len(branch1_training_kws)
    self._branches.append(branch1)
    self._training_kws.append(branch1_training_kws)

    # build branches, up, and internal keras layer structure
    assert 2 == len(self._branches) == len(self._training_kws)
    super(Block17, self).build(input_shape)


class Block8(InceptionResNetBlock):
  """Inception ResNet 8x8 block layer."""

  def build(self, input_shape: tf.TensorShape) -> None:
    # branch 0
    branch0 = []  # type: List[tf.keras.layers.Layer]
    branch0_training_kws = []  # type: List[bool]

    branch0_0 = tf.keras.layers.Conv2D(
        filters=192,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch0.append(branch0_0)
    branch0_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch0.append(batch_norm)
    branch0_training_kws.append(True)

    assert 2 == len(branch0) == len(branch0_training_kws)
    self._branches.append(branch0)
    self._training_kws.append(branch0_training_kws)

    # branch 1
    branch1 = []  # type: List[tf.keras.layers.Layer]
    branch1_training_kws = []  # type: List[bool]

    branch1_0 = tf.keras.layers.Conv2D(
        filters=192,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch1.append(branch1_0)
    branch1_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch1.append(batch_norm)
    branch1_training_kws.append(True)

    branch1_1 = tf.keras.layers.Conv2D(
        filters=192,
        kernel_size=[1, 3],
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch1.append(branch1_1)
    branch1_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch1.append(batch_norm)
    branch1_training_kws.append(True)

    branch1_2 = tf.keras.layers.Conv2D(
        filters=192,
        kernel_size=[3, 1],
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch1.append(branch1_2)
    branch1_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch1.append(batch_norm)
    branch1_training_kws.append(True)

    assert 6 == len(branch1) == len(branch1_training_kws)
    self._branches.append(branch1)
    self._training_kws.append(branch1_training_kws)

    # build branches, up, and internal keras layer structure
    assert 2 == len(self._branches) == len(self._training_kws)
    super(Block8, self).build(input_shape)


class InceptionResNetReduction(tf.keras.layers.Layer, ABC):
  """Base Inception ResNet reduction keras layer.

  This layer encapsulates a general Inception ResNet reduction.
  """

  def __init__(self, **kwargs: Dict[str, Any]) -> None:
    self._branches = []  # type: List[List[tf.keras.layers.Layer]]
    self._training_kws = []  # type: List[List[bool]]
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

  def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
    # forward through branches
    branch_outputs = []
    for branch, training_kws in zip(self._branches, self._training_kws):
      branch_output = inputs
      for layer, training_kw in zip(branch, training_kws):
        if training_kw:
          branch_output = layer.call(branch_output, training=training)
        else:
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


class ReductionA(InceptionResNetReduction):
  """Inception ResNet reduction A layer."""

  def build(self, input_shape: tf.TensorShape) -> None:
    # branch 0
    branch0 = []  # type: List[tf.keras.layers.Layer]
    branch0_training_kws = []  # type: List[bool]

    branch0_0 = tf.keras.layers.Conv2D(
        filters=384,
        kernel_size=3,
        strides=2,
        padding='valid',
        activation=tf.nn.relu)
    branch0.append(branch0_0)
    branch0_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch0.append(batch_norm)
    branch0_training_kws.append(True)

    assert 2 == len(branch0) == len(branch0_training_kws)
    self._branches.append(branch0)
    self._training_kws.append(branch0_training_kws)

    # branch 1
    branch1 = []  # type: List[tf.keras.layers.Layer]
    branch1_training_kws = []  # type: List[bool]

    branch1_0 = tf.keras.layers.Conv2D(
        filters=192,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch1.append(branch1_0)
    branch1_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch1.append(batch_norm)
    branch1_training_kws.append(True)

    branch1_1 = tf.keras.layers.Conv2D(
        filters=192,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch1.append(branch1_1)
    branch1_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch1.append(batch_norm)
    branch1_training_kws.append(True)

    branch1_2 = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=3,
        strides=2,
        padding='valid',
        activation=tf.nn.relu)
    branch1.append(branch1_2)
    branch1_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch1.append(batch_norm)
    branch1_training_kws.append(True)

    assert 6 == len(branch1) == len(branch1_training_kws)
    self._branches.append(branch1)
    self._training_kws.append(branch1_training_kws)

    # branch 2
    branch2 = []  # type: List[tf.keras.layers.Layer]
    branch2_training_kws = []  # type: List[bool]

    branch2_0 = tf.keras.layers.MaxPool2D(
        pool_size=3, strides=2, padding='valid')
    branch2.append(branch2_0)
    branch2_training_kws.append(False)

    assert 1 == len(branch2) == len(branch2_training_kws)
    self._branches.append(branch2)
    self._training_kws.append(branch2_training_kws)

    # build branches and internal keras layer structure
    assert 3 == len(self._branches) == len(self._training_kws)
    super(ReductionA, self).build(input_shape)

  def get_num_filters(self):
    return 640


class ReductionB(InceptionResNetReduction):
  """Inception ResNet reduction B layer."""

  def build(self, input_shape: tf.TensorShape) -> None:
    # branch 0
    branch0 = []  # type: List[tf.keras.layers.Layer]
    branch0_training_kws = []  # type: List[bool]

    branch0_0 = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch0.append(branch0_0)
    branch0_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch0.append(batch_norm)
    branch0_training_kws.append(True)

    branch0_1 = tf.keras.layers.Conv2D(
        filters=384,
        kernel_size=3,
        strides=2,
        padding='valid',
        activation=tf.nn.relu)
    branch0.append(branch0_1)
    branch0_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch0.append(batch_norm)
    branch0_training_kws.append(True)

    assert 4 == len(branch0) == len(branch0_training_kws)
    self._branches.append(branch0)
    self._training_kws.append(branch0_training_kws)

    # branch 1
    branch1 = []  # type: List[tf.keras.layers.Layer]
    branch1_training_kws = []  # type: List[bool]

    branch1_0 = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch1.append(branch1_0)
    branch1_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch1.append(batch_norm)
    branch1_training_kws.append(True)

    branch1_1 = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=3,
        strides=2,
        padding='valid',
        activation=tf.nn.relu)
    branch1.append(branch1_1)
    branch1_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch1.append(batch_norm)
    branch1_training_kws.append(True)

    assert 4 == len(branch1) == len(branch1_training_kws)
    self._branches.append(branch1)
    self._training_kws.append(branch1_training_kws)

    # branch 2
    branch2 = []  # type: List[tf.keras.layers.Layer]
    branch2_training_kws = []  # type: List[bool]

    branch2_0 = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch2.append(branch2_0)
    branch2_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch2.append(batch_norm)
    branch2_training_kws.append(True)

    branch2_1 = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    branch2.append(branch2_1)
    branch2_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch2.append(batch_norm)
    branch2_training_kws.append(True)

    branch2_2 = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=3,
        strides=2,
        padding='valid',
        activation=tf.nn.relu)
    branch2.append(branch2_2)
    branch2_training_kws.append(False)

    batch_norm = tf.keras.layers.BatchNormalization()
    branch2.append(batch_norm)
    branch2_training_kws.append(True)

    assert 6 == len(branch2) == len(branch2_training_kws)
    self._branches.append(branch2)
    self._training_kws.append(branch2_training_kws)

    # branch 3
    branch3 = []  # type: List[tf.keras.layers.Layer]
    branch3_training_kws = []  # type: List[bool]

    branch3_0 = tf.keras.layers.MaxPool2D(
        pool_size=3, strides=2, padding='valid')
    branch3.append(branch3_0)
    branch3_training_kws.append(False)

    assert 1 == len(branch3) == len(branch3_training_kws)
    self._branches.append(branch3)
    self._training_kws.append(branch3_training_kws)

    # build branches and internal keras layer structure
    assert 4 == len(self._branches) == len(self._training_kws)
    super(ReductionB, self).build(input_shape)

  def get_num_filters(self):
    return 896
