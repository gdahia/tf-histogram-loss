import pytest
from hypothesis import given, settings, unlimited, strategies as st
from hypothesis.extra import numpy as np_st

import numpy as np
import tensorflow as tf
from tensorflow.python.eager.context import eager_mode

from histogram_loss.models import InceptionResNetV1


class TestInceptionResNetV1:
  @given(
      bottleneck_units=st.integers(1, 4096),
      input_shape=st.tuples(
          st.integers(1), st.integers(71), st.integers(71), st.integers(1, 3)))
  def test_compute_output_shape(self, bottleneck_units, input_shape):
    model = InceptionResNetV1(bottleneck_units)

    output_shape = model.compute_output_shape(input_shape)

    assert len(output_shape) == 2
    assert output_shape == (input_shape[0], bottleneck_units)

  @given(
      bottleneck_units=st.integers(1, 4096),
      input_shape=st.tuples(
          st.integers(1), st.integers(1, 70), st.integers(1, 70),
          st.integers(1, 3)))
  def test_invalid_input_shape(self, bottleneck_units, input_shape):
    model = InceptionResNetV1(bottleneck_units)

    with pytest.raises(ValueError):
      model.compute_output_shape(input_shape)

  @given(
      bottleneck_units=st.integers(1, 4096),
      inputs=np_st.arrays(
          dtype=np.uint8,
          shape=(1, 160, 160, 3),
          elements=st.integers(min_value=0, max_value=255)))
  @settings(timeout=unlimited)
  def test_forward_propagation(self, bottleneck_units, inputs):
    inputs = np.array(inputs, dtype=np.float32) / 255

    model = InceptionResNetV1(bottleneck_units)
    inputs_pl = tf.placeholder(tf.float32, [None, None, None, 3])
    outputs = model(inputs_pl)
    with tf.Session() as sess:
      sess.run(tf.initializers.variables(model.variables))
      sess.run(outputs, feed_dict={inputs_pl: inputs})

  @given(
      bottleneck_units=st.integers(1, 4096),
      inputs=np_st.arrays(
          dtype=np.uint8,
          shape=(1, 160, 160, 3),
          elements=st.integers(min_value=0, max_value=255)))
  @settings(timeout=unlimited)
  def test_eager_forward_propagation(self, bottleneck_units, inputs):
    inputs = np.array(inputs, dtype=np.float32) / 255

    with eager_mode():
      model = InceptionResNetV1(bottleneck_units)
      model(inputs)
