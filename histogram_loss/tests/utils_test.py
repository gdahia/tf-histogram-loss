import pytest
from hypothesis import given, assume, strategies as st
import hypothesis.extra.numpy as np_st

import numpy as np
import tensorflow as tf
from tensorflow.python.eager.context import eager_mode

from histogram_loss import utils


@pytest.fixture(scope='function')
def float_graph():
  input_pl = tf.placeholder(tf.float32, [None, None])
  output = utils.flat_strict_upper(input_pl)
  yield input_pl, output


@pytest.fixture(scope='function')
def int_graph():
  input_pl = tf.placeholder(tf.int64, [None, None])
  output = utils.flat_strict_upper(input_pl)
  yield input_pl, output


class TestUtils:
  @given(st.data())
  def test_eager_flat_strict_upper_int_verification(self, data):
    with eager_mode():
      # sample square matrix
      n = data.draw(st.integers(min_value=3, max_value=10))
      input_ = data.draw(
          np_st.arrays(dtype=np_st.integer_dtypes(), shape=(n, n)))

      output = np.array(utils.flat_strict_upper(input_))

      # verify
      k = 0
      for i in range(n):
        for j in range(i + 1, n):
          assert output[k] == input_[i, j]
          k += 1

  @given(st.data())
  def test_eager_flat_strict_upper_float_verification(self, data):
    with eager_mode():
      # sample square matrix
      n = data.draw(st.integers(min_value=3, max_value=10))
      elements = st.floats(allow_nan=False, allow_infinity=False, width=32)
      input_ = data.draw(
          np_st.arrays(dtype=np.float32, elements=elements, shape=(n, n)))

      output = utils.flat_strict_upper(input_).numpy()

      # verify
      k = 0
      for i in range(n):
        for j in range(i + 1, n):
          assert output[k] == input_[i, j]
          k += 1

  @given(data=st.data(), dtype=np_st.integer_dtypes())
  def test_flat_strict_upper_int_verification(self, int_graph, data, dtype):
    # sample square matrix
    n = data.draw(st.integers(min_value=3, max_value=10))
    input_ = data.draw(np_st.arrays(dtype=dtype, shape=(n, n)))

    # graph building
    input_pl, output = int_graph

    # running
    with tf.Session() as sess:
      output_val = sess.run(output, feed_dict={input_pl: input_})

    # verify
    k = 0
    for i in range(n):
      for j in range(i + 1, n):
        assert output_val[k] == input_[i, j]
        k += 1

  @given(data=st.data())
  def test_flat_strict_upper_float_verification(self, float_graph, data):
    # sample square matrix
    n = data.draw(st.integers(min_value=3, max_value=10))
    elements = st.floats(allow_nan=False, allow_infinity=False, width=32)
    input_ = data.draw(
        np_st.arrays(dtype=np.float32, elements=elements, shape=(n, n)))

    # graph building
    input_pl, output = float_graph

    # running
    with tf.Session() as sess:
      output_val = sess.run(output, feed_dict={input_pl: input_})

    # verify
    k = 0
    for i in range(n):
      for j in range(i + 1, n):
        assert output_val[k] == input_[i, j]
        k += 1

  @given(data=st.data(), dtype=np_st.integer_dtypes())
  def test_flat_strict_upper_int_eager_graph_consistency(
      self, int_graph, data, dtype):
    # sample square matrix
    n = data.draw(st.integers(min_value=3, max_value=10))
    input_ = data.draw(np_st.arrays(dtype=dtype, shape=(n, n)))

    # compute eager output
    with eager_mode():
      eager_output = utils.flat_strict_upper(input_)

    # compute graph output
    input_pl, output = int_graph
    with tf.Session() as sess:
      graph_output = sess.run(output, feed_dict={input_pl: input_})

    np.testing.assert_equal(eager_output, graph_output)

  @given(data=st.data())
  def test_flat_strict_upper_float_eager_graph_consistency(
      self, float_graph, data):
    # sample square matrix
    n = data.draw(st.integers(min_value=3, max_value=10))
    elements = st.floats(allow_nan=False, allow_infinity=False, width=32)
    input_ = data.draw(
        np_st.arrays(dtype=np.float32, elements=elements, shape=(n, n)))

    # compute eager output
    with eager_mode():
      eager_output = utils.flat_strict_upper(input_)

    # compute graph output
    input_pl, output = float_graph
    with tf.Session() as sess:
      graph_output = sess.run(output, feed_dict={input_pl: input_})

    np.testing.assert_equal(eager_output, graph_output)

  @given(shape=np_st.array_shapes(max_dims=4))
  def test_flat_strict_upper_wrong_input_dims(self, shape):
    assume(len(shape) != 2)

    input_ = tf.placeholder(tf.int32, shape)
    with pytest.raises(ValueError):
      utils.flat_strict_upper(input_)
