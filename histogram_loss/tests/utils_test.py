import tensorflow as tf
from tensorflow.python.eager.context import eager_mode
import numpy as np

import pytest
import hypothesis
from hypothesis import given, assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as np_st

from histogram_loss import utils


class TestUtils:
  @given(st.data())
  def test_eager_flat_strict_upper_oracle(self, data):
    with eager_mode():
      # sample square matrix
      n = data.draw(st.integers(min_value=3, max_value=10))
      input_ = data.draw(
          np_st.arrays(dtype=np.int32, shape=(n, n), unique=True))

      # dont use nan test cases
      assume(not np.any(np.isnan(input_)))

      output = utils.flat_strict_upper(input_).numpy()

      # check manually
      k = 0
      for i in range(n):
        for j in range(i + 1, n):
          assert output[k] == input_[i, j]
          k += 1

  @given(st.data())
  def test_eager_flat_strict_upper_unique_transposition_different(self, data):
    with eager_mode():
      # sample square matrix
      n = data.draw(st.integers(min_value=3, max_value=10))
      input_ = data.draw(np_st.arrays(dtype=np.int32, shape=(n, n)))

      # dont use symmetrical matrix test cases
      assume(not np.all(input_ == input_.T))

      # compute for inputs and its transpose
      output = utils.flat_strict_upper(input_).numpy()
      output_T = utils.flat_strict_upper(input_.T).numpy()

      with pytest.raises(AssertionError):
        np.testing.assert_array_equal(output_T, output)

  @given(st.data())
  def test_eager_flat_strict_upper_symmetric_invariance_to_transposition(
      self, data):
    # remove filter too much health check
    hypothesis.settings(
        suppress_health_check=[hypothesis.HealthCheck.filter_too_much])

    with eager_mode():
      # sample square matrix
      n = data.draw(st.integers(min_value=3, max_value=10))
      input_ = data.draw(np_st.arrays(dtype=np.int32, shape=(n, n)))

      # use only symmetrical matrix test cases
      assume(np.all(input_ == input_.T))

      # compute for inputs and its transpose
      output = utils.flat_strict_upper(input_).numpy()
      output_T = utils.flat_strict_upper(input_.T).numpy()

      np.testing.assert_array_equal(output_T, output)

  @given(st.data())
  def test_flat_strict_upper_symmetric_invariance_to_transposition(self, data):
    # remove filter too much health check
    hypothesis.settings(
        suppress_health_check=[hypothesis.HealthCheck.filter_too_much])

    # sample square matrix
    n = data.draw(st.integers(min_value=3, max_value=10))
    input_ = data.draw(np_st.arrays(dtype=np.int32, shape=(n, n)))

    # use only symmetrical matrix test cases
    assume(np.all(input_ == input_.T))

    # compute for inputs and its transpose
    input_pl = tf.placeholder(tf.int32)
    output = utils.flat_strict_upper(input_pl)

    with tf.Session() as sess:
      output_val = sess.run(output, feed_dict={input_pl: input_})
      output_T_val = sess.run(output, feed_dict={input_pl: input_.T})
      np.testing.assert_array_equal(output_T_val, output_val)
