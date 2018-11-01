import pytest
from hypothesis import given, assume, strategies as st
import hypothesis.extra.numpy as np_st

import numpy as np
import tensorflow as tf
from tensorflow.python.eager.context import eager_mode

from histogram_loss.loss import histogram_loss


@pytest.fixture(scope='function')
def graph():
  descs_pl = tf.placeholder(tf.float32, [None, None])
  labels_pl = tf.placeholder(tf.int32, [None])
  loss = histogram_loss(descs_pl, labels_pl, 128)
  yield descs_pl, labels_pl, loss


class TestLoss:
  @given(
      descs=np_st.arrays(
          shape=(32, 128),
          dtype=np.float32,
          elements=st.floats(
              min_value=-1e9,
              max_value=1e9,
              allow_nan=False,
              allow_infinity=False,
              width=32)),
      labels=np_st.arrays(dtype=np.int32, shape=(32, )),
      n_bins=st.integers(min_value=64, max_value=256),
      offset=st.integers(min_value=-100, max_value=100))
  def test_eager_histogram_loss_adding_constants_to_labels_invariance(
      self, descs, labels, n_bins, offset):
    # discard uniform labels
    assume(len(np.unique(labels)) != 1)

    # l2 normalize
    norm = np.sqrt(np.sum(descs**2, axis=1) + 1e-12)
    descs /= np.expand_dims(norm, -1)

    with eager_mode():
      desired = histogram_loss(descs, labels, n_bins)
      actual = histogram_loss(descs, labels + offset, n_bins)

      np.testing.assert_almost_equal(actual, desired)

  @given(
      descs=np_st.arrays(
          shape=(32, 128),
          dtype=np.float32,
          elements=st.floats(
              min_value=-1e9,
              max_value=1e9,
              allow_nan=False,
              allow_infinity=False,
              width=32)),
      labels=np_st.arrays(dtype=np.int32, shape=(32, )),
      offset=st.integers(min_value=-100, max_value=100))
  def test_histogram_loss_adding_constants_to_labels_invariance(
      self, graph, descs, labels, offset):
    # discard uniform labels
    assume(len(np.unique(labels)) != 1)

    # l2 normalize
    norm = np.sqrt(np.sum(descs**2, axis=1) + 1e-12)
    descs /= np.expand_dims(norm, -1)

    # get graph
    descs_pl, labels_pl, loss = graph

    # run
    with tf.Session() as sess:
      desired = sess.run(loss, feed_dict={descs_pl: descs, labels_pl: labels})
      actual = sess.run(
          loss, feed_dict={
              descs_pl: descs,
              labels_pl: labels + offset
          })

    np.testing.assert_almost_equal(actual, desired)

  @given(
      descs=np_st.arrays(
          shape=(32, 128),
          dtype=np.float32,
          elements=st.floats(
              min_value=-1e9,
              max_value=1e9,
              allow_nan=False,
              allow_infinity=False,
              width=32)),
      labels=np_st.arrays(dtype=np.int32, shape=(32, )))
  def test_histogram_loss_eager_graph_consistency(self, graph, descs, labels):
    # discard uniform labels
    assume(len(np.unique(labels)) != 1)

    # l2 normalize
    norm = np.sqrt(np.sum(descs**2, axis=1) + 1e-12)
    descs /= np.expand_dims(norm, -1)

    # compute eager output
    with eager_mode():
      eager_output = histogram_loss(descs, labels, 128)

    # compute graph output
    descs_pl, labels_pl, loss = graph
    with tf.Session() as sess:
      graph_output = sess.run(
          loss, feed_dict={
              descs_pl: descs,
              labels_pl: labels
          })

    np.testing.assert_almost_equal(np.array(eager_output), graph_output)

  @given(descs_shape=np_st.array_shapes(min_dims=1, max_dims=4))
  def test_histogram_loss_wrong_descriptors_dims(self, descs_shape):
    assume(len(descs_shape) != 2)

    descs_pl = tf.placeholder(tf.float32, descs_shape)
    labels_pl = tf.placeholder(tf.int32, [None])

    with pytest.raises(ValueError):
      histogram_loss(descs_pl, labels_pl, 128)

  @given(labels_shape=np_st.array_shapes(min_dims=2, max_dims=4))
  def test_histogram_loss_wrong_labels_dims(self, labels_shape):
    descs_pl = tf.placeholder(tf.float32, [None, None])
    labels_pl = tf.placeholder(tf.int32, labels_shape)

    with pytest.raises(ValueError):
      histogram_loss(descs_pl, labels_pl, 128)

  @given(descs_n=st.integers(min_value=2), labels_n=st.integers(min_value=2))
  def test_histogram_loss_incompatible_shapes(self, descs_n, labels_n):
    assume(descs_n != labels_n)

    descs_pl = tf.placeholder(tf.float32, [descs_n, None])
    labels_pl = tf.placeholder(tf.int32, [labels_n])

    with pytest.raises(ValueError):
      histogram_loss(descs_pl, labels_pl, 128)
