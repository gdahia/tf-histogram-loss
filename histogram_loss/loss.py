import tensorflow as tf

from histogram_loss import utils


def histogram_loss(descriptors: tf.Tensor,
                   labels: tf.Tensor,
                   n_bins: int = 256) -> tf.Tensor:
  """Computes the histogram loss for the given `descriptors` and `labels`, with a `n_bins`-dimensional uniformly distributed histogram.

  This function implements the loss described in:
    ["Learning Deep Embeddings with Histogram Loss". E. Ustinova, V. Lempitsky](https://arxiv.org/abs/1611.00822)

  It differs from the proposed implementation because it computes the squared L2 distance between descriptors, instead of their dot product.
  The loss to minimize, then, becomes the estimated probability of the distance of a negative pair to be less than the distance of a positive pair.
  Since the descriptors are L2 normalized, the similarity S(d1, d2) = cos(d1, d2) between two descriptors d1 and d2 and their L2 distance D(d1, d2) are related via:
    D(d1, d2) = 2 - 2 * acos(S(d1, d2))

  The squared L2 distance between L2-normalized vectors is bounded to [0; 4] and the histogram is fit with `n_bins` bins uniformly distributed in this range.

  Args:
    descriptors: a `tf.Tensor` of `tf.float32` L2-normalized descriptors of shape `[batch_size, descriptor_dims]`.
    labels: a `tf.Tensor` of `tf.int32` labels for the given descriptors. Its shape must be `[batch_size]`.
    n_bins: an `int` indicating the number of bins to uniformly divide the range [0; 4].

  Returns:
    the histogram loss of the descriptors, computed with their L2 distances.

  Raises:
    ValueError: if `descriptors` has `ndmis` other than 2.
    ValueError: if `labels` has `ndmis` other than 1.
    ValueError: if `descriptors` and `labels` have incompatible shapes.
  """
  with tf.name_scope('histogram_loss'):
    descriptors = tf.convert_to_tensor(descriptors, name='descriptors_input')
    labels = tf.convert_to_tensor(labels, name='labels_input')

    # check if descriptors have proper dims
    descs_shape = descriptors.get_shape().as_list()
    if len(descs_shape) != 2:
      raise ValueError("expected descriptors ndmis=2, "
                       "found ndims={}. Full shape received: "
                       "{}.".format(descs_shape[0], descs_shape))

    # check if labels have proper dims
    labels_shape = labels.get_shape().as_list()
    if len(labels_shape) != 1:
      raise ValueError("expected labels ndmis=2, found "
                       "ndims={}. Full shape received: "
                       "{}.".format(labels_shape[0], labels_shape))

    # check if descriptors and labels have compatible shapes
    if descs_shape[0] is not None and labels_shape[0] is not None:
      if descs_shape[0] != 1 and labels_shape[0] != 1:
        if descs_shape[0] != labels_shape[0]:
          raise ValueError("descriptors and labels have "
                           "incompatible first dimensions, "
                           "{} vs {}.".format(descs_shape[0], labels_shape[0]))

    # compute pairwise distances for l2-normalized descriptors
    dists_mat = 2 - 2 * tf.matmul(descriptors, tf.transpose(descriptors))
    dists = utils.flat_strict_upper(dists_mat)

    # separate positive and negative pair distances
    partitions_mat = tf.equal(labels, tf.expand_dims(labels, 1))
    partitions = utils.flat_strict_upper(partitions_mat)
    partitions = tf.cast(partitions, tf.int32)
    neg_dists, pos_dists = tf.dynamic_partition(dists, partitions, 2)

    # quantize [0; 4] range in n_bins bins
    bins = tf.linspace(0.0, 4.0, n_bins)
    bins = tf.expand_dims(bins, 0)

    # compute positive pairs distance histogram
    n_pos = tf.reduce_sum(partitions)
    pos_dists = tf.expand_dims(pos_dists, 1)
    pos_binned = 1 - (n_bins - 1) * tf.abs(pos_dists - bins) / 2
    pos_binned = tf.nn.relu(pos_binned)
    pos_hist = tf.reduce_sum(pos_binned, axis=0) / tf.cast(n_pos, tf.float32)

    # compute negative pairs distance histogram and cdf approximation
    n_neg = tf.reduce_sum(1 - partitions)
    neg_dists = tf.expand_dims(neg_dists, 1)
    neg_binned = 1 - (n_bins - 1) * tf.abs(neg_dists - bins) / 2
    neg_binned = tf.nn.relu(neg_binned)
    neg_hist = tf.reduce_sum(neg_binned, axis=0) / tf.cast(n_neg, tf.float32)
    neg_cdf = tf.cumsum(neg_hist)

    return tf.reduce_sum(pos_hist * neg_cdf)
