import tensorflow as tf

from histogram_loss import utils


def histogram_loss(descriptors: tf.Tensor, labels: tf.Tensor,
                   n_histograms: int) -> tf.Tensor:
  """
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

    # get positive distances
    pos_dist_mat = tf.equal(labels, tf.expand_dims(labels, 1))
    pos_dist_mask = utils.flat_strict_upper(pos_dist_mat)
    n_pos = tf.reduce_sum(tf.cast(pos_dist_mask, dtype=tf.int32))
    pos_dists = tf.boolean_mask(dists, pos_dist_mask)
    pos_dists = tf.expand_dims(pos_dists, 1)

    # get negative distances
    neg_dist_mask = ~pos_dist_mask
    n_neg = tf.reduce_sum(tf.cast(neg_dist_mask, dtype=tf.int32))
    neg_dists = tf.boolean_mask(dists, neg_dist_mask)
    neg_dists = tf.expand_dims(neg_dists, 1)

    # quantize [0; 4] range in `n_histograms` bins
    bins = tf.linspace(0.0, 4.0, n_histograms)
    bins = tf.expand_dims(bins, 0)

    # compute positive histogram
    pos_binned = 1 - (n_histograms - 1) * tf.abs(pos_dists - bins) / 2
    pos_binned = tf.nn.relu(pos_binned)
    pos_hist = tf.reduce_sum(pos_binned, axis=0) / tf.cast(n_pos, tf.float32)
    pos_cdf = tf.cumsum(pos_hist)

    # compute negative histogram
    neg_binned = 1 - (n_histograms - 1) * tf.abs(neg_dists - bins) / 2
    neg_binned = tf.nn.relu(neg_binned)
    neg_hist = tf.reduce_sum(neg_binned, axis=0) / tf.cast(n_neg, tf.float32)

    return tf.reduce_sum(neg_hist * pos_cdf)
