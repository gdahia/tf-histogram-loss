import tensorflow as tf


def flat_strict_upper(mat: tf.Tensor) -> tf.Tensor:
  """Converts the strictly upper triangular part of a matrix into an array.

  Args:
    mat: a `tf.Tensor` matrix.

  Returns:
    The strictly upper triangular part of `mat` in a `tf.Tensor` array.

  Raises:
    ValueError: if `mat` does not have exactly 2 dims.
  """
  with tf.name_scope('flat_strict_upper'):
    mat = tf.convert_to_tensor(mat, name='mat_input')

    # check if it is matrix
    if len(mat.get_shape().as_list()) != 2:
      shape = mat.get_shape().as_list()
      raise ValueError("expected mat ndims=2, found "
                       "ndmins={}. Full shape received: "
                       "{}.".format(len(shape), shape))

    partitions = tf.ones_like(mat, dtype=tf.int32)
    eye = tf.matrix_band_part(partitions, 0, 0)
    partitions = tf.matrix_band_part(partitions, 0, -1) - eye
    return tf.dynamic_partition(mat, partitions, 2)[1]
