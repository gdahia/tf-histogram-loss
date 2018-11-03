import os
import numpy as np
import tensorflow as tf
from tensorflow.python.eager.context import eager_mode

from histogram_loss.datasets import vggface2


class TestVGGFace2:
  def test_load_test_set(self):
    dir_path = os.path.join('datasets', 'vggface2')
    path = os.path.join(dir_path, 'test_list.txt')
    dir_path = os.path.join(dir_path, 'test' + os.path.sep)
    dataset = vggface2.load(path, dir_path)

    # confirm images are in right format
    assert dataset.output_shapes[0].as_list() == [160, 160, 3]
    assert dataset.output_types[0] is tf.float32

    # confirm labels are in right format
    assert dataset.output_types[1] is tf.int32

    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(128)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
      images, labels = sess.run(next_element)

      # every image has a label
      assert len(images) == len(labels)

      # images are in range [0, 1]
      assert np.all(images <= 1)
      assert np.all(0 <= images)

      # labels are integers
      assert np.issubdtype(labels.dtype, np.integer)

  def test_eager_load_test_set(self):
    dir_path = os.path.join('datasets', 'vggface2')
    path = os.path.join(dir_path, 'test_list.txt')
    dir_path = os.path.join(dir_path, 'test' + os.path.sep)

    with eager_mode():
      dataset = vggface2.load(path, dir_path)

      # confirm images are in right format
      assert dataset.output_shapes[0].as_list() == [160, 160, 3]
      assert dataset.output_types[0] is tf.float32

      # confirm labels are in right format
      assert dataset.output_types[1] is tf.int32

      dataset = dataset.shuffle(buffer_size=1000)
      dataset = dataset.batch(128)

      for images, labels in dataset:
        images = images.numpy()
        labels = labels.numpy()

        # images are in range [0, 1]
        assert np.all(images <= 1)
        assert np.all(0 <= images)

        # labels are integers
        assert np.issubdtype(labels.dtype, np.integer)

        break

  def test_load_train_set(self):
    dir_path = os.path.join('datasets', 'vggface2')
    path = os.path.join(dir_path, 'train_list.txt')
    dir_path = os.path.join(dir_path, 'train' + os.path.sep)
    dataset = vggface2.load(path, dir_path)

    # confirm images are in right format
    assert dataset.output_shapes[0].as_list() == [160, 160, 3]
    assert dataset.output_types[0] is tf.float32

    # confirm labels are in right format
    assert dataset.output_types[1] is tf.int32

    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(128)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
      images, labels = sess.run(next_element)

      # every image has a label
      assert len(images) == len(labels)

      # images are in range [0, 1]
      assert np.all(images <= 1)
      assert np.all(0 <= images)

      # labels are integers
      assert np.issubdtype(labels.dtype, np.integer)

  def test_eager_load_train_set(self):
    dir_path = os.path.join('datasets', 'vggface2')
    path = os.path.join(dir_path, 'train_list.txt')
    dir_path = os.path.join(dir_path, 'train' + os.path.sep)

    with eager_mode():
      dataset = vggface2.load(path, dir_path)

      # confirm images are in right format
      assert dataset.output_shapes[0].as_list() == [160, 160, 3]
      assert dataset.output_types[0] is tf.float32

      # confirm labels are in right format
      assert dataset.output_types[1] is tf.int32

      dataset = dataset.shuffle(buffer_size=1000)
      dataset = dataset.batch(128)

      for images, labels in dataset:
        images = images.numpy()
        labels = labels.numpy()

        # images are in range [0, 1]
        assert np.all(images <= 1)
        assert np.all(0 <= images)

        # labels are integers
        assert np.issubdtype(labels.dtype, np.integer)

        break
