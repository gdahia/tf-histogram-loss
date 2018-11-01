from typing import Callable, List, Tuple

import tensorflow as tf


def create_tf_dataset(paths_by_label: List[List[str]],
                      preprocess: Callable[[str], Tuple[tf.Tensor, tf.Tensor]]
                      ) -> tf.train.Dataset:
  """
  """
  # create, first, a tf dataset per label
  datasets_per_label = []  # type: List[tf.train.Dataset]
  for label_paths in paths_by_label:
    label_dataset = tf.train.Dataset.from_tensor_slices(label_paths)
    label_dataset = label_dataset.map(preprocess)
    datasets_per_label.append(label_dataset)

  # concatenate label datasets
  dataset = tf.train.Dataset()
  for label_dataset in datasets_per_label:
    dataset.concatenate(label_dataset)

  return dataset
