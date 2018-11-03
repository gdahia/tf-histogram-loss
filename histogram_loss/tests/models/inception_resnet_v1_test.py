import pytest
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as np_st

import numpy as np

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
