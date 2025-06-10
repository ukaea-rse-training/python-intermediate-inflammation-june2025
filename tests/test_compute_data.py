from unittest.mock import Mock

import numpy as np

from inflammation.compute_data import analyse_data


def test_analyse_data_mock_source():
    data_source = Mock()
    data_source.load_inflammation_data.return_value = [
        np.array([[0, 2, 0]]),
        np.array([[0, 1, 0]]),
    ]

    analyse_data(data_source)
