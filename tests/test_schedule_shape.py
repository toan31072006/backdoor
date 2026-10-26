import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_utils.feeder import _next_piw_frame_name  # noqa: E402
from eval.metrics import schedule_shape_analysis  # noqa: E402


@pytest.mark.parametrize('mode', ['linear', 'sqrt', 'quad'])
def test_schedule_shape_mad_is_zero_for_the_prescribed_curve(mode):
    doses = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    expected = {
        'linear': doses,
        'sqrt': np.sqrt(doses),
        'quad': doses ** 2,
    }[mode]
    result = schedule_shape_analysis(doses, 0.37 * expected, mode)
    assert result['mad'] == pytest.approx(0.0, abs=1e-12)


def test_schedule_shape_rejects_missing_or_zero_reference():
    with pytest.raises(ValueError, match='exactly one reference'):
        schedule_shape_analysis([0.0, 0.5], [0.0, 0.5], 'linear')
    with pytest.raises(ValueError, match='zero displacement'):
        schedule_shape_analysis([0.0, 1.0], [0.0, 0.0], 'linear')


def test_piw_next_frame_rule_matches_dt_pose():
    assert _next_piw_frame_name('P01_A03_17') == 'P01_A03_18'
