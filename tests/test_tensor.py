import numpy as np
import pytest

from soma_connect._tensor import matrix_to_rows, rows_to_matrix
from soma_connect.exceptions import TensorShapeError
from soma_connect.models import Tensor


def _make_tensor(**kwargs) -> Tensor:
    defaults = {"id": 1, "sorting_id": 1, "neuron_id": 1,
                "stimuli": "s0", "trial": 0, "firing_rate": 1.0, "animal_id": "m01"}
    defaults.update(kwargs)
    return Tensor.model_validate(defaults)


# ── rows_to_matrix ────────────────────────────────────────────────────────────

def test_rows_to_matrix_shape():
    rows = [
        _make_tensor(neuron_id=1, stimuli="s0", trial=0, firing_rate=1.0),
        _make_tensor(neuron_id=1, stimuli="s0", trial=1, firing_rate=2.0),
        _make_tensor(neuron_id=1, stimuli="s1", trial=0, firing_rate=3.0),
        _make_tensor(neuron_id=2, stimuli="s0", trial=0, firing_rate=4.0),
        _make_tensor(neuron_id=2, stimuli="s0", trial=1, firing_rate=5.0),
        _make_tensor(neuron_id=2, stimuli="s1", trial=0, firing_rate=6.0),
    ]
    mat, neurons, stimuli, trials = rows_to_matrix(rows)
    assert mat.shape == (2, 2, 2)
    assert neurons == [1, 2]
    assert stimuli == ["s0", "s1"]
    assert trials == [0, 1]


def test_rows_to_matrix_values():
    rows = [
        _make_tensor(neuron_id=1, stimuli="s0", trial=0, firing_rate=7.5),
        _make_tensor(neuron_id=2, stimuli="s0", trial=0, firing_rate=3.2),
    ]
    mat, _, _, _ = rows_to_matrix(rows)
    assert mat[0, 0, 0] == pytest.approx(7.5)
    assert mat[1, 0, 0] == pytest.approx(3.2)


def test_rows_to_matrix_nan_for_missing_cell():
    rows = [
        _make_tensor(neuron_id=1, stimuli="s0", trial=0, firing_rate=1.0),
        # trial=1 for neuron 1 is absent — should be NaN
        _make_tensor(neuron_id=2, stimuli="s0", trial=0, firing_rate=2.0),
        _make_tensor(neuron_id=2, stimuli="s0", trial=1, firing_rate=3.0),
    ]
    mat, _, _, _ = rows_to_matrix(rows)
    assert np.isnan(mat[0, 0, 1])  # neuron 1, stimuli s0, trial 1
    assert not np.isnan(mat[1, 0, 0])


def test_rows_to_matrix_empty_raises():
    with pytest.raises(TensorShapeError, match="empty"):
        rows_to_matrix([])


# ── matrix_to_rows ────────────────────────────────────────────────────────────

def test_matrix_to_rows_round_trip():
    mat = np.array([[[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]]])  # shape (2, 2, 2)
    rows = matrix_to_rows(mat, [10, 20], ["grating_0", "grating_90"], "m01", 5)
    assert len(rows) == 8  # 2×2×2, no NaN
    neuron_ids = {r["neuron_id"] for r in rows}
    assert neuron_ids == {10, 20}


def test_matrix_to_rows_skips_nan():
    mat = np.array([[[1.0, np.nan], [3.0, 4.0]]])  # shape (1, 2, 2)
    rows = matrix_to_rows(mat, [1], ["s0", "s1"], "m01", 1)
    assert len(rows) == 3  # 4 cells minus 1 NaN


def test_matrix_to_rows_wrong_ndim_raises():
    mat = np.ones((3, 4))  # 2-D, not 3-D
    with pytest.raises(TensorShapeError, match="3-D"):
        matrix_to_rows(mat, [1, 2, 3], ["s0", "s1", "s2", "s3"], "m01", 1)


def test_matrix_to_rows_neuron_length_mismatch_raises():
    mat = np.ones((3, 2, 5))
    with pytest.raises(TensorShapeError, match="neuron_ids"):
        matrix_to_rows(mat, [1, 2], ["s0", "s1"], "m01", 1)  # 2 IDs but 3 neurons


def test_matrix_to_rows_stimuli_length_mismatch_raises():
    mat = np.ones((2, 3, 5))
    with pytest.raises(TensorShapeError, match="stimuli_labels"):
        matrix_to_rows(mat, [1, 2], ["s0", "s1"], "m01", 1)  # 2 labels but 3 stimuli


def test_matrix_to_rows_trial_index_is_positional():
    mat = np.ones((1, 1, 3))
    rows = matrix_to_rows(mat, [99], ["stim"], "m01", 7)
    trial_values = sorted(r["trial"] for r in rows)
    assert trial_values == [0, 1, 2]
