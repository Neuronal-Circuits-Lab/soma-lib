from __future__ import annotations

from typing import Sequence

import numpy as np

from .exceptions import TensorShapeError
from .models import Tensor


def rows_to_matrix(
    rows: list[Tensor],
) -> tuple[np.ndarray, list[int], list[str], list[int]]:
    """
    Convert a flat list of Tensor rows into a 3-D numpy array.

    Parameters
    ----------
    rows:
        List of Tensor model instances fetched from the database.

    Returns
    -------
    matrix : np.ndarray, shape (n_neurons, n_stimuli, n_trials), dtype float64
        Firing rates. Cells without data are filled with NaN.
    neuron_ids : list[int]
        Sorted unique neuron IDs corresponding to axis 0.
    stimuli_labels : list[str]
        Sorted unique stimuli labels corresponding to axis 1.
    trial_ids : list[int]
        Sorted unique trial numbers corresponding to axis 2.

    Raises
    ------
    TensorShapeError
        If rows is empty.

    Examples
    --------
    >>> matrix, neurons, stimuli, trials = rows_to_matrix(tensor_rows)
    >>> matrix.shape
    (12, 5, 20)
    >>> # Which neuron is row 0?
    >>> neurons[0]
    1
    """
    if not rows:
        raise TensorShapeError(
            "Cannot build a tensor matrix from an empty result set. "
            "Check your filters — no rows were returned from the database."
        )

    neuron_ids = sorted({r.neuron_id for r in rows})
    stimuli_labels = sorted({r.stimuli for r in rows if r.stimuli is not None})
    trial_ids = sorted({r.trial for r in rows if r.trial is not None})

    n_idx = {v: i for i, v in enumerate(neuron_ids)}
    s_idx = {v: i for i, v in enumerate(stimuli_labels)}
    t_idx = {v: i for i, v in enumerate(trial_ids)}

    matrix = np.full(
        (len(neuron_ids), len(stimuli_labels), len(trial_ids)),
        fill_value=np.nan,
        dtype=np.float64,
    )

    for row in rows:
        if row.stimuli is None or row.trial is None or row.firing_rate is None:
            continue
        matrix[n_idx[row.neuron_id], s_idx[row.stimuli], t_idx[row.trial]] = (
            row.firing_rate
        )

    return matrix, neuron_ids, stimuli_labels, trial_ids


def matrix_to_rows(
    matrix: np.ndarray,
    neuron_ids: Sequence[int],
    stimuli_labels: Sequence[str],
    animal_id: str,
    sorting_id: int,
) -> list[dict]:
    """
    Flatten a 3-D numpy array into a list of dicts for the tensor table.

    Parameters
    ----------
    matrix:
        Array of shape (n_neurons, n_stimuli, n_trials) containing firing rates.
        NaN cells are skipped — they represent missing observations, not zero.
    neuron_ids:
        Labels for axis 0. Must have the same length as matrix.shape[0].
    stimuli_labels:
        Labels for axis 1. Must have the same length as matrix.shape[1].
    animal_id:
        FK to the animal table, written to every row.
    sorting_id:
        FK to the sorting table, written to every row.

    Returns
    -------
    list[dict]
        Flat list of row dicts ready for insertion into the tensor table.
        Trial numbers are 0-based indices along axis 2.

    Raises
    ------
    TensorShapeError
        If the array is not 3-D or the label lengths do not match.

    Examples
    --------
    >>> rows = matrix_to_rows(mat, [1, 2, 3], ["grating_0", "grating_90"], "m01", 5)
    >>> len(rows)  # 3 neurons × 2 stimuli × 10 trials = 60 (minus NaN cells)
    60
    """
    if matrix.ndim != 3:
        raise TensorShapeError(
            f"Expected a 3-D array with shape (n_neurons, n_stimuli, n_trials), "
            f"got {matrix.ndim}-D array with shape {matrix.shape!r}."
        )

    n_neurons, n_stimuli, n_trials = matrix.shape

    if len(neuron_ids) != n_neurons:
        raise TensorShapeError(
            f"neuron_ids has {len(neuron_ids)} entries but the matrix has "
            f"{n_neurons} neurons on axis 0. They must match."
        )
    if len(stimuli_labels) != n_stimuli:
        raise TensorShapeError(
            f"stimuli_labels has {len(stimuli_labels)} entries but the matrix has "
            f"{n_stimuli} stimuli on axis 1. They must match."
        )

    rows = []
    for ni, neuron_id in enumerate(neuron_ids):
        for si, stimuli in enumerate(stimuli_labels):
            for ti in range(n_trials):
                value = matrix[ni, si, ti]
                if np.isnan(value):
                    continue
                rows.append(
                    {
                        "neuron_id": int(neuron_id),
                        "stimuli": str(stimuli),
                        "trial": ti,
                        "firing_rate": float(value),
                        "animal_id": animal_id,
                        "sorting_id": sorting_id,
                    }
                )
    return rows
