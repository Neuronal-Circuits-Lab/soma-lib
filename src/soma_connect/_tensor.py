from __future__ import annotations

import base64
import io
from typing import Sequence

import numpy as np

from .exceptions import TensorShapeError
from .models import Tensor


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _encode_firing_rates(arr: np.ndarray) -> str:
    """Serialise a 1-D float32 array to a base64 string for DB storage."""
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_firing_rates(blob: str | bytes) -> np.ndarray:
    """Deserialise a base64 string (or raw bytes) back to a 1-D float32 array."""
    raw = base64.b64decode(blob) if isinstance(blob, (str, bytes)) else blob
    return np.load(io.BytesIO(raw))


# ---------------------------------------------------------------------------
# rows_to_tensor  (was rows_to_matrix)
# ---------------------------------------------------------------------------

def rows_to_tensor(
    rows: list[Tensor],
) -> tuple[np.ndarray, list[int], list[str], list[int]]:
    """
    Convert a flat list of Tensor rows into a 4-D numpy array.

    Each row stores the full firing-rate timeseries for one
    (neuron, stimuli, trial) combination as a base64-encoded blob.
    The timepoints dimension is reconstructed by decoding that blob.

    Parameters
    ----------
    rows:
        List of Tensor model instances fetched from the database.

    Returns
    -------
    tensor : np.ndarray, shape (n_neurons, n_stimuli, n_trials, n_timepoints)
        Firing rates, dtype float32.
        Slices without data are filled with NaN.
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
    >>> tensor, neurons, stimuli, trials = rows_to_tensor(tensor_rows)
    >>> tensor.shape
    (12, 5, 20, 600)
    >>> # Firing-rate timeseries for neuron 0, stimulus 0, trial 0
    >>> tensor[0, 0, 0]
    array([0.12, 0.45, ...], dtype=float32)
    """
    if not rows:
        raise TensorShapeError(
            "Cannot build a tensor from an empty result set. "
            "Check your filters — no rows were returned from the database."
        )

    neuron_ids = sorted({r.neuron_id for r in rows})
    stimuli_labels = sorted({r.stimuli for r in rows if r.stimuli is not None})
    trial_ids = sorted({r.trial for r in rows if r.trial is not None})

    n_idx = {v: i for i, v in enumerate(neuron_ids)}
    s_idx = {v: i for i, v in enumerate(stimuli_labels)}
    t_idx = {v: i for i, v in enumerate(trial_ids)}

    # Decode one row to determine n_timepoints
    first_blob = next(
        (r.firing_rates for r in rows if r.firing_rates is not None), None
    )
    if first_blob is None:
        raise TensorShapeError(
            "All rows have a null firing_rates blob. "
            "The tensor table may have been populated with the old schema."
        )
    n_timepoints = len(_decode_firing_rates(first_blob))

    tensor = np.full(
        (len(neuron_ids), len(stimuli_labels), len(trial_ids), n_timepoints),
        fill_value=np.nan,
        dtype=np.float32,
    )

    for row in rows:
        if row.stimuli is None or row.trial is None or row.firing_rates is None:
            continue
        ni = n_idx[row.neuron_id]
        si = s_idx[row.stimuli]
        ti = t_idx[row.trial]
        tensor[ni, si, ti, :] = _decode_firing_rates(row.firing_rates)

    return tensor, neuron_ids, stimuli_labels, trial_ids


# Keep the old name as an alias so any code that imported rows_to_matrix
# directly from this module continues to work during the migration.
rows_to_matrix = rows_to_tensor


# ---------------------------------------------------------------------------
# tensor_to_rows  (was matrix_to_rows)
# ---------------------------------------------------------------------------

def tensor_to_rows(
    tensor: np.ndarray,
    neuron_ids: Sequence[int],
    stimuli_labels: Sequence[str],
    animal_id: str,
    sorting_id: int,
) -> list[dict]:
    """
    Flatten a 4-D numpy array into a list of dicts for the tensor table.

    One row is produced per (neuron, stimuli, trial) combination.
    The firing-rate timeseries (axis 3) is serialised to a base64-encoded
    .npy blob and stored in the ``firing_rates`` column.
    NaN slices (all timepoints NaN) are skipped.

    Parameters
    ----------
    tensor:
        Array of shape (n_neurons, n_stimuli, n_trials, n_timepoints).
        Values are cast to float32 before serialisation.
    neuron_ids:
        Labels for axis 0. Must match ``tensor.shape[0]``.
    stimuli_labels:
        Labels for axis 1. Must match ``tensor.shape[1]``.
    animal_id:
        FK to the animal table, written to every row.
    sorting_id:
        FK to the sorting table, written to every row.

    Returns
    -------
    list[dict]
        Flat list of row dicts ready for batch insertion into the tensor table.
        Trial numbers are 0-based indices along axis 2.

    Raises
    ------
    TensorShapeError
        If the array is not 4-D or the label lengths do not match.

    Examples
    --------
    >>> rows = tensor_to_rows(tensor, [1, 2, 3], ["A", "B"], "m01", 5)
    >>> len(rows)   # 3 neurons × 2 stimuli × 20 trials = 120 (minus NaN slices)
    120
    >>> rows[0].keys()
    dict_keys(['neuron_id', 'stimuli', 'trial', 'firing_rates', 'animal_id', 'sorting_id'])
    """
    if tensor.ndim != 4:
        raise TensorShapeError(
            f"Expected a 4-D array with shape (n_neurons, n_stimuli, n_trials, n_timepoints), "
            f"got {tensor.ndim}-D array with shape {tensor.shape!r}."
        )

    n_neurons, n_stimuli, n_trials, _ = tensor.shape

    if len(neuron_ids) != n_neurons:
        raise TensorShapeError(
            f"neuron_ids has {len(neuron_ids)} entries but the tensor has "
            f"{n_neurons} neurons on axis 0. They must match."
        )
    if len(stimuli_labels) != n_stimuli:
        raise TensorShapeError(
            f"stimuli_labels has {len(stimuli_labels)} entries but the tensor has "
            f"{n_stimuli} stimuli on axis 1. They must match."
        )

    rows = []
    for ni, neuron_id in enumerate(neuron_ids):
        for si, stimuli in enumerate(stimuli_labels):
            for ti in range(n_trials):
                timeseries = tensor[ni, si, ti]          # shape (n_timepoints,)
                if np.all(np.isnan(timeseries)):
                    continue
                rows.append(
                    {
                        "neuron_id":    int(neuron_id),
                        "stimuli":      str(stimuli),
                        "trial":        ti,
                        "firing_rates": _encode_firing_rates(timeseries),
                        "animal_id":    animal_id,
                        "sorting_id":   sorting_id,
                    }
                )
    return rows


# Keep the old name as an alias for migration compatibility.
matrix_to_rows = tensor_to_rows