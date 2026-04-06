from unittest.mock import MagicMock

import pytest

from soma_connect import ArchiveClient, Animal
from soma_connect.exceptions import AuthenticationError, FilterError, RecordNotFoundError


def _make_response(data: list) -> MagicMock:
    resp = MagicMock()
    resp.data = data
    resp.error = None
    return resp


# ── Authentication ───────────────────────────────────────────────────────────

def test_missing_credentials_raises(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    with pytest.raises(AuthenticationError):
        ArchiveClient()


def test_repr(client):
    assert "ArchiveClient" in repr(client)


# ── get_animal ───────────────────────────────────────────────────────────────

def test_get_animal_no_filters(client, mock_supabase):
    row = {"animal_id": "m01", "type": "mouse", "sex": "M", "cage": "A1",
           "mark": "red", "birth_date": None, "surgery_date": None, "notes": None}
    mock_supabase.table.return_value.select.return_value.execute.return_value = (
        _make_response([row])
    )
    result = client.get_animal()
    assert len(result) == 1
    assert result[0].animal_id == "m01"


def test_get_animal_with_filter(client, mock_supabase):
    chain = mock_supabase.table.return_value.select.return_value
    chain.eq.return_value.execute.return_value = _make_response([])
    result = client.get_animal(filters={"type": "mouse"})
    chain.eq.assert_called_once_with("type", "mouse")
    assert result == []


def test_get_animal_as_dataframe(client, mock_supabase):
    import pandas as pd

    row = {"animal_id": "m01", "type": "mouse", "sex": "M", "cage": "A1",
           "mark": "red", "birth_date": None, "surgery_date": None, "notes": None}
    mock_supabase.table.return_value.select.return_value.execute.return_value = (
        _make_response([row])
    )
    df = client.get_animal(as_dataframe=True)
    assert isinstance(df, pd.DataFrame)
    assert "animal_id" in df.columns


def test_get_animal_invalid_filter_raises(client):
    with pytest.raises(FilterError, match="animla_id"):
        client.get_animal(filters={"animla_id": "m01"})


# ── insert_animal ─────────────────────────────────────────────────────────────

def test_insert_animal(client, mock_supabase):
    row = {"animal_id": "m02", "type": "rat", "sex": "F", "cage": "B1",
           "mark": "blue", "birth_date": None, "surgery_date": None, "notes": None}
    mock_supabase.table.return_value.insert.return_value.execute.return_value = (
        _make_response([row])
    )
    animal = client.insert_animal({"animal_id": "m02", "type": "rat"})
    assert isinstance(animal, Animal)
    assert animal.animal_id == "m02"


# ── update_animal ─────────────────────────────────────────────────────────────

def test_update_animal_not_found_raises(client, mock_supabase):
    mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = (
        _make_response([])
    )
    with pytest.raises(RecordNotFoundError, match="m99"):
        client.update_animal("m99", {"cage": "C1"})


# ── relation select strings ───────────────────────────────────────────────────

def test_get_recording_with_animal_select(client, mock_supabase):
    mock_supabase.table.return_value.select.return_value.execute.return_value = (
        _make_response([])
    )
    client.get_recording_with_animal()
    mock_supabase.table.return_value.select.assert_called_with("*, animal(*)")


def test_get_tensor_with_relations_select(client, mock_supabase):
    mock_supabase.table.return_value.select.return_value.execute.return_value = (
        _make_response([])
    )
    client.get_tensor_with_relations()
    mock_supabase.table.return_value.select.assert_called_with("*, animal(*), sorting(*)")


# ── get_tensor_matrix ─────────────────────────────────────────────────────────

def test_get_tensor_matrix_delegates(client, mock_supabase):
    rows = [
        {"id": 1, "sorting_id": 1, "neuron_id": 1, "stimuli": "s0", "trial": 0, "firing_rate": 5.0, "animal_id": "m01"},
        {"id": 2, "sorting_id": 1, "neuron_id": 1, "stimuli": "s0", "trial": 1, "firing_rate": 6.0, "animal_id": "m01"},
        {"id": 3, "sorting_id": 1, "neuron_id": 2, "stimuli": "s0", "trial": 0, "firing_rate": 3.0, "animal_id": "m01"},
        {"id": 4, "sorting_id": 1, "neuron_id": 2, "stimuli": "s0", "trial": 1, "firing_rate": 4.0, "animal_id": "m01"},
    ]
    mock_supabase.table.return_value.select.return_value.execute.return_value = (
        _make_response(rows)
    )
    import numpy as np
    mat, neurons, stimuli, trials = client.get_tensor_matrix()
    assert mat.shape == (2, 1, 2)
    assert neurons == [1, 2]
    assert stimuli == ["s0"]
    assert trials == [0, 1]
    assert mat[0, 0, 0] == pytest.approx(5.0)
