from datetime import date

from soma_connect.models import Animal, Recording, Tensor


def test_animal_basic():
    a = Animal(animal_id="m01", type="mouse", sex="M")
    assert a.animal_id == "m01"
    assert a.notes is None


def test_recording_recoding_date_typo_normalised():
    # DB uses "recoding_date"; both spellings should work
    r1 = Recording(id=1, animal_id="m01", recoding_date=date(2024, 1, 15))
    r2 = Recording(id=1, animal_id="m01", recording_date=date(2024, 1, 15))
    assert r1.recoding_date == r2.recoding_date == date(2024, 1, 15)


def test_recording_date_alias_property():
    r = Recording(id=1, animal_id="m01", recoding_date=date(2024, 3, 10))
    assert r.recording_date == date(2024, 3, 10)


def test_tensor_optional_fields_default_none():
    t = Tensor(id=1, sorting_id=2, neuron_id=3)
    assert t.firing_rate is None
    assert t.stimuli is None
    assert t.trial is None


def test_repr_html_contains_field_names():
    a = Animal(animal_id="m01", type="mouse")
    html = a._repr_html_()
    assert "animal_id" in html
    assert "m01" in html
    assert "<table" in html


def test_model_tolerates_extra_join_fields():
    # When Supabase returns nested join data, extra fields should not raise
    data = {
        "id": 1,
        "animal_id": "m01",
        "recoding_date": None,
        "notes": None,
        "sample": 30000,
        "animal": {"animal_id": "m01", "type": "mouse"},  # joined field
    }
    r = Recording.model_validate(data)
    assert r.animal_id == "m01"
