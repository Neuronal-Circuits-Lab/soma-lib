from __future__ import annotations

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ArchiveModel(BaseModel):
    """Base model for all database tables."""

    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
        extra="allow",  # tolerates nested join columns from Supabase
    )

    def _repr_html_(self) -> str:
        """Render a pretty HTML table in Jupyter notebooks."""
        rows = "".join(
            f"<tr><td style='text-align:left;padding:4px 8px'><b>{k}</b></td>"
            f"<td style='padding:4px 8px'>{v}</td></tr>"
            for k, v in self.model_dump().items()
            if not k.startswith("_")
        )
        return (
            f"<table style='border-collapse:collapse'>"
            f"<thead><tr><th colspan='2' style='text-align:left;padding:4px 8px'>"
            f"{self.__class__.__name__}</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )


class Animal(ArchiveModel):
    """A research animal in the study."""

    animal_id: str
    birth_date: Optional[date] = None
    surgery_date: Optional[date] = None
    type: Optional[str] = None
    sex: Optional[str] = None
    cage: Optional[str] = None
    mark: Optional[str] = None
    notes: Optional[str] = None


class Recording(ArchiveModel):
    """A recording session for an animal."""

    id: int
    animal_id: str
    # The DB column is spelled "recoding_date" (typo). Both spellings are
    # accepted on input and serialised back as "recoding_date" for DB writes.
    recoding_date: Optional[date] = Field(None, alias="recoding_date")
    notes: Optional[str] = None
    sample: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def _normalise_date_typo(cls, values: dict[str, Any]) -> dict[str, Any]:
        if isinstance(values, dict):
            if "recording_date" in values and "recoding_date" not in values:
                values["recoding_date"] = values.pop("recording_date")
        return values

    @property
    def recording_date(self) -> Optional[date]:
        """Alias for recoding_date (corrected spelling)."""
        return self.recoding_date


class Sorting(ArchiveModel):
    """A spike-sorting run associated with a recording."""

    id: int
    recording_id: int
    sorting_date: Optional[date] = None


class NeuronInfo(ArchiveModel):
    """Metadata for a single neuron identified during sorting."""

    id: int
    neuron_id: int
    sorting_id: int


class SpikeTimes(ArchiveModel):
    """A single spike time measurement for a neuron."""

    id: int
    neuron_id: int
    times: float


class Tensor(ArchiveModel):
    """
    A single row in the tensor table.

    Each row represents the firing rate of one neuron for one
    (stimuli, trial) combination.
    """

    id: int
    sorting_id: int
    neuron_id: int
    trial: Optional[int] = None
    animal_id: Optional[str] = None
    firing_rate: Optional[float] = None
    stimuli: Optional[str] = None
