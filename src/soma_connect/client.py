from __future__ import annotations

import os
from typing import Any, Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client

from ._query import apply_filters
from ._tensor import tensor_to_rows, rows_to_tensor
from .exceptions import ArchiveError, AuthenticationError, RecordNotFoundError
from .models import Animal, NeuronInfo, Recording, Sorting, SpikeTimes, Tensor

load_dotenv()

_TABLE_MODEL = {
    "animal": Animal,
    "recording": Recording,
    "sorting": Sorting,
    "neuron_info": NeuronInfo,
    "spike_times": SpikeTimes,
    "tensor": Tensor,
}


class ArchiveClient:
    """
    Client for the neuro archive Supabase database.

    Parameters
    ----------
    url : str, optional
        Supabase project URL. Falls back to the SUPABASE_URL environment
        variable (or a .env file in your working directory).
    key : str, optional
        Supabase API key (anon or service role). Falls back to SUPABASE_KEY.
    email : str, optional
        User email for password-based sign-in. Falls back to SUPABASE_EMAIL.
        Required for INSERT/UPDATE/DELETE (authenticated role RLS policies).
    password : str, optional
        User password. Falls back to SUPABASE_PASSWORD.

    Examples
    --------
    >>> from soma_connect import ArchiveClient
    >>> client = ArchiveClient()                    # reads all credentials from .env
    >>> client = ArchiveClient(url="https://...", key="eyJ...", email="u@x.com", password="secret")

    >>> animals = client.get_animal()              # list of Animal models
    >>> df = client.get_animal(as_dataframe=True)  # pandas DataFrame

    >>> mat, neurons, stimuli, trials = client.get_tensor_matrix(
    ...     filters={"animal_id": "mouse01"}
    ... )
    >>> mat.shape
    (12, 5, 20)
    """

    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
        email: str | None = None,
        password: str | None = None,
    ) -> None:
        _url = url or os.environ.get("SUPABASE_URL")
        _key = (
            key
            or os.environ.get("SUPABASE_KEY")
            or os.environ.get("SUPABASE_ANON_KEY")
            or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        )

        if not _url or not _key:
            raise AuthenticationError(
                "Supabase credentials are missing.\n"
                "  Option 1: Create a .env file with:\n"
                "    SUPABASE_URL=https://your-project.supabase.co\n"
                "    SUPABASE_KEY=your-api-key\n"
                "  Option 2: Pass url= and key= when creating the client:\n"
                "    ArchiveClient(url='https://...', key='eyJ...')"
            )

        try:
            self._client: Client = create_client(_url, _key)
        except Exception as exc:
            raise AuthenticationError(
                f"Failed to connect to Supabase: {exc}"
            ) from exc

        _email = email or os.environ.get("SUPABASE_EMAIL")
        _password = password or os.environ.get("SUPABASE_PASSWORD")
        if _email and _password:
            self.sign_in(_email, _password)

    def __repr__(self) -> str:
        return f"ArchiveClient(url={self._client.supabase_url!r})"

    # ── Auth ─────────────────────────────────────────────────────────────────

    def sign_in(self, email: str, password: str) -> None:
        """Authenticate as a user. Required for INSERT/UPDATE/DELETE."""
        try:
            self._client.auth.sign_in_with_password({"email": email, "password": password})
        except Exception as exc:
            raise AuthenticationError(f"Sign-in failed: {exc}") from exc

    def sign_out(self) -> None:
        """End the current user session."""
        self._client.auth.sign_out()

    # ── Internal helpers ────────────────────────────────────────────────────

    def _execute(self, query: Any) -> list[dict]:
        """Run a supabase-py query and return the data rows."""
        try:
            response = query.execute()
        except Exception as exc:
            raise ArchiveError(f"Database request failed: {exc}") from exc
        if hasattr(response, "error") and response.error:
            raise ArchiveError(str(response.error))
        return response.data or []

    def _to_models_or_df(
        self,
        rows: list[dict],
        table: str,
        as_dataframe: bool,
    ) -> Union[list, pd.DataFrame]:
        if as_dataframe:
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        model_cls = _TABLE_MODEL[table]
        return [model_cls.model_validate(row) for row in rows]

    # ── animal ──────────────────────────────────────────────────────────────

    def get_animal(
        self,
        filters: dict[str, Any] | None = None,
        as_dataframe: bool = False,
    ) -> list[Animal] | pd.DataFrame:
        """
        Fetch animals from the database.

        Parameters
        ----------
        filters : dict, optional
            Equality filters, e.g. ``{"type": "mouse", "sex": "M"}``.
        as_dataframe : bool, default False
            Return a pandas DataFrame instead of a list of Animal models.

        Returns
        -------
        list[Animal] or pandas.DataFrame
        """
        q = apply_filters(self._client.table("animal").select("*"), "animal", filters)
        return self._to_models_or_df(self._execute(q), "animal", as_dataframe)

    def insert_animal(self, data: dict[str, Any] | Animal) -> Animal:
        """Insert a new animal record and return the created Animal."""
        payload = data.model_dump(by_alias=True) if isinstance(data, Animal) else data
        rows = self._execute(self._client.table("animal").insert(payload))
        if not rows:
            raise ArchiveError("Insert succeeded but the database returned no data.")
        return Animal.model_validate(rows[0])

    def update_animal(self, animal_id: str, data: dict[str, Any]) -> Animal:
        """Update an animal record by animal_id and return the updated Animal."""
        rows = self._execute(
            self._client.table("animal").update(data).eq("animal_id", animal_id)
        )
        if not rows:
            raise RecordNotFoundError(
                f"No animal found with animal_id={animal_id!r}. "
                "Check the ID and try again."
            )
        return Animal.model_validate(rows[0])

    def delete_animal(self, animal_id: str) -> None:
        """Delete an animal record by animal_id."""
        self._execute(
            self._client.table("animal").delete().eq("animal_id", animal_id)
        )

    # ── recording ───────────────────────────────────────────────────────────

    def get_recording(
        self,
        filters: dict[str, Any] | None = None,
        as_dataframe: bool = False,
    ) -> list[Recording] | pd.DataFrame:
        """
        Fetch recording sessions from the database.

        Parameters
        ----------
        filters : dict, optional
            Equality filters, e.g. ``{"animal_id": "mouse01"}``.
        as_dataframe : bool, default False
            Return a pandas DataFrame instead of a list of Recording models.
        """
        q = apply_filters(
            self._client.table("recording").select("*"), "recording", filters
        )
        return self._to_models_or_df(self._execute(q), "recording", as_dataframe)

    def insert_recording(self, data: dict[str, Any] | Recording) -> Recording:
        """Insert a new recording and return the created Recording."""
        payload = (
            data.model_dump(by_alias=True) if isinstance(data, Recording) else data
        )
        rows = self._execute(self._client.table("recording").insert(payload))
        if not rows:
            raise ArchiveError("Insert succeeded but the database returned no data.")
        return Recording.model_validate(rows[0])

    def update_recording(self, id: int, data: dict[str, Any]) -> Recording:
        """Update a recording by id and return the updated Recording."""
        rows = self._execute(
            self._client.table("recording").update(data).eq("id", id)
        )
        if not rows:
            raise RecordNotFoundError(f"No recording found with id={id}.")
        return Recording.model_validate(rows[0])

    def delete_recording(self, id: int) -> None:
        """Delete a recording by id."""
        self._execute(self._client.table("recording").delete().eq("id", id))

    # ── sorting ─────────────────────────────────────────────────────────────

    def get_sorting(
        self,
        filters: dict[str, Any] | None = None,
        as_dataframe: bool = False,
    ) -> list[Sorting] | pd.DataFrame:
        """
        Fetch spike-sorting runs from the database.

        Parameters
        ----------
        filters : dict, optional
            Equality filters, e.g. ``{"recording_id": 3}``.
        as_dataframe : bool, default False
            Return a pandas DataFrame instead of a list of Sorting models.
        """
        q = apply_filters(
            self._client.table("sorting").select("*"), "sorting", filters
        )
        return self._to_models_or_df(self._execute(q), "sorting", as_dataframe)

    def insert_sorting(self, data: dict[str, Any] | Sorting) -> Sorting:
        """Insert a new sorting run and return the created Sorting."""
        payload = (
            data.model_dump(by_alias=True) if isinstance(data, Sorting) else data
        )
        rows = self._execute(self._client.table("sorting").insert(payload))
        if not rows:
            raise ArchiveError("Insert succeeded but the database returned no data.")
        return Sorting.model_validate(rows[0])

    def update_sorting(self, id: int, data: dict[str, Any]) -> Sorting:
        """Update a sorting run by id and return the updated Sorting."""
        rows = self._execute(
            self._client.table("sorting").update(data).eq("id", id)
        )
        if not rows:
            raise RecordNotFoundError(f"No sorting found with id={id}.")
        return Sorting.model_validate(rows[0])

    def delete_sorting(self, id: int) -> None:
        """Delete a sorting run by id."""
        self._execute(self._client.table("sorting").delete().eq("id", id))

    # ── neuron_info ──────────────────────────────────────────────────────────

    def get_neuron_info(
        self,
        filters: dict[str, Any] | None = None,
        as_dataframe: bool = False,
    ) -> list[NeuronInfo] | pd.DataFrame:
        """
        Fetch neuron metadata from the database.

        Parameters
        ----------
        filters : dict, optional
            Equality filters, e.g. ``{"sorting_id": 5}``.
        as_dataframe : bool, default False
            Return a pandas DataFrame instead of a list of NeuronInfo models.
        """
        q = apply_filters(
            self._client.table("neuron_info").select("*"), "neuron_info", filters
        )
        return self._to_models_or_df(self._execute(q), "neuron_info", as_dataframe)

    def insert_neuron_info(self, data: dict[str, Any] | NeuronInfo) -> NeuronInfo:
        """Insert a new neuron info record and return the created NeuronInfo."""
        payload = (
            data.model_dump(by_alias=True) if isinstance(data, NeuronInfo) else data
        )
        rows = self._execute(
            self._client.table("neuron_info").insert(payload)
        )
        if not rows:
            raise ArchiveError("Insert succeeded but the database returned no data.")
        return NeuronInfo.model_validate(rows[0])

    def update_neuron_info(self, id: int, data: dict[str, Any]) -> NeuronInfo:
        """Update a neuron info record by id and return the updated NeuronInfo."""
        rows = self._execute(
            self._client.table("neuron_info").update(data).eq("id", id)
        )
        if not rows:
            raise RecordNotFoundError(f"No neuron_info found with id={id}.")
        return NeuronInfo.model_validate(rows[0])

    def delete_neuron_info(self, id: int) -> None:
        """Delete a neuron info record by id."""
        self._execute(self._client.table("neuron_info").delete().eq("id", id))

    # ── spike_times ──────────────────────────────────────────────────────────

    def get_spike_times(
        self,
        filters: dict[str, Any] | None = None,
        as_dataframe: bool = False,
    ) -> list[SpikeTimes] | pd.DataFrame:
        """
        Fetch spike times from the database.

        Parameters
        ----------
        filters : dict, optional
            Equality filters, e.g. ``{"neuron_id": 42}``.
        as_dataframe : bool, default False
            Return a pandas DataFrame instead of a list of SpikeTimes models.
        """
        q = apply_filters(
            self._client.table("spike_times").select("*"), "spike_times", filters
        )
        return self._to_models_or_df(self._execute(q), "spike_times", as_dataframe)

    def insert_spike_times(self, data: dict[str, Any] | SpikeTimes) -> SpikeTimes:
        """Insert a new spike time record and return the created SpikeTimes."""
        payload = (
            data.model_dump(by_alias=True) if isinstance(data, SpikeTimes) else data
        )
        rows = self._execute(
            self._client.table("spike_times").insert(payload)
        )
        if not rows:
            raise ArchiveError("Insert succeeded but the database returned no data.")
        return SpikeTimes.model_validate(rows[0])

    def update_spike_times(self, id: int, data: dict[str, Any]) -> SpikeTimes:
        """Update a spike time record by id and return the updated SpikeTimes."""
        rows = self._execute(
            self._client.table("spike_times").update(data).eq("id", id)
        )
        if not rows:
            raise RecordNotFoundError(f"No spike_times found with id={id}.")
        return SpikeTimes.model_validate(rows[0])

    def delete_spike_times(self, id: int) -> None:
        """Delete a spike time record by id."""
        self._execute(self._client.table("spike_times").delete().eq("id", id))

    # ── tensor ───────────────────────────────────────────────────────────────

    def get_tensor(
        self,
        filters: dict[str, Any] | None = None,
        as_dataframe: bool = False,
    ) -> list[Tensor] | pd.DataFrame:
        """
        Fetch tensor rows from the database.

        Each row is one (neuron, stimuli, trial) combination whose full
        firing-rate timeseries is stored as a base64-encoded blob in the
        ``firing_rates`` column.
        To reconstruct a 4-D numpy array use :meth:`get_tensor_matrix` instead.

        Parameters
        ----------
        filters : dict, optional
            Equality filters, e.g. ``{"animal_id": "mouse01", "stimuli": "grating_0"}``.
        as_dataframe : bool, default False
            Return a pandas DataFrame instead of a list of Tensor models.
        """
        q = apply_filters(
            self._client.table("tensor").select("*"), "tensor", filters
        )
        return self._to_models_or_df(self._execute(q), "tensor", as_dataframe)

    def insert_tensor(self, data: dict[str, Any] | Tensor) -> Tensor:
        """Insert a single tensor row and return the created Tensor."""
        payload = data.model_dump(by_alias=True) if isinstance(data, Tensor) else data
        rows = self._execute(self._client.table("tensor").insert(payload))
        if not rows:
            raise ArchiveError("Insert succeeded but the database returned no data.")
        return Tensor.model_validate(rows[0])

    def update_tensor(self, id: int, data: dict[str, Any]) -> Tensor:
        """Update a tensor row by id and return the updated Tensor."""
        rows = self._execute(
            self._client.table("tensor").update(data).eq("id", id)
        )
        if not rows:
            raise RecordNotFoundError(f"No tensor row found with id={id}.")
        return Tensor.model_validate(rows[0])

    def delete_tensor(self, id: int) -> None:
        """Delete a tensor row by id."""
        self._execute(self._client.table("tensor").delete().eq("id", id))

    # ── Relation queries ─────────────────────────────────────────────────────

    def get_recording_with_animal(
        self,
        filters: dict[str, Any] | None = None,
        as_dataframe: bool = False,
    ) -> list[Recording] | pd.DataFrame:
        """
        Fetch recording sessions with nested animal data.

        Each result includes an ``animal`` key containing the related Animal row.

        Parameters
        ----------
        filters : dict, optional
            Equality filters applied to the recording table.
        as_dataframe : bool, default False
            Return a pandas DataFrame instead of a list of Recording models.
        """
        q = apply_filters(
            self._client.table("recording").select("*, animal(*)"),
            "recording",
            filters,
        )
        return self._to_models_or_df(self._execute(q), "recording", as_dataframe)

    def get_sorting_with_recording(
        self,
        filters: dict[str, Any] | None = None,
        as_dataframe: bool = False,
    ) -> list[Sorting] | pd.DataFrame:
        """
        Fetch sorting runs with nested recording data.

        Each result includes a ``recording`` key containing the related Recording row.

        Parameters
        ----------
        filters : dict, optional
            Equality filters applied to the sorting table.
        as_dataframe : bool, default False
            Return a pandas DataFrame instead of a list of Sorting models.
        """
        q = apply_filters(
            self._client.table("sorting").select("*, recording(*)"),
            "sorting",
            filters,
        )
        return self._to_models_or_df(self._execute(q), "sorting", as_dataframe)

    def get_neuron_info_with_sorting(
        self,
        filters: dict[str, Any] | None = None,
        as_dataframe: bool = False,
    ) -> list[NeuronInfo] | pd.DataFrame:
        """
        Fetch neuron info with nested sorting data.

        Each result includes a ``sorting`` key containing the related Sorting row.

        Parameters
        ----------
        filters : dict, optional
            Equality filters applied to the neuron_info table.
        as_dataframe : bool, default False
            Return a pandas DataFrame instead of a list of NeuronInfo models.
        """
        q = apply_filters(
            self._client.table("neuron_info").select("*, sorting(*)"),
            "neuron_info",
            filters,
        )
        return self._to_models_or_df(self._execute(q), "neuron_info", as_dataframe)

    def get_tensor_with_relations(
        self,
        filters: dict[str, Any] | None = None,
        as_dataframe: bool = False,
    ) -> list[Tensor] | pd.DataFrame:
        """
        Fetch tensor rows with nested animal and sorting data.

        Each result includes ``animal`` and ``sorting`` keys with related rows.

        Parameters
        ----------
        filters : dict, optional
            Equality filters applied to the tensor table.
        as_dataframe : bool, default False
            Return a pandas DataFrame instead of a list of Tensor models.
        """
        q = apply_filters(
            self._client.table("tensor").select("*, animal(*), sorting(*)"),
            "tensor",
            filters,
        )
        return self._to_models_or_df(self._execute(q), "tensor", as_dataframe)

    # ── Tensor matrix helpers ─────────────────────────────────────────────────

    def get_tensor_matrix(
        self,
        filters: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, list[int], list[str], list[int]]:
        """
        Fetch tensor rows and reconstruct a 4-D numpy array.

        Each row in the database holds the full firing-rate timeseries for one
        (neuron, stimuli, trial) combination as a base64-encoded blob.
        This method decodes all blobs and stacks them into a single tensor.

        Parameters
        ----------
        filters : dict, optional
            Equality filters, e.g. ``{"animal_id": "mouse01", "sorting_id": 3}``.

        Returns
        -------
        tensor : np.ndarray, shape (n_neurons, n_stimuli, n_trials, n_timepoints)
            Firing rates, dtype float32. NaN where no data exists.
        neuron_ids : list[int]
            Sorted unique neuron IDs — index into axis 0.
        stimuli_labels : list[str]
            Sorted unique stimuli labels — index into axis 1.
        trial_ids : list[int]
            Sorted unique trial numbers — index into axis 2.

        Examples
        --------
        >>> tensor, neurons, stimuli, trials = client.get_tensor_matrix(
        ...     {"animal_id": "mouse01", "sorting_id": 3}
        ... )
        >>> tensor.shape
        (12, 5, 20, 600)
        >>> # Full firing-rate timeseries for neuron 0, stimulus 0, trial 0
        >>> tensor[0, 0, 0]
        array([0.12, 0.45, ...], dtype=float32)
        """
        tensor_rows = self.get_tensor(filters=filters)
        return rows_to_tensor(tensor_rows)

    def upload_tensor_from_matrix(
        self,
        tensor: np.ndarray,
        neuron_ids: list[int],
        stimuli_labels: list[str],
        animal_id: str,
        sorting_id: int,
        batch_size: int = 500,
    ) -> int:
        """
        Serialise a 4-D numpy array and insert one row per (neuron, stimuli, trial)
        into the tensor table.

        The firing-rate timeseries (axis 3) is encoded as a base64 .npy blob and
        stored in the ``firing_rates`` column, so each DB row represents a full
        trial curve rather than a single scalar value.

        Parameters
        ----------
        tensor : np.ndarray, shape (n_neurons, n_stimuli, n_trials, n_timepoints)
            Firing rates. Slices where all timepoints are NaN are skipped.
            Values are cast to float32 before serialisation.
        neuron_ids : list[int]
            Labels for axis 0. Must match ``tensor.shape[0]``.
        stimuli_labels : list[str]
            Labels for axis 1. Must match ``tensor.shape[1]``.
        animal_id : str
            FK to the animal table. Written to every inserted row.
        sorting_id : int
            FK to the sorting table. Written to every inserted row.
        batch_size : int, default 500
            Number of rows per Supabase request.
            Each row is ~few KB, so 500 rows ≈ a few MB per request.

        Returns
        -------
        int
            Total number of rows inserted.

        Examples
        --------
        >>> import numpy as np
        >>> # 10 neurons, 4 stimuli, 20 trials, 600 timepoints
        >>> t = np.random.rand(10, 4, 20, 600).astype(np.float32)
        >>> n_inserted = client.upload_tensor_from_matrix(
        ...     t,
        ...     neuron_ids=list(range(10)),
        ...     stimuli_labels=["A", "B", "Bc", "M"],
        ...     animal_id="mouse01",
        ...     sorting_id=3,
        ... )
        >>> print(f"Inserted {n_inserted} rows")
        Inserted 800 rows   # 10 × 4 × 20 — vs 480 000 rows in the old schema
        """
        rows = tensor_to_rows(tensor, neuron_ids, stimuli_labels, animal_id, sorting_id)
        inserted = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            self._execute(self._client.table("tensor").insert(batch))
            inserted += len(batch)
        return inserted