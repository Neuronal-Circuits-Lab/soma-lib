from __future__ import annotations

from typing import Any

from .exceptions import FilterError

# Valid filterable columns per table. Used to catch typos before the network call.
_VALID_FILTERS: dict[str, set[str]] = {
    "animal": {"animal_id", "type", "sex", "cage", "mark"},
    "recording": {"id", "animal_id", "recoding_date", "sample"},
    "sorting": {"id", "recording_id", "sorting_date"},
    "neuron_info": {"id", "neuron_id", "sorting_id"},
    "spike_times": {"id", "neuron_id"},
    "tensor": {"id", "sorting_id", "neuron_id", "trial", "animal_id", "stimuli"},
}


def apply_filters(query: Any, table: str, filters: dict[str, Any] | None) -> Any:
    """
    Apply equality filters from a dict to a supabase-py query builder.

    Parameters
    ----------
    query:
        A supabase-py query builder (result of .select(), .update(), etc.)
    table:
        The table name, used to validate filter keys.
    filters:
        Dict of {column: value} equality filters. Pass None or {} to skip.

    Returns
    -------
    The query builder with .eq() calls chained for each filter entry.

    Raises
    ------
    FilterError
        If any key is not a recognised column for the given table.
    """
    if not filters:
        return query

    valid = _VALID_FILTERS.get(table, set())
    for key, value in filters.items():
        if valid and key not in valid:
            raise FilterError(
                f"'{key}' is not a valid filter for the '{table}' table. "
                f"Valid filter columns are: {sorted(valid)}. "
                f"Check for typos in your filter key."
            )
        query = query.eq(key, value)

    return query
