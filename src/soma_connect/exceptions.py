class ArchiveError(Exception):
    """Base exception for all soma_connect errors."""


class AuthenticationError(ArchiveError):
    """
    Raised when Supabase credentials are missing or rejected.

    Make sure SUPABASE_URL and SUPABASE_KEY are set in your environment
    or in a .env file in your working directory.
    """


class TableNotFoundError(ArchiveError):
    """Raised when a query targets a table that does not exist."""


class RecordNotFoundError(ArchiveError):
    """Raised when an update or delete targets a row that does not exist."""


class TensorShapeError(ArchiveError):
    """
    Raised when a numpy array does not match the expected tensor dimensions.

    The tensor must be a 3-D array with shape (n_neurons, n_stimuli, n_trials).
    """


class FilterError(ArchiveError):
    """
    Raised when a filter dictionary contains an invalid column name.

    Check for typos in your filter keys — for example, use 'animal_id'
    not 'animla_id'.
    """
