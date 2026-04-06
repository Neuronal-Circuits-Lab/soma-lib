from .client import ArchiveClient
from .exceptions import (
    ArchiveError,
    AuthenticationError,
    FilterError,
    RecordNotFoundError,
    TableNotFoundError,
    TensorShapeError,
)
from .models import Animal, NeuronInfo, Recording, Sorting, SpikeTimes, Tensor

__version__ = "0.1.0"

__all__ = [
    "ArchiveClient",
    # Models
    "Animal",
    "Recording",
    "Sorting",
    "NeuronInfo",
    "SpikeTimes",
    "Tensor",
    # Exceptions
    "ArchiveError",
    "AuthenticationError",
    "TableNotFoundError",
    "RecordNotFoundError",
    "TensorShapeError",
    "FilterError",
]
