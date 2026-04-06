from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_supabase():
    """Patch create_client and return the mocked Supabase client."""
    with patch("soma_connect.client.create_client") as mock_create:
        mock_client = MagicMock()
        mock_client.supabase_url = "https://fake.supabase.co"
        mock_create.return_value = mock_client
        yield mock_client


@pytest.fixture
def client(mock_supabase):
    """An ArchiveClient wired to a mocked Supabase backend."""
    from soma_connect import ArchiveClient

    return ArchiveClient(url="https://fake.supabase.co", key="fake-key")
