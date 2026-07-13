from unittest.mock import patch, MagicMock


class TestIsRemoteConfigured:
    def test_returns_true_when_config_has_endpoints(self):
        from ttk.remote import is_remote_configured
        mock_config = MagicMock()
        mock_config.endpoints = [MagicMock()]  # non-empty
        with patch("ttk.remote.config.get_remote_config", return_value=mock_config):
            assert is_remote_configured() is True

    def test_returns_false_when_config_has_no_endpoints(self):
        from ttk.remote import is_remote_configured
        mock_config = MagicMock()
        mock_config.endpoints = []
        with patch("ttk.remote.config.get_remote_config", return_value=mock_config):
            assert is_remote_configured() is False

    def test_returns_false_when_config_is_none(self):
        from ttk.remote import is_remote_configured
        with patch("ttk.remote.config.get_remote_config", return_value=None):
            assert is_remote_configured() is False
