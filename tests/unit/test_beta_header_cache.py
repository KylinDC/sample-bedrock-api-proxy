"""Tests for BetaHeaderConfigCache and the enable_beta_header_dynamo config flag."""
from unittest.mock import MagicMock, patch

from app.db.beta_header_cache import BetaHeaderConfigCache


def _fresh_cache() -> BetaHeaderConfigCache:
    """Return a new (non-singleton) cache instance for testing."""
    return BetaHeaderConfigCache(refresh_interval=300)


# Patch targets: DynamoDBClient and BetaHeaderManager are imported locally inside
# start() via `from app.db.dynamodb import ...`, so we patch at the source module.
_DYNAMO_CLIENT = "app.db.dynamodb.DynamoDBClient"
_BETA_MANAGER = "app.db.dynamodb.BetaHeaderManager"


class TestBetaHeaderCacheDynamoDisabled:
    def test_loads_defaults_without_dynamodb(self):
        """When ENABLE_BETA_HEADER_DYNAMO=False, cache loads config defaults without touching DynamoDB."""
        cache = _fresh_cache()

        with patch("app.db.beta_header_cache.settings") as mock_settings:
            mock_settings.enable_beta_header_dynamo = False
            mock_settings.beta_headers_blocklist = ["some-unsupported-beta"]
            mock_settings.beta_header_mapping = {"client-beta": ["bedrock-beta"]}

            with patch(_DYNAMO_CLIENT) as mock_dynamo_cls:
                cache.start()
                mock_dynamo_cls.assert_not_called()

        assert "some-unsupported-beta" in cache.get_blocklist()
        assert cache.get_mapping() == {"client-beta": ["bedrock-beta"]}

    def test_no_background_timer_when_disabled(self):
        """When ENABLE_BETA_HEADER_DYNAMO=False, no periodic refresh timer is started."""
        cache = _fresh_cache()

        with patch("app.db.beta_header_cache.settings") as mock_settings:
            mock_settings.enable_beta_header_dynamo = False
            mock_settings.beta_headers_blocklist = []
            mock_settings.beta_header_mapping = {}

            cache.start()

        assert cache._timer is None

    def test_cache_loaded_flag_set(self):
        """Cache marks itself as loaded after start() even without DynamoDB."""
        cache = _fresh_cache()

        with patch("app.db.beta_header_cache.settings") as mock_settings:
            mock_settings.enable_beta_header_dynamo = False
            mock_settings.beta_headers_blocklist = []
            mock_settings.beta_header_mapping = {}

            cache.start()

        assert cache._loaded is True


class TestBetaHeaderCacheDynamoEnabled:
    def test_falls_back_to_defaults_on_dynamo_error(self):
        """When DynamoDB is unreachable, cache falls back to config defaults gracefully."""
        cache = _fresh_cache()

        mock_manager = MagicMock()
        mock_manager.list_all.side_effect = Exception("Connection refused")

        with patch("app.db.beta_header_cache.settings") as mock_settings:
            mock_settings.enable_beta_header_dynamo = True
            mock_settings.beta_headers_blocklist = ["fallback-header"]
            mock_settings.beta_header_mapping = {}

            with patch(_DYNAMO_CLIENT):
                with patch(_BETA_MANAGER, return_value=mock_manager):
                    cache.start()
                    cache.stop()

        assert "fallback-header" in cache.get_blocklist()
        assert cache._loaded is True

    def test_loads_rules_from_dynamo_when_available(self):
        """When DynamoDB has data, cache uses it instead of config defaults."""
        cache = _fresh_cache()

        mock_manager = MagicMock()
        mock_manager.list_all.return_value = [
            {"header_name": "dynamo-blocked", "header_type": "blocklist"},
            {"header_name": "dynamo-src", "header_type": "mapping", "mapped_to": ["dynamo-dest"]},
        ]

        with patch("app.db.beta_header_cache.settings") as mock_settings:
            mock_settings.enable_beta_header_dynamo = True
            mock_settings.beta_headers_blocklist = ["config-blocked"]
            mock_settings.beta_header_mapping = {}

            with patch(_DYNAMO_CLIENT):
                with patch(_BETA_MANAGER, return_value=mock_manager):
                    cache.start()
                    cache.stop()

        assert "dynamo-blocked" in cache.get_blocklist()
        assert "config-blocked" not in cache.get_blocklist()
        assert cache.get_mapping() == {"dynamo-src": ["dynamo-dest"]}
