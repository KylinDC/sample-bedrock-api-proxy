"""Tests for UsageTracker and the enable_usage_tracking config flag."""
import pytest
from unittest.mock import MagicMock, patch
from moto import mock_aws

from app.db.dynamodb import DynamoDBClient, UsageTracker


@pytest.fixture
def mock_dynamodb_client():
    with mock_aws():
        import boto3
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        dynamodb.create_table(
            TableName="anthropic-proxy-usage",
            KeySchema=[
                {"AttributeName": "api_key", "KeyType": "HASH"},
                {"AttributeName": "timestamp", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "api_key", "AttributeType": "S"},
                {"AttributeName": "timestamp", "AttributeType": "S"},
                {"AttributeName": "request_id", "AttributeType": "S"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "request_id-index",
                    "KeySchema": [{"AttributeName": "request_id", "KeyType": "HASH"}],
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        client = MagicMock(spec=DynamoDBClient)
        client.dynamodb = dynamodb
        client.usage_table_name = "anthropic-proxy-usage"
        yield client


@pytest.fixture
def usage_tracker(mock_dynamodb_client):
    return UsageTracker(mock_dynamodb_client)


SAMPLE_USAGE = dict(
    api_key="sk-test-key",
    request_id="req-123",
    model="claude-sonnet-4-6",
    input_tokens=100,
    output_tokens=50,
)


class TestUsageTrackerRecordUsage:
    def test_records_usage_when_tracking_enabled(self, usage_tracker, mock_dynamodb_client):
        """record_usage writes to DynamoDB when enable_usage_tracking=True."""
        with patch("app.db.dynamodb.settings") as mock_settings:
            mock_settings.enable_usage_tracking = True
            mock_settings.usage_ttl_days = 0

            usage_tracker.record_usage(**SAMPLE_USAGE)

        table = mock_dynamodb_client.dynamodb.Table("anthropic-proxy-usage")
        items = table.scan()["Items"]
        assert len(items) == 1
        assert items[0]["api_key"] == "sk-test-key"
        assert items[0]["request_id"] == "req-123"
        assert items[0]["input_tokens"] == 100
        assert items[0]["output_tokens"] == 50

    def test_skips_write_when_tracking_disabled(self, usage_tracker, mock_dynamodb_client):
        """record_usage is a no-op when enable_usage_tracking=False."""
        with patch("app.db.dynamodb.settings") as mock_settings:
            mock_settings.enable_usage_tracking = False
            mock_settings.usage_ttl_days = 0

            usage_tracker.record_usage(**SAMPLE_USAGE)

        table = mock_dynamodb_client.dynamodb.Table("anthropic-proxy-usage")
        items = table.scan()["Items"]
        assert len(items) == 0

    def test_tracking_disabled_does_not_raise(self, usage_tracker):
        """Disabling usage tracking never raises even if DynamoDB is unreachable."""
        with patch("app.db.dynamodb.settings") as mock_settings:
            mock_settings.enable_usage_tracking = False

            # Should not raise regardless of DynamoDB state
            usage_tracker.record_usage(**SAMPLE_USAGE)

    def test_records_success_flag(self, usage_tracker, mock_dynamodb_client):
        with patch("app.db.dynamodb.settings") as mock_settings:
            mock_settings.enable_usage_tracking = True
            mock_settings.usage_ttl_days = 0

            usage_tracker.record_usage(**SAMPLE_USAGE, success=True)

        items = mock_dynamodb_client.dynamodb.Table("anthropic-proxy-usage").scan()["Items"]
        assert items[0]["success"] is True

    def test_records_error_message(self, usage_tracker, mock_dynamodb_client):
        with patch("app.db.dynamodb.settings") as mock_settings:
            mock_settings.enable_usage_tracking = True
            mock_settings.usage_ttl_days = 0

            usage_tracker.record_usage(
                **SAMPLE_USAGE,
                success=False,
                error_message="Throttling error",
            )

        items = mock_dynamodb_client.dynamodb.Table("anthropic-proxy-usage").scan()["Items"]
        assert items[0]["error_message"] == "Throttling error"

    def test_records_cache_ttl(self, usage_tracker, mock_dynamodb_client):
        with patch("app.db.dynamodb.settings") as mock_settings:
            mock_settings.enable_usage_tracking = True
            mock_settings.usage_ttl_days = 0

            usage_tracker.record_usage(**SAMPLE_USAGE, cache_ttl="1h")

        items = mock_dynamodb_client.dynamodb.Table("anthropic-proxy-usage").scan()["Items"]
        assert items[0]["cache_ttl"] == "1h"

    def test_ttl_field_added_when_enabled(self, usage_tracker, mock_dynamodb_client):
        with patch("app.db.dynamodb.settings") as mock_settings:
            mock_settings.enable_usage_tracking = True
            mock_settings.usage_ttl_days = 7

            usage_tracker.record_usage(**SAMPLE_USAGE)

        items = mock_dynamodb_client.dynamodb.Table("anthropic-proxy-usage").scan()["Items"]
        assert "ttl" in items[0]

    def test_ttl_field_absent_when_days_zero(self, usage_tracker, mock_dynamodb_client):
        with patch("app.db.dynamodb.settings") as mock_settings:
            mock_settings.enable_usage_tracking = True
            mock_settings.usage_ttl_days = 0

            usage_tracker.record_usage(**SAMPLE_USAGE)

        items = mock_dynamodb_client.dynamodb.Table("anthropic-proxy-usage").scan()["Items"]
        assert "ttl" not in items[0]
