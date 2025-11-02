"""
Unit tests for validation logic.
"""
import pytest
from src.models.schemas import TrainData, validate_series_id
from src.exceptions import ValidationError, InvalidSeriesIdError


class TestTrainDataValidation:
    """Tests for TrainData validation."""

    def test_valid_train_data(self):
        """Test valid training data."""
        data = TrainData(
            timestamps=[1, 2, 3, 4, 5],
            values=[10.0, 10.5, 10.2, 10.3, 10.1]
        )
        assert len(data.timestamps) == 5
        assert len(data.values) == 5

    def test_minimum_data_points_timestamps(self):
        """Test minimum data points validation for timestamps."""
        with pytest.raises(ValidationError) as exc_info:
            TrainData(
                timestamps=[1, 2],
                values=[10.0, 10.5]
            )
        assert "Minimum 3 data points required" in str(exc_info.value.message)

    def test_minimum_data_points_values(self):
        """Test minimum data points validation for values."""
        with pytest.raises(ValidationError) as exc_info:
            TrainData(
                timestamps=[1, 2],
                values=[10.0, 10.5]
            )
        assert "Minimum 3 data points required" in str(exc_info.value.message)

    def test_empty_timestamps(self):
        """Test empty timestamps list."""
        with pytest.raises(ValidationError) as exc_info:
            TrainData(
                timestamps=[],
                values=[10.0, 10.5, 10.2]
            )
        assert "cannot be empty" in str(exc_info.value.message).lower()

    def test_empty_values(self):
        """Test empty values list."""
        with pytest.raises(ValidationError) as exc_info:
            TrainData(
                timestamps=[1, 2, 3],
                values=[]
            )
        assert "cannot be empty" in str(exc_info.value.message).lower()

    def test_constant_values(self):
        """Test constant values (std = 0)."""
        with pytest.raises(ValidationError) as exc_info:
            TrainData(
                timestamps=[1, 2, 3, 4, 5],
                values=[10.0, 10.0, 10.0, 10.0, 10.0]
            )
        assert "constant values" in str(exc_info.value.message).lower()
        assert "standard deviation" in str(exc_info.value.message).lower()

    def test_nan_values(self):
        """Test NaN values detection."""
        with pytest.raises(ValidationError) as exc_info:
            TrainData(
                timestamps=[1, 2, 3, 4, 5],
                values=[10.0, 10.5, float('nan'), 10.3, 10.1]
            )
        assert "NaN" in str(exc_info.value.message)

    def test_infinite_values(self):
        """Test infinite values detection."""
        with pytest.raises(ValidationError) as exc_info:
            TrainData(
                timestamps=[1, 2, 3, 4, 5],
                values=[10.0, 10.5, float('inf'), 10.3, 10.1]
            )
        assert "Infinite" in str(exc_info.value.message)

    def test_unordered_timestamps(self):
        """Test unordered timestamps validation."""
        with pytest.raises(ValidationError) as exc_info:
            TrainData(
                timestamps=[1, 3, 2, 4, 5],
                values=[10.0, 10.5, 10.2, 10.3, 10.1]
            )
        assert "ascending order" in str(exc_info.value.message).lower()

    def test_to_time_series_mismatched_lengths(self):
        """Test mismatched array lengths."""
        with pytest.raises(ValidationError):
            TrainData(
                timestamps=[1, 2, 3],
                values=[10.0, 10.5, 10.2, 10.3]  # Different length
            )
        # Should be caught by Pydantic or our validator


class TestSeriesIdValidation:
    """Tests for series_id validation."""

    def test_valid_series_id_simple(self):
        """Test valid simple series_id."""
        validate_series_id("sensor_001")
        # Should not raise

    def test_valid_series_id_with_hyphen(self):
        """Test valid series_id with hyphen."""
        validate_series_id("sensor-001")
        # Should not raise

    def test_valid_series_id_with_dot(self):
        """Test valid series_id with dot."""
        validate_series_id("sensor.001")
        # Should not raise

    def test_valid_series_id_alphanumeric(self):
        """Test valid alphanumeric series_id."""
        validate_series_id("SensorABC123")
        # Should not raise

    def test_empty_series_id(self):
        """Test empty series_id."""
        with pytest.raises(InvalidSeriesIdError) as exc_info:
            validate_series_id("")
        assert "cannot be empty" in str(exc_info.value.message).lower()

    def test_path_traversal_dotdot(self):
        """Test path traversal with .."""
        with pytest.raises(InvalidSeriesIdError) as exc_info:
            validate_series_id("../malicious")
        assert "path traversal" in str(exc_info.value.message).lower()

    def test_path_traversal_slash(self):
        """Test path traversal with /."""
        with pytest.raises(InvalidSeriesIdError) as exc_info:
            validate_series_id("path/to/file")
        assert (
            "path traversal" in str(exc_info.value.message).lower()
            or "invalid characters" in str(exc_info.value.message).lower()
        )

    def test_path_traversal_backslash(self):
        """Test path traversal with backslash."""
        with pytest.raises(InvalidSeriesIdError) as exc_info:
            validate_series_id("path\\to\\file")
        assert (
            "path traversal" in str(exc_info.value.message).lower()
            or "invalid characters" in str(exc_info.value.message).lower()
        )

    def test_special_characters(self):
        """Test special characters not allowed."""
        with pytest.raises(InvalidSeriesIdError) as exc_info:
            validate_series_id("sensor@001")
        assert (
            "invalid characters" in str(exc_info.value.message).lower()
            or "can only contain" in str(exc_info.value.message).lower()
        )

    def test_too_long_series_id(self):
        """Test series_id that exceeds maximum length."""
        long_id = "a" * 101
        with pytest.raises(InvalidSeriesIdError) as exc_info:
            validate_series_id(long_id)
        assert "too long" in str(exc_info.value.message).lower()

    def test_maximum_length_series_id(self):
        """Test series_id at maximum allowed length."""
        max_id = "a" * 100
        validate_series_id(max_id)
        # Should not raise
