"""
Integration tests for API routes.
"""
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


class TestTrainingEndpoint:
    """Tests for /fit/{series_id} endpoint."""

    def test_train_model_success(self):
        """Test successful model training."""
        response = client.post(
            "/fit/sensor_001",
            json={
                "timestamps": [1609459200, 1609545600, 1609632000, 1609718400, 1609804800],
                "values": [23.5, 24.1, 23.8, 24.0, 23.9]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["series_id"] == "sensor_001"
        assert "version" in data
        assert data["points_used"] == 5

    def test_train_model_insufficient_data(self):
        """Test training with insufficient data points."""
        response = client.post(
            "/fit/sensor_002",
            json={
                "timestamps": [1609459200, 1609545600],
                "values": [23.5, 24.1]
            }
        )
        assert response.status_code == 422
        assert "Minimum 3 data points required" in response.text

    def test_train_model_constant_values(self):
        """Test training with constant values."""
        response = client.post(
            "/fit/sensor_003",
            json={
                "timestamps": [1609459200, 1609545600, 1609632000],
                "values": [10.0, 10.0, 10.0]
            }
        )
        assert response.status_code == 422
        assert "constant values" in response.text.lower()

    def test_train_model_invalid_series_id(self):
        """Test training with invalid series_id containing special characters."""
        response = client.post(
            "/fit/sensor@invalid",
            json={
                "timestamps": [1609459200, 1609545600, 1609632000],
                "values": [23.5, 24.1, 23.8]
            }
        )
        assert response.status_code == 400
        assert "invalid" in response.text.lower() or "characters" in response.text.lower()

    def test_train_model_unordered_timestamps(self):
        """Test training with unordered timestamps."""
        response = client.post(
            "/fit/sensor_004",
            json={
                "timestamps": [1609632000, 1609459200, 1609545600],
                "values": [23.5, 24.1, 23.8]
            }
        )
        assert response.status_code == 422
        assert "ascending order" in response.text.lower()


class TestPredictionEndpoint:
    """Tests for /predict/{series_id} endpoint."""

    def test_predict_anomaly_model_not_found(self):
        """Test prediction when model doesn't exist."""
        response = client.post(
            "/predict/nonexistent_series",
            json={
                "timestamp": "1609804800",
                "value": 50.0
            }
        )
        assert response.status_code == 404
        assert ("not found" in response.text.lower() or "no model found" in response.text.lower())

    def test_predict_after_training(self):
        """Test prediction after training a model."""
        # First train a model
        train_response = client.post(
            "/fit/sensor_predict_test",
            json={
                "timestamps": [1, 2, 3, 4, 5],
                "values": [10.0, 10.5, 10.2, 10.3, 10.1]
            }
        )
        assert train_response.status_code == 200

        # Then make a prediction (normal value)
        predict_response = client.post(
            "/predict/sensor_predict_test",
            json={
                "timestamp": "6",
                "value": 10.4
            }
        )
        assert predict_response.status_code == 200
        data = predict_response.json()
        assert "anomaly" in data
        assert "model_version" in data
        assert isinstance(data["anomaly"], bool)

        # Test with anomalous value
        predict_anomaly_response = client.post(
            "/predict/sensor_predict_test",
            json={
                "timestamp": "7",
                "value": 100.0  # Very different from training data
            }
        )
        assert predict_anomaly_response.status_code == 200
        data = predict_anomaly_response.json()
        assert data["anomaly"] is True


class TestHealthCheckEndpoint:  # pylint: disable=too-few-public-methods
    """Tests for /healthcheck endpoint."""

    def test_healthcheck_success(self):
        """Test health check endpoint."""
        response = client.get("/healthcheck")
        assert response.status_code == 200
        data = response.json()
        assert "series_trained" in data
        assert "inference_latency_ms" in data
        assert "training_latency_ms" in data
        assert isinstance(data["series_trained"], int)


class TestVisualizationEndpoint:
    """Tests for /plot/{series_id} endpoint."""

    def test_plot_model_not_found(self):
        """Test plot generation when model doesn't exist."""
        response = client.get("/plot/nonexistent_plot")
        assert response.status_code == 404

    def test_plot_after_training(self):
        """Test plot generation after training a model."""
        # First train a model
        train_response = client.post(
            "/fit/sensor_plot_test",
            json={
                "timestamps": [1, 2, 3, 4, 5],
                "values": [10.0, 10.5, 10.2, 10.3, 10.1]
            }
        )
        assert train_response.status_code == 200

        # Generate plot
        plot_response = client.get("/plot/sensor_plot_test")
        assert plot_response.status_code == 200
        assert plot_response.headers["content-type"] == "image/png"
        assert len(plot_response.content) > 0

    def test_plot_invalid_format(self):
        """Test plot with invalid format parameter."""
        # Train a model first
        client.post(
            "/fit/sensor_format_test",
            json={
                "timestamps": [1, 2, 3, 4, 5],
                "values": [10.0, 10.5, 10.2, 10.3, 10.1]
            }
        )

        # Request invalid format
        response = client.get("/plot/sensor_format_test?format=invalid")
        assert response.status_code == 422


class TestRootEndpoint:  # pylint: disable=too-few-public-methods
    """Tests for root endpoint."""

    def test_root_endpoint(self):
        """Test root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
