import unittest
from unittest.mock import MagicMock
from src.services.anomaly_service import AnomalyDetectionService
from src.models.schemas import TrainData, DataPoint, TrainResponse
from src.anomaly_models.statistical_model import StatisticalAnomalyModel


class TestAnomalyDetectionService(unittest.TestCase):

    def setUp(self):
        """Set up mocks for ModelStorage, MetricsExporter, and TrainingService."""
        self.mock_model_storage = MagicMock()
        self.mock_metrics_exporter = MagicMock()
        self.mock_training_service = MagicMock()
        self.service = AnomalyDetectionService(
            model_storage=self.mock_model_storage,
            metrics_exporter=self.mock_metrics_exporter,
            training_service=self.mock_training_service
        )

    def test_train_model(self):
        """Tests the model training flow."""
        series_id = "test-series"
        train_data = TrainData(
            timestamps=[1.0, 2.0, 3.0, 4.0, 5.0],
            values=[1.0, 1.1, 1.2, 5.0, 1.3])

        # Mock the training service response
        self.mock_training_service.train.return_value = TrainResponse(
            series_id=series_id,
            version="v1",
            points_used=5
        )

        # Call the training method
        response = self.service.train_model(series_id, train_data)

        # Checks that the training service was called
        self.mock_training_service.train.assert_called_once_with(series_id, train_data, None)

        # Checks the response
        self.assertEqual(response.series_id, series_id)
        self.assertEqual(response.version, "v1")
        self.assertEqual(response.points_used, 5)

    def test_predict_anomaly(self):
        """Tests the anomaly prediction flow."""
        series_id = "test-series"
        data_point = DataPoint(timestamp=1.0, value=10.0)

        mock_model = StatisticalAnomalyModel()
        mock_model.mean = 1.0
        mock_model.std = 0.5

        self.mock_model_storage.load_model.return_value = (mock_model, "v1")

        response = self.service.predict_anomaly(series_id, data_point)

        self.mock_model_storage.load_model.assert_called_once_with(series_id, None)

        self.assertTrue(response.anomaly)
        self.assertEqual(response.model_version, "v1")

    def test_predict_anomaly_with_specific_version(self):
        """Tests prediction with a specified model version."""
        series_id = "test-series"
        data_point = DataPoint(timestamp=1.0, value=1.1)
        version = "v2"

        mock_model = StatisticalAnomalyModel()
        mock_model.mean = 1.0
        mock_model.std = 0.5

        self.mock_model_storage.load_model.return_value = (mock_model, version)

        response = self.service.predict_anomaly(series_id, data_point, version=version)

        self.mock_model_storage.load_model.assert_called_once_with(series_id, version)
        self.assertFalse(response.anomaly)
        self.assertEqual(response.model_version, version)

    def test_get_trained_series_count(self):
        """Tests the count of trained series."""
        self.mock_model_storage.list_all_series.return_value = ["series-1", "series-2"]

        count = self.service.get_trained_series_count()

        self.assertEqual(count, 2)
        self.mock_model_storage.list_all_series.assert_called_once()


if __name__ == '__main__':
    unittest.main()
