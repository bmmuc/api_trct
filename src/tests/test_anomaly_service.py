import unittest
from unittest.mock import MagicMock
from src.services.anomaly_service import AnomalyDetectionService
from src.models.schemas import TrainData, DataPoint
from src.anomaly_models.anomaly_model import AnomalyDetectionModel


class TestAnomalyDetectionService(unittest.TestCase):

    def setUp(self):
        """Set up mocks for ModelStore and MetricsTracker."""
        self.mock_model_store = MagicMock()
        self.mock_metrics_tracker = MagicMock()
        self.service = AnomalyDetectionService(
            model_store=self.mock_model_store,
            metrics_tracker=self.mock_metrics_tracker
        )

    def test_train_model(self):
        """Tests the model training flow."""
        series_id = "test-series"
        train_data = TrainData(
            timestamps=[1.0, 2.0, 3.0, 4.0, 5.0],
            values=[1.0, 1.1, 1.2, 5.0, 1.3])

        self.mock_model_store.save_model.return_value = "v1"

        # Call the training method
        response = self.service.train_model(series_id, train_data)

        # Checks that the model was saved
        self.mock_model_store.save_model.assert_called_once()
        saved_model_arg = self.mock_model_store.save_model.call_args[0][1]
        self.assertIsInstance(saved_model_arg, AnomalyDetectionModel)
        self.assertTrue(saved_model_arg._is_fitted)  # noqa: SLF001 pylint: disable=protected-access

        # Checks the response
        self.assertEqual(response.series_id, series_id)
        self.assertEqual(response.version, "v1")
        self.assertEqual(response.points_used, 5)

    def test_predict_anomaly(self):
        """Tests the anomaly prediction flow."""
        series_id = "test-series"
        data_point = DataPoint(timestamp=1.0, value=10.0)

        mock_model = AnomalyDetectionModel()
        mock_model.mean = 1.0
        mock_model.std = 0.5

        self.mock_model_store.load_model.return_value = (mock_model, "v1")

        response = self.service.predict_anomaly(series_id, data_point)

        self.mock_model_store.load_model.assert_called_once_with(series_id, None)

        self.assertTrue(response.anomaly)
        self.assertEqual(response.model_version, "v1")

    def test_predict_anomaly_with_specific_version(self):
        """Tests prediction with a specified model version."""
        series_id = "test-series"
        data_point = DataPoint(timestamp=1.0, value=1.1)
        version = "v2"

        mock_model = AnomalyDetectionModel()
        mock_model.mean = 1.0
        mock_model.std = 0.5

        self.mock_model_store.load_model.return_value = (mock_model, version)

        response = self.service.predict_anomaly(series_id, data_point, version=version)

        self.mock_model_store.load_model.assert_called_once_with(series_id, version)
        self.assertFalse(response.anomaly)
        self.assertEqual(response.model_version, version)

    def test_get_trained_series_count(self):
        """Tests the count of trained series."""
        self.mock_model_store.list_all_series.return_value = ["series-1", "series-2"]

        count = self.service.get_trained_series_count()

        self.assertEqual(count, 2)
        self.mock_model_store.list_all_series.assert_called_once()


if __name__ == '__main__':
    unittest.main()
