import unittest
import tempfile
import shutil
import json
from pathlib import Path
from src.storage.model_store import ModelStore
from src.anomaly_models.anomaly_model import AnomalyDetectionModel
from src.models.schemas import TrainData

class TestModelStore(unittest.TestCase):

    def setUp(self):
        """Creates a temporary directory for model storage and a train_data fixture."""
        self.test_dir = tempfile.mkdtemp()
        self.model_store = ModelStore(storage_path=self.test_dir)
        self.train_data = TrainData(
            timestamps=[1.0, 2.0, 3.0, 4.0, 5.0],
            values=[1.0, 1.1, 1.2, 5.0, 1.3]
        )

    def tearDown(self):
        """Removes the temporary directory after tests."""
        shutil.rmtree(self.test_dir)

    def _create_fitted_mock_model(self):
        """Creates a mock AnomalyDetectionModel and fits it using the train_data fixture."""
        model = AnomalyDetectionModel()
        model.fit(self.train_data.to_time_series())
        return model

    def test_save_model(self):
        """Tests saving a new model."""
        series_id = "series-1"
        model = self._create_fitted_mock_model()
        
        version = self.model_store.save_model(series_id, model)
        self.assertEqual(version, "v0")

        model_path = Path(self.test_dir) / series_id / "v0.json"
        self.assertTrue(model_path.exists())

        with open(model_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(data["series_id"], series_id)
            self.assertEqual(data["version"], "v0")
            self.assertIn("model_params", data)

    def test_save_unfitted_model_raises_error(self):
        """Tests that saving an unfitted model raises a ValueError."""
        series_id = "series-1"
        model = AnomalyDetectionModel()
        with self.assertRaises(ValueError):
            self.model_store.save_model(series_id, model)

    def test_load_model(self):
        """Tests loading an existing model."""
        series_id = "series-2"
        model = self._create_fitted_mock_model()
        version = self.model_store.save_model(series_id, model)

        loaded_model, loaded_version = self.model_store.load_model(series_id, version)
        self.assertIsInstance(loaded_model, AnomalyDetectionModel)
        self.assertTrue(loaded_model._is_fitted)
        self.assertEqual(loaded_version, version)

    def test_load_nonexistent_model_raises_error(self):
        """Tests that loading a non-existent model raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.model_store.load_model("nonexistent-series")

    def test_versioning(self):
        """Tests automatic versioning logic."""
        series_id = "series-3"
        model = self._create_fitted_mock_model()

        v0 = self.model_store.save_model(series_id, model)
        self.assertEqual(v0, "v0")

        v1 = self.model_store.save_model(series_id, model)
        self.assertEqual(v1, "v1")

        versions = self.model_store.list_versions(series_id)
        self.assertEqual(versions, ["v0", "v1"])

        latest_version = self.model_store.get_latest_version(series_id)
        self.assertEqual(latest_version, "v1")

    def test_load_latest_model(self):
        """Tests loading the most recent version of a model."""
        series_id = "series-4"
        model = self._create_fitted_mock_model()
        
        self.model_store.save_model(series_id, model) # v0
        self.model_store.save_model(series_id, model) # v1

        loaded_model, loaded_version = self.model_store.load_model(series_id) # Without version specified
        self.assertEqual(loaded_version, "v1")
        self.assertTrue(loaded_model._is_fitted)

    def test_list_all_series(self):
        """Tests listing all series with saved models."""
        model = self._create_fitted_mock_model()

        model.fit(self.train_data.to_time_series())
        self.model_store.save_model("series-a", model)
        self.model_store.save_model("series-b", model)
        
        all_series = self.model_store.list_all_series()
        self.assertIn("series-a", all_series)
        self.assertIn("series-b", all_series)
        self.assertEqual(len(all_series), 2)
