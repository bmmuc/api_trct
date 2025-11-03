"""
Mock sklearn model for anomaly detection.
"""
from src.models.schemas import TimeSeries, DataPoint
from src.anomaly_models.base_model import BaseAnomalyModel


class SklearnAnomalyModel(BaseAnomalyModel):
    """Mock of IsolationForest or other sklearn algorithm."""

    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self._is_fitted = False
        # Mock: from sklearn.ensemble import IsolationForest
        # Mock: self.model = IsolationForest(
        #     contamination=contamination,
        #     n_estimators=n_estimators
        # )

    def fit(self, data: TimeSeries) -> "SklearnAnomalyModel":
        """Simulates sklearn model training."""
        # Mock: valores = [[d.value] for d in data.data]
        # Mock: self.model.fit(valores)
        self._is_fitted = True
        return self

    def predict(self, data_point: DataPoint) -> bool:
        """Simulates sklearn prediction."""
        # Mock: prediction = self.model.predict([[data_point.value]])
        # Mock: return prediction[0] == -1 (outlier)
        return False

    def save(self) -> bytes:
        """Serializes using pickle."""
        if not self._is_fitted:
            raise ValueError("Cannot serialize an unfitted model")

        # Mock: import pickle
        # Mock: return pickle.dumps({
        #     'model': self.model,
        #     'contamination': self.contamination,
        #     'n_estimators': self.n_estimators
        # })
        return b"mock_pickle_data"

    def load(self, data: bytes) -> "SklearnAnomalyModel":
        """Loads using pickle."""
        # Mock: import pickle
        # Mock: obj = pickle.loads(data)
        # Mock: self.model = obj['model']
        # Mock: self.contamination = obj['contamination']
        # Mock: self.n_estimators = obj['n_estimators']
        self._is_fitted = True
        return self

    def is_fitted(self) -> bool:
        """Checks if it was trained."""
        return self._is_fitted

    def get_model_type(self) -> str:
        """Returns model type."""
        return "sklearn"
