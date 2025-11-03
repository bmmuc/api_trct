"""
Factory to create different model types.
"""
from src.anomaly_models.base_model import BaseAnomalyModel
from src.anomaly_models.statistical_model import StatisticalAnomalyModel
from src.anomaly_models.sklearn_model import SklearnAnomalyModel


class ModelFactory:  # pylint: disable=too-few-public-methods
    """Creates models based on configured type."""

    _registry = {
        "statistical": StatisticalAnomalyModel,
        "sklearn": SklearnAnomalyModel,
    }

    @classmethod
    def create(cls, model_type: str, **kwargs) -> BaseAnomalyModel:
        """Instantiates model with its specific parameters."""
        if model_type not in cls._registry:
            raise ValueError(f"Type {model_type} not supported")

        model_class = cls._registry[model_type]
        return model_class(**kwargs)
