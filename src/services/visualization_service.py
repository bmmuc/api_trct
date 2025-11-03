"""
Visualization service using abstract storage.
"""
import io
from typing import Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402 pylint: disable=wrong-import-position
from src.storage.base_storage import BaseModelStorage  # noqa: E402 pylint: disable=wrong-import-position
from src.exceptions import ModelNotFoundError  # noqa: E402 pylint: disable=wrong-import-position
from src.utils.logger import logger  # noqa: E402 pylint: disable=wrong-import-position


class VisualizationService:  # pylint: disable=too-few-public-methods
    """Visualization service using abstract storage."""

    def __init__(self, model_storage: BaseModelStorage):
        self.model_storage = model_storage

    def plot_time_series(
        self,
        series_id: str,
        version: Optional[str] = None,
        img_format: str = 'png'
    ) -> bytes:
        """
        Generate a plot of the time series with anomaly detection boundaries.

        Args:
            series_id: Identifier for the time series
            version: Optional model version, uses latest if not provided
            img_format: Image format ('png', 'jpg', 'svg')

        Returns:
            Image bytes

        Raises:
            ModelNotFoundError: If model for series_id doesn't exist
        """
        try:
            model, used_version = self.model_storage.load_model(series_id, version)
        except FileNotFoundError as exc:
            logger.warning(
                "Model not found for visualization: series_id='%s', version='%s'",
                series_id, version
            )
            raise ModelNotFoundError(series_id, version) from exc

        fig, ax = plt.subplots(figsize=(12, 6))

        mean = model.mean
        std = model.std
        upper_bound = mean + 3 * std
        lower_bound = mean - 3 * std

        ax.axhline(
            y=mean, color='blue', linestyle='-', linewidth=2,
            label=f'Mean ({mean:.2f})'
        )
        ax.axhline(
            y=upper_bound, color='red', linestyle='--', linewidth=1.5,
            label=f'Upper Bound ({upper_bound:.2f})'
        )
        ax.axhline(
            y=lower_bound, color='red', linestyle='--', linewidth=1.5,
            label=f'Lower Bound ({lower_bound:.2f})'
        )

        ax.fill_between([0, 1], lower_bound, upper_bound, alpha=0.2, color='green',
                        label='Normal Range')

        ax.set_xlabel('Data Points', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(
            f'Time Series: {series_id} (Version: {used_version})',
            fontsize=14, fontweight='bold'
        )
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        stats_text = f'μ = {mean:.4f}\nσ = {std:.4f}\nThreshold = μ ± 3σ'
        ax.text(
            0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}
        )

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format=img_format, dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        logger.info(
            "Generated plot for series_id='%s', version='%s'",
            series_id, used_version
        )

        return buf.read()
