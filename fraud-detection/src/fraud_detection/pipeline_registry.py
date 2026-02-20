"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from fraud_detection.pipelines.data_processing.pipeline import create_pipeline as create_data_processing_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["data_processing"] = create_data_processing_pipeline()
    pipelines["__default__"] = pipelines["data_processing"]
    return pipelines
