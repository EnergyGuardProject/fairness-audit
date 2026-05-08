"""UploadedInputBundle dataclass for the API file-upload flow.

The API endpoint writes uploaded files to a temp directory, then builds an
UploadedInputBundle before merging paths into the config dict and calling
RunConfig.model_validate().
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UploadedInputBundle:
    """Resolved temp-directory paths for an uploaded evaluation request.

    Created by app/main.py after saving uploaded files. Passed to the config
    merge step so the resolved paths are injected into the YAML dict before
    RunConfig validation.

    Attributes:
        model_path: Absolute path to the saved model file.
        dataset_path: Absolute path to the saved dataset file.
        config_bytes: Raw bytes of the uploaded YAML config.
        job_id: Pre-generated job_id used as the temp subdirectory name.
    """

    model_path: Path
    dataset_path: Path
    config_bytes: bytes
    job_id: str
