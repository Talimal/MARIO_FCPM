"""TID3: Time Interval Duration Driven Discretization."""

from .tid3 import TID3, tid3
from .standardize import panel_to_long, load_uea_tsfile, read_ts_file
from .run import run_tid3
from .datasets import PAPER_UEA_DATASETS, download_uea_dataset, load_paper_dataset

__all__ = [
    "TID3",
    "tid3",
    "panel_to_long",
    "load_uea_tsfile",
    "read_ts_file",
    "run_tid3",
    "PAPER_UEA_DATASETS",
    "download_uea_dataset",
    "load_paper_dataset",
]
