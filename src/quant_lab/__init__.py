"""quant_lab — cross-project experiment index and comparison."""

__version__ = "0.3.0"
from quant_lab.contracts import RunManifest, load_and_validate_run, write_standard_run
from quant_lab.contracts_v2 import (
    ArtifactRecordV2,
    RunManifestV2,
    load_and_validate_run_v2,
    load_and_validate_standard_run,
    write_standard_run_v2,
)

__all__ = [
    "ArtifactRecordV2",
    "RunManifest",
    "RunManifestV2",
    "load_and_validate_run",
    "load_and_validate_run_v2",
    "load_and_validate_standard_run",
    "write_standard_run",
    "write_standard_run_v2",
]
