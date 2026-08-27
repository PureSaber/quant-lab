"""quant_lab — cross-project experiment index and comparison."""

__version__ = "0.2.0"
from quant_lab.contracts import RunManifest, load_and_validate_run, write_standard_run

__all__ = ["RunManifest", "load_and_validate_run", "write_standard_run"]
