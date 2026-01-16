"""
Configuration management for srt_zero.

Provides settings for precision, tolerances, output formats.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import json

# Default paths
DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_RESULTS_FILE = DEFAULT_RESULTS_DIR / "derivations.json"


@dataclass
class SRTConfig:
    """Configuration settings for SRT calculations."""
    
    # Precision settings
    decimal_places: int = 512
    
    # Validation tolerances
    exact_threshold: float = 0.01      # <0.01% = EXACT
    pass_threshold: float = 0.5        # <0.5% = PASS
    
    # Mining settings
    mining_tolerance: float = 0.05     # 5% tolerance for mining
    max_base_integer: int = 1000       # Maximum N to try in E*×N formulas
    
    # Output settings
    output_format: Literal["text", "json", "latex"] = "text"
    verbose: bool = False
    save_results: bool = True
    results_file: Path = field(default_factory=lambda: DEFAULT_RESULTS_FILE)
    
    # Display settings
    show_tree_values: bool = True
    show_corrections: bool = True
    precision_digits: int = 4          # Digits for display
    
    def __post_init__(self):
        """Ensure results directory exists."""
        if self.save_results:
            self.results_file.parent.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "decimal_places": self.decimal_places,
            "exact_threshold": self.exact_threshold,
            "pass_threshold": self.pass_threshold,
            "mining_tolerance": self.mining_tolerance,
            "max_base_integer": self.max_base_integer,
            "output_format": self.output_format,
            "verbose": self.verbose,
            "save_results": self.save_results,
            "results_file": str(self.results_file),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SRTConfig":
        """Load from dictionary."""
        if "results_file" in data:
            data["results_file"] = Path(data["results_file"])
        return cls(**data)
    
    def save(self, path: Path | None = None):
        """Save configuration to JSON file."""
        path = path or (self.results_file.parent / "config.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "SRTConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# Global configuration instance
config = SRTConfig()


def get_config() -> SRTConfig:
    """Get the global configuration."""
    return config


def set_config(new_config: SRTConfig):
    """Set the global configuration."""
    global config
    config = new_config


# Convenience functions
def set_precision(decimal_places: int):
    """Set calculation precision."""
    config.decimal_places = decimal_places


def set_tolerance(pass_threshold: float):
    """Set validation pass threshold."""
    config.pass_threshold = pass_threshold


def enable_verbose():
    """Enable verbose output."""
    config.verbose = True


def disable_save():
    """Disable saving results to file."""
    config.save_results = False
