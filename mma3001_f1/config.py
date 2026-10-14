"""Offline configuration for a small pilot; this module never downloads data.

The batch exporter uses these pilot defaults and enforces the session budget
after schedule resolution, before starting session extraction.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class ProjectConfig:
    """Reproducible pilot scope. Limit counts race sessions, not byte volume."""

    seasons: tuple[int, ...] = (2024,)
    circuits: tuple[str, ...] = ("Bahrain", "Monaco", "Japan")
    drivers: tuple[str, ...] = ("VER", "HAM")
    seed: int = 42
    max_sessions: int = 3
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    cache_dir: str = "cache"
    results_dir: str = "results"

    def __post_init__(self) -> None:
        if not self.seasons or any(type(y) is not int or y < 2018 for y in self.seasons):
            raise ValueError("seasons must contain FastF1-era integer years >= 2018")
        if len(set(self.seasons)) != len(self.seasons):
            raise ValueError("seasons must be unique")
        if not self.circuits or any(not isinstance(c, str) or not c.strip() for c in self.circuits):
            raise ValueError("circuits must be explicit nonempty event filters")
        if len(set(self.circuits)) != len(self.circuits):
            raise ValueError("circuits must be unique")
        if any(not isinstance(d, str) or len(d) != 3 or not d.isalpha() or not d.isupper() for d in self.drivers):
            raise ValueError("drivers must be uppercase three-letter codes")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if type(self.max_sessions) is not int or self.max_sessions < 1:
            raise ValueError("max_sessions must be a positive integer")
        self.validate_session_count(len(self.seasons) * len(self.circuits))
        for directory in (self.raw_dir, self.processed_dir, self.cache_dir, self.results_dir):
            # Both separators are checked so a Windows config stays safe on Colab.
            parts = directory.replace("\\", "/").split("/")
            if not directory or parts[0] == "" or ".." in parts or ":" in directory or directory == ".":
                raise ValueError("directories must be relative paths inside the project")
        if len({self.raw_dir, self.processed_dir, self.cache_dir, self.results_dir}) != 4:
            raise ValueError("data, cache and results directories must be distinct")

    def validate_session_count(self, count: int) -> None:
        """Reject an oversized resolved schedule; raising the limit needs review."""
        if type(count) is not int or not 1 <= count <= self.max_sessions:
            raise ValueError(f"session count must be between 1 and {self.max_sessions}")

    def paths(self, root: Path) -> dict[str, Path]:
        """Resolve paths without creating directories; reject escaping symlinks."""
        root = root.resolve()
        resolved = {name: (root / getattr(self, name)).resolve() for name in
                    ("raw_dir", "processed_dir", "cache_dir", "results_dir")}
        if any(not p.is_relative_to(root) or p == root for p in resolved.values()):
            raise ValueError("directory resolves outside the project")
        return resolved


if __name__ == "__main__":
    print(json.dumps(asdict(ProjectConfig()), indent=2))
