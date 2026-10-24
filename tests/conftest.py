"""Keep temporary test files in the writable project cache on Windows."""
from pathlib import Path
from uuid import uuid4


def pytest_configure(config):
    if config.option.basetemp is None:
        # A new, dedicated directory avoids touching existing temporary files.
        parent = Path(config.rootpath) / '.pytest_cache' / 'tmp'
        parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(parent / uuid4().hex)
