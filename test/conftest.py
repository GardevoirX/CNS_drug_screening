import sys
import pytest
from os.path import dirname, abspath, join

ROOT = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT)
pytest.DATA = join(ROOT, "test", "data")
pytest.EXAMPLE_DATA = join(pytest.DATA, "small_dataset.csv")
