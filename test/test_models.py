import pytest

from src.models import AVAILABLE_MODELS

TEST_DATA = [
    [0, 1],
    [2, 1],
    [0, 1],
    [0, 1],
    [2, 1]
]

TEST_LABELS = [
    0,
    1,
    0,
    1,
    1
]


@pytest.mark.parametrize("test_model", AVAILABLE_MODELS)
def test_models(test_model):
    model = test_model()
    model.fit(TEST_DATA, TEST_LABELS)
    result = model.predict(TEST_DATA)
    assert len(result) == 5
