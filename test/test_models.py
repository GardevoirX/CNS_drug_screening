from src.models import AVAILABLE_MODELS

TEST_DATA = [
    [0, 1],
    [2, 1],
]

TEST_LABELS = [
    0,
    1,
]

def test_models():
    for model in AVAILABLE_MODELS:
        model = model()
        model.fit(TEST_DATA, TEST_LABELS)
        result = model.predict(TEST_DATA)
        assert len(result) == 2