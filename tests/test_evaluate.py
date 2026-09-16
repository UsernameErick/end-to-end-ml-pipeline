from src.evaluate import evaluate_model

def test_model_passes_quality_gate():
    accuracy = 0.95
    passed = evaluate_model(accuracy)

    assert passed is True

def test_model_fails_quality_gate():
    accuracy = 0.949
    passed = evaluate_model(accuracy)

    assert passed is False