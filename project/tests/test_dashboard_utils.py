# tests/test_dashboard_utils.py

import os
import json
import pytest
from src.utils.dashboard_utils import (
    save_feedback,
    save_explanation,
    save_query,
    save_attention_weights,
    save_training_metrics
)

test_data = {
    "feedback": {"timestamp": "2025-05-15", "content": "Test feedback."},
    "explanation": {"timestamp": "2025-05-15", "content": "Test explanation."},
    "query": {"timestamp": "2025-05-15", "content": "Test query."},
    "attention": {"layer1": [0.1, 0.2]},
    "metrics": {"epoch": 1, "loss": 0.2, "accuracy": 0.95}
}

@pytest.mark.parametrize("data, filename, func", [
    (test_data["feedback"], "logs/dashboard_feedback.jsonl", save_feedback),
    (test_data["explanation"], "logs/dashboard_explanations.jsonl", save_explanation),
    (test_data["query"], "logs/dashboard_queries.jsonl", save_query),
    (test_data["attention"], "logs/dashboard_attention_weights.json", save_attention_weights),
    (test_data["metrics"], "logs/dashboard_training_metrics.json", save_training_metrics),
])
def test_dashboard_utils(data, filename, func):
    func(data)
    assert os.path.exists(filename)

    if filename.endswith(".jsonl"):
        with open(filename, "r") as f:
            lines = f.readlines()
            assert json.loads(lines[-1]) == data
    else:
        with open(filename, "r") as f:
            saved_data = json.load(f)
            assert saved_data == data

    # Cleanup
    os.remove(filename)
