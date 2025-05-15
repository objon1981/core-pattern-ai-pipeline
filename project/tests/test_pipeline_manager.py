import unittest
from src.pipeline_manager import PipelineManager

class DummyEnv:
    def reset(self):
        return [0.0]

    def step(self, action):
        return [0.0], 1.0, True, {}

class TestPipelineManager(unittest.TestCase):
    def setUp(self):
        self.config = {
            "wavelet": {"wavelet_name": "db1", "level": 2},
            "learner": {"in_dim": 10, "hidden_dim": 16, "out_dim": 2},
            "expander": {"input_dim": 5, "hidden_dim": 8, "output_dim": 3},
            "llm": {"model": "mock-llm"},
            "feedback_db": {"db_path": ":memory:"}
        }
        self.pipeline = PipelineManager(self.config)
        self.env = DummyEnv()

    def test_pipeline_run(self):
        data = [1.0, 2.0, 3.0, 4.0]
        result = self.pipeline.run(data, env=self.env)
        self.assertIsInstance(result, str)

if __name__ == '__main__':
    unittest.main()
