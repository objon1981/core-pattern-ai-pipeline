import unittest
import numpy as np
from src.rl_expander import RLExpander

# Mock Environment for testing
class MockEnv:
    def __init__(self):
        self.state = 0
        self.steps = 0
        self.max_steps = 5
    
    def reset(self):
        self.state = 0
        self.steps = 0
        return self.state
    
    def step(self, action):
        self.state += action
        self.steps += 1
        reward = 1 if action == 1 else 0  # Reward 1 for action 1, else 0
        done = self.steps >= self.max_steps
        info = {}
        return self.state, reward, done, info

class TestRLExpander(unittest.TestCase):
    def setUp(self):
        self.env = MockEnv()
        self.expander = RLExpander(input_dim=1, hidden_dim=8, output_dim=2)  # 2 actions: 0 or 1

    def test_select_action(self):
        state = [0.0]
        action = self.expander.select_action(state)
        self.assertIn(action, [0, 1], "Action should be either 0 or 1")

    def test_training_episode(self):
        self.expander.train(self.env, episodes=10)  # Train briefly to see no errors

    def test_expand_method(self):
        compressed_input = [0.0]
        output = self.expander.expand(compressed_input, self.env)
        self.assertEqual(len(output), self.env.max_steps, "Output length should equal max steps")

if __name__ == "__main__":
    unittest.main()
