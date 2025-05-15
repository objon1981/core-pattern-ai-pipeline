import unittest
from src.llm_integration import LLMIntegration

class TestLLMIntegration(unittest.TestCase):
    def setUp(self):
        # Use a small model for testing (e.g., distilgpt2)
        self.llm = LLMIntegration(model_name="distilgpt2", device="cpu")

    def test_generate_text(self):
        prompt = "The core pattern of AI is"
        generated = self.llm.generate(prompt, max_length=20)
        self.assertIsInstance(generated, str)
        self.assertTrue(len(generated) > len(prompt))

    # Note: Fine-tuning tests can be resource-heavy, so a full fine-tune test is often skipped in unit tests
    # but you can mock or do a very short fine-tune in integration tests if needed.

if __name__ == "__main__":
    unittest.main()
