import logging
from src.wavelet_transformer import WaveletTransformer
from src.universal_graph_converter import UniversalGraphConverter
from src.core_pattern_learner import CorePatternLearner
from src.fibonacci_compressor import FibonacciCompressor
from src.rl_expander import RLExpander
from src.llm_integration import LLMIntegration
from src.feedback_db import FeedbackDB

class PipelineManager:
    def __init__(self, config):
        self.config = config
        self.wavelet = WaveletTransformer(**config.get("wavelet", {}))
        self.graph_converter = UniversalGraphConverter()
        self.learner = CorePatternLearner(**config.get("learner", {}))
        self.compressor = FibonacciCompressor()
        self.expander = RLExpander(**config.get("expander", {}))
        self.llm = LLMIntegration(config.get("llm", {}))
        self.feedback_db = FeedbackDB(config.get("feedback_db", {}))

    def run(self, data, env=None):
        logging.info("Starting pipeline...")

        # Step 1: Wavelet Decomposition
        decomposed = self.wavelet.transform(data)

        # Step 2: Convert to Graph
        graph = self.graph_converter.to_graph(decomposed)

        # Step 3: Learn Patterns
        learned_patterns = self.learner.train(graph)

        # Step 4: Compress with Fibonacci
        compressed = self.compressor.compress(learned_patterns)

        # Step 5: Expand with Reinforcement Learning
        expanded = self.expander.expand(compressed, env)

        # Step 6: Explain with LLM
        explanation = self.llm.explain(expanded)

        # Step 7: Store feedback
        self.feedback_db.store_feedback({"input": data, "output": explanation})

        return explanation
