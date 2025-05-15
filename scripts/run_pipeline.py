import argparse
import json
from src.pipeline_manager import PipelineManager
from src.utils.dashboard_utils import load_environment

def main():
    parser = argparse.ArgumentParser(description="Run the Core Pattern AI Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to pipeline config JSON")
    parser.add_argument("--input", type=str, required=True, help="Input data (comma-separated)")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    # Parse input
    input_data = [float(x.strip()) for x in args.input.split(",")]

    # Load environment
    env = load_environment(config)

    # Run pipeline
    pipeline = PipelineManager(config)
    explanation = pipeline.run(input_data, env)

    print("\n📘 Explanation:")
    print(explanation)

if __name__ == "__main__":
    main()
