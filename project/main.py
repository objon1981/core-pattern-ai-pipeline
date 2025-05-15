import argparse
import json
import logging
from src.pipeline_manager import PipelineManager

# Setup logging
logging.basicConfig(level=logging.INFO)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_input_data(data_path):
    with open(data_path, 'r') as f:
        return json.load(f)  # assuming input is in JSON

def main(config_path, data_path):
    config = load_config(config_path)
    data = load_input_data(data_path)

    pipeline = PipelineManager(config)
    result = pipeline.run(data)

    print("Pipeline result:", result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SOGUM AI Core Pattern Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration JSON file")
    parser.add_argument("--data", type=str, required=True, help="Path to the input data file")

    args = parser.parse_args()
    main(args.config, args.data)
