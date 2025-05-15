import os
import torch
from transformers import AutoTokenizer

def get_tokenizer(model_name_or_path):
    """Load and return tokenizer for the base model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return tokenizer

def save_model(model, save_path):
    """Save the LoRA fine-tuned model to the given path."""
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

def load_model(model_class, model_path):
    """Load the fine-tuned model from path."""
    model = model_class.from_pretrained(model_path)
    print(f"Model loaded from {model_path}")
    return model
