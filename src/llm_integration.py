# LLM explanation & fine-tuning functions

# src/llm_integration.py
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
import logging

class LLMIntegration:
    """
    Handles loading, fine-tuning, and inference with a Large Language Model (LLM).
    """

    def __init__(self, model_name: str = "gpt2", device: str = None):
        """
        Initialize the LLMIntegration with a pre-trained model.

        Args:
            model_name (str): HuggingFace model identifier.
            device (str): Device to load the model on (e.g., 'cuda' or 'cpu').
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.logger = logging.getLogger('LLMIntegration')

    def fine_tune(self, train_dataset, output_dir: str, epochs: int = 3, batch_size: int = 8):
        """
        Fine-tune the model on a custom dataset.

        Args:
            train_dataset (Dataset): A PyTorch Dataset or compatible dataset.
            output_dir (str): Directory to save the fine-tuned model.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )

        self.logger.info(f"Starting fine-tuning for {epochs} epochs...")
        trainer.train()
        trainer.save_model(output_dir)
        self.logger.info(f"Model fine-tuned and saved at {output_dir}.")

    def generate(self, prompt: str, max_length: int = 100):
        """
        Generate text from the model given a prompt.

        Args:
            prompt (str): Input prompt text.
            max_length (int): Max tokens to generate.

        Returns:
            str: Generated text.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_length=max_length)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
