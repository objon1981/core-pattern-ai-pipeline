import argparse
import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

from .utils import get_tokenizer, save_model
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Example usage
base_model = config['base_model']
epochs = config['training']['epochs']
lora_r = config['lora']['r']


def main(args):
    # Load tokenizer and base model
    tokenizer = get_tokenizer(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    # Prepare LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    model = get_peft_model(model, lora_config)

    # Load dataset
    dataset = load_dataset(args.dataset_name, split="train")

    # Tokenize function
    def tokenize_fn(example):
        return tokenizer(example['text'], truncation=True, max_length=args.max_length)

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_steps=1000,
        save_total_limit=2,
        evaluation_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()
    save_model(model, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA fine-tuned model")

    parser.add_argument("--base_model", type=str, default="gpt2", help="Base pretrained model")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Dataset name to use")
    parser.add_argument("--output_dir", type=str, default="./lora_model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")

    args = parser.parse_args()
    main(args)
