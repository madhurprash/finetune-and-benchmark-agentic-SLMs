"""
Supervised Fine-Tuning (SFT) Script for Devstral-Small-2-24B
Using TRL's SFTTrainer for efficient fine-tuning on OpenThoughts dataset.

Reference: https://huggingface.co/docs/trl/en/sft_trainer
"""

import os
import sys
import yaml
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)

def load_config(config_path="train.yaml"):
    """Load configuration from YAML file."""
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model_and_tokenizer(config):
    """Load the model and tokenizer."""
    print("Loading model and tokenizer...")
    model_config = config['model']
    quant_config = config.get('quantization', {})

    # For DDP training, we don't use device_map - let the trainer handle device placement
    # The model will be loaded to CPU and then moved to the correct GPU by the trainer
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank != -1:
        print(f"Process local_rank={local_rank}, DDP mode - model will be placed on correct device by trainer")
    else:
        print("Single GPU/CPU mode")

    # Load tokenizer
    tokenizer = MistralCommonBackend.from_pretrained(
        model_config['name']
    )
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup quantization if enabled
    quantization_config = None
    if quant_config.get('enabled', False):
        print("Setting up 4-bit quantization for memory efficiency...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=quant_config.get('load_in_4bit', True),
            bnb_4bit_compute_dtype=getattr(torch, quant_config.get('bnb_4bit_compute_dtype', 'bfloat16')),
            bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', True),
        )

    # Load model - with quantization if enabled, otherwise no device_map for DDP
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_config['name'],
        trust_remote_code=model_config['trust_remote_code'],
        torch_dtype=getattr(torch, model_config['torch_dtype']),
        attn_implementation=model_config.get('attn_implementation', 'eager'),
        quantization_config=quantization_config,
        device_map={"": local_rank} if local_rank != -1 and quantization_config else None,
    )
    print("Model and tokenizer loaded successfully")
    return model, tokenizer

def setup_lora(model, config):
    """Setup LoRA for parameter-efficient fine-tuning."""
    lora_config = config['lora']

    if not lora_config['enabled']:
        print("LoRA is disabled, using full fine-tuning")
        return model

    print("Setting up LoRA configuration...")

    # Prepare model for training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    peft_config = LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        bias=lora_config['bias'],
        task_type=lora_config['task_type'],
        target_modules=lora_config['target_modules'],
    )

    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("LoRA setup completed")
    return model


def load_and_format_dataset(config, tokenizer):
    """Load and format the dataset for training."""
    print("Loading dataset...")
    dataset_config = config['dataset']

    # Load dataset
    dataset = load_dataset(
        dataset_config['name'],
        dataset_config.get('config'),
        split=dataset_config['split']
    )

    # Optionally limit dataset size for testing
    if dataset_config.get('num_train_samples') is not None:
        num_samples = dataset_config['num_train_samples']
        print(f"Limiting training to {num_samples} samples")
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Dataset size: {len(dataset)} examples")

    # Format dataset using chat template
    def formatting_func(example):
        """Format conversations using the tokenizer's chat template."""
        conversation = example["conversations"]

        # Filter out messages with empty content
        filtered_conversation = [
            msg for msg in conversation
            if msg.get("content", "").strip() != ""
        ]

        # Skip if no valid messages remain
        if not filtered_conversation:
            return {"text": ""}

        text = tokenizer.apply_chat_template(
            filtered_conversation,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True
        )
        return {"text": text}

    dataset = dataset.map(formatting_func, remove_columns=dataset.column_names)

    # Filter out examples with empty text
    original_size = len(dataset)
    dataset = dataset.filter(lambda x: x["text"].strip() != "")
    filtered_size = len(dataset)

    if filtered_size < original_size:
        print(f"Filtered out {original_size - filtered_size} examples with empty content")

    print(f"Final dataset size: {filtered_size} examples")

    # Log a sample
    print("\nSample formatted text:")
    print("-" * 80)
    print(dataset[0]["text"][:500] + "...")
    print("-" * 80)

    return dataset


def create_trainer(model, tokenizer, dataset, config):
    """Create the SFT trainer."""
    print("Creating SFT trainer...")
    training_config = config['training']
    sft_config = config['sft']

    # Setup training arguments
    training_args = SFTConfig(
        output_dir=training_config['output_dir'],
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        gradient_checkpointing=training_config['gradient_checkpointing'],
        max_steps=training_config.get('max_steps', 100),

        # Optimizer
        optim=training_config['optim'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        warmup_ratio=training_config['warmup_ratio'],
        lr_scheduler_type=training_config['lr_scheduler_type'],

        # Logging and saving
        logging_steps=training_config['logging_steps'],
        save_strategy=training_config['save_strategy'],
        save_steps=training_config['save_steps'],
        save_total_limit=training_config['save_total_limit'],

        # Mixed precision
        bf16=training_config['bf16'],
        fp16=training_config['fp16'],

        # Other settings
        max_grad_norm=training_config['max_grad_norm'],
        seed=training_config['seed'],
        dataloader_num_workers=training_config['dataloader_num_workers'],
        remove_unused_columns=training_config['remove_unused_columns'],

        # Reporting
        report_to=training_config['report_to'],
        logging_dir=training_config.get('logging_dir', './logs'),

        # SFT specific
        packing=sft_config['packing'],
        dataset_text_field=sft_config['dataset_text_field'],
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("Trainer created successfully")
    return trainer


def main():
    """Main training function."""
    # Load configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else "train.yaml"
    config = load_config(config_path)

    # Print training configuration
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Max sequence length: {config['model']['max_seq_length']}")
    print(f"LoRA enabled: {config['lora']['enabled']}")
    print(f"Output directory: {config['training']['output_dir']}")
    print(f"Batch size: {config['training']['per_device_train_batch_size']}")
    print(f"Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Epochs: {config['training']['num_train_epochs']}")
    print(f"Max steps: {config['training'].get('max_steps', 'Not set')}")
    print("=" * 80 + "\n")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Setup LoRA if enabled
    if config['lora']['enabled']:
        model = setup_lora(model, config)

    # Load and format dataset
    dataset = load_and_format_dataset(config, tokenizer)

    # Create trainer
    trainer = create_trainer(model, tokenizer, dataset, config)

    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")

    trainer.train()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80 + "\n")

    # Save model
    print(f"Saving model to {config['training']['output_dir']}")
    trainer.save_model()
    tokenizer.save_pretrained(config['training']['output_dir'])

    print("\n" + "=" * 80)
    print(f"Model saved successfully to: {config['training']['output_dir']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
