"""
Supervised Fine-Tuning (SFT) Script for NVIDIA Nemotron-3-Nano-30B-A3B-BF16

This script fine-tunes the Nemotron model on the OpenThoughts-Agent-v1-SFT dataset
using QLoRA for efficient training on a single GPU.

Author: Claude
License: Apache 2.0
"""

import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    HfArgumentParser,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""

    model_name_or_path: str = field(
        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading the model"}
    )
    use_flash_attention_2: bool = field(
        default=False,
        metadata={"help": "Whether to use Flash Attention 2"}
    )


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training."""

    dataset_name: str = field(
        default="open-thoughts/OpenThoughts-Agent-v1-SFT",
        metadata={"help": "The name of the dataset to use"}
    )
    dataset_config: str = field(
        default="default",
        metadata={"help": "The configuration name of the dataset to use"}
    )
    max_seq_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length for training"}
    )
    num_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Number of training samples to use (for debugging)"}
    )


@dataclass
class LoRAArguments:
    """Arguments for LoRA configuration."""

    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout probability"}
    )
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA"}
    )
    use_qlora: bool = field(
        default=True,
        metadata={"help": "Whether to use QLoRA (4-bit quantization with LoRA)"}
    )


def format_conversations_for_training(example: Dict) -> Dict:
    """
    Format the OpenThoughts dataset conversations into a single text field for training.

    The dataset has conversations in the format:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

    We'll format this as a chat template compatible with the model.
    """
    conversations = example["conversations"]

    # Format the conversation using a simple template
    # You can customize this based on the model's chat template
    formatted_text = ""

    for turn in conversations:
        role = turn["role"]
        content = turn["content"]

        if role == "user":
            formatted_text += f"<|user|>\n{content}\n"
        elif role == "assistant":
            formatted_text += f"<|assistant|>\n{content}\n"
        elif role == "system":
            formatted_text += f"<|system|>\n{content}\n"

    # Add end of sequence token
    formatted_text += "<|end|>"

    return {"text": formatted_text}


def create_bnb_config(use_qlora: bool = True) -> Optional[BitsAndBytesConfig]:
    """Create BitsAndBytes configuration for 4-bit quantization."""
    if not use_qlora:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def create_lora_config(lora_args: LoRAArguments) -> LoraConfig:
    """Create LoRA configuration."""
    target_modules = lora_args.lora_target_modules.split(",") if lora_args.lora_target_modules else None

    return LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def main():
    """Main training function."""

    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, LoRAArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load arguments from JSON file
        model_args, data_args, lora_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    # Log training parameters
    logger.info("=" * 60)
    logger.info("Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Dataset: {data_args.dataset_name}")
    logger.info(f"Max sequence length: {data_args.max_seq_length}")
    logger.info(f"Use LoRA: {lora_args.use_lora}")
    logger.info(f"Use QLoRA: {lora_args.use_qlora}")
    logger.info(f"Output directory: {training_args.output_dir}")
    logger.info(f"Batch size per device: {training_args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info(f"Number of epochs: {training_args.num_train_epochs}")
    logger.info("=" * 60)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    # Load dataset
    logger.info(f"Loading dataset: {data_args.dataset_name}")
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config,
        split="train"
    )

    # Optionally limit dataset size for debugging
    if data_args.num_train_samples is not None:
        logger.info(f"Limiting training to {data_args.num_train_samples} samples")
        dataset = dataset.select(range(min(data_args.num_train_samples, len(dataset))))

    logger.info(f"Dataset size: {len(dataset)} examples")

    # Format the dataset
    logger.info("Formatting dataset...")
    dataset = dataset.map(
        format_conversations_for_training,
        remove_columns=dataset.column_names,
        desc="Formatting conversations"
    )

    # Log a sample
    logger.info("Sample formatted text:")
    logger.info("-" * 60)
    logger.info(dataset[0]["text"][:500] + "...")
    logger.info("-" * 60)

    # Create BitsAndBytes config for quantization
    bnb_config = create_bnb_config(use_qlora=lora_args.use_qlora)

    # Load model
    logger.info("Loading model...")
    model_kwargs = {
        "pretrained_model_name_or_path": model_args.model_name_or_path,
        "trust_remote_code": model_args.trust_remote_code,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }

    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    if model_args.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    # Prepare model for k-bit training if using QLoRA
    if lora_args.use_qlora:
        logger.info("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    if lora_args.use_lora:
        logger.info("Applying LoRA...")
        lora_config = create_lora_config(lora_args)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Create trainer
    logger.info("Creating trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=data_args.max_seq_length,
        packing=False,  # Disable packing for simplicity
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save the final model
    logger.info(f"Saving final model to {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    # Save training metrics
    metrics_file = os.path.join(training_args.output_dir, "training_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {training_args.output_dir}")
    logger.info("=" * 60)

    # Optionally merge and save the full model (if using LoRA)
    if lora_args.use_lora:
        logger.info("\nTo merge LoRA weights with base model, run:")
        logger.info("python SFT/merge_lora.py --model_path <output_dir> --output_path <merged_output_dir>")


if __name__ == "__main__":
    main()
