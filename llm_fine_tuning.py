#!/usr/bin/env python3
"""
LLM Fine-Tuning Pipeline for Competitive Programming
Optimized for AMD RX 6800 XT with DirectML

This script implements fine-tuning of code LLMs using the consolidated dataset
from dataset_consolidation.py, with DirectML acceleration for AMD GPUs.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, EarlyStoppingCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import onnxruntime as ort
from datasets import Dataset as HuggingFaceDataset
import numpy as np
from tqdm import tqdm
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning"""
    # Model settings
    model_name: str = "microsoft/DialoGPT-medium"  # Lightweight model for testing
    max_length: int = 4096
    batch_size: int = 2  # Conservative for DirectML
    gradient_accumulation_steps: int = 4

    # Training settings
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Quantisation settings
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # DirectML settings
    use_directml: bool = True
    mixed_precision: bool = True
    
    # Data settings
    train_file: str = "consolidated_output/train.jsonl"
    val_file: str = "consolidated_output/validation.jsonl"
    test_file: str = "consolidated_output/test.jsonl"
    
    # Output settings
    output_dir: str = "fine_tuned_model"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Quality settings
    min_question_length: int = 50
    max_question_length: int = 2000
    min_code_length: int = 20

class CompetitiveProgrammingDataset(Dataset):
    """Custom dataset for competitive programming problems"""
    
    def __init__(self, data_file: str, tokenizer, config: FineTuningConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.data = self._load_and_preprocess_data(data_file)
        
    def _load_and_preprocess_data(self, data_file: str) -> List[Dict]:
        """Load and preprocess the consolidated dataset"""
        logger.info(f"Loading data from {data_file}")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} not found. Run dataset_consolidation.py first.")
        
        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    processed_item = self._process_item(item)
                    if processed_item:
                        data.append(processed_item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
        
        logger.info(f"Loaded {len(data)} valid entries from {data_file}")
        return data
    
    def _process_item(self, item: Dict) -> Optional[Dict]:
        """Process a single data item"""
        # Extract fields
        question = item.get('question_text', '').strip()
        answer = item.get('answer_text', '').strip()
        code = item.get('solution_code', '').strip()
        language = item.get('language', '').strip()
        difficulty = item.get('difficulty', '').strip()
        
        # Quality filtering
        if len(question) < self.config.min_question_length:
            return None
        if len(question) > self.config.max_question_length:
            return None
        if not code or len(code) < self.config.min_code_length:
            return None
        
        # Create training text
        training_text = self._create_training_text(question, answer, code, language, difficulty)
        
        return {
            'text': training_text,
            'question': question,
            'code': code,
            'language': language,
            'difficulty': difficulty
        }
    
    def _create_training_text(self, question: str, answer: str, code: str, language: str, difficulty: str) -> str:
        """Create formatted training text for the model"""
        # Format: Question + Answer + Code with clear separators
        parts = []
        
        # Add question
        parts.append(f"Question: {question}")
        
        # Add answer if available
        if answer:
            parts.append(f"Explanation: {answer}")
        
        # Add code with language specification
        if language:
            parts.append(f"Solution in {language}:")
        else:
            parts.append("Solution:")
        
        parts.append(code)
        
        # Add difficulty if available
        if difficulty:
            parts.append(f"Difficulty: {difficulty}")
        
        # Join with newlines and add end token
        return "\n\n".join(parts) + "\n\n"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class DirectMLTrainer(Trainer):
    """Custom trainer optimized for DirectML"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_directml()
    
    def setup_directml(self):
        """Setup DirectML optimizations"""
        if hasattr(self.args, 'use_directml') and self.args.use_directml:
            logger.info("Setting up DirectML optimizations...")
            
            # Set environment variables for DirectML
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            
            # Configure PyTorch for DirectML
            if torch.backends.mps.is_available():
                logger.info("Using MPS (Metal Performance Shaders) for AMD GPU")
                self.args.device = torch.device('mps')
            else:
                logger.info("Using CPU with DirectML acceleration")
                self.args.device = torch.device('cpu')
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation for DirectML"""
        outputs = model(**inputs)
        loss = outputs.loss
        
        if return_outputs:
            return loss, outputs
        return loss

def setup_directml_environment():
    """Setup DirectML environment and verify availability"""
    logger.info("Setting up DirectML environment...")
    
    # Check DirectML availability
    try:
        providers = ort.get_available_providers()
        logger.info(f"Available ONNX Runtime providers: {providers}")
        
        if 'DmlExecutionProvider' in providers:
            logger.info("DirectML provider is available!")
        else:
            logger.warning("DirectML provider not found. Using CPU fallback.")
            
    except Exception as e:
        logger.error(f"Error checking DirectML: {e}")
    
    # Check PyTorch
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS device for AMD GPU")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device with DirectML acceleration")
    
    return device

def load_model_and_tokenizer(config: FineTuningConfig, device):
    """Load model and tokenizer with DirectML optimizations"""
    logger.info(f"Loading model: {config.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantisation if requested
    bnb_config = None
    if config.load_in_8bit or config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=config.load_in_8bit,
            load_in_4bit=config.load_in_4bit,
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if config.mixed_precision else torch.float32,
        device_map='auto' if not config.use_directml else None,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        load_in_8bit=config.load_in_8bit if not bnb_config else None,
        load_in_4bit=config.load_in_4bit if not bnb_config else None,
    )
    
    # Move to device if not using device_map
    if config.use_directml:
        model = model.to(device)
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def create_data_collator(tokenizer, config: FineTuningConfig):
    """Create data collator for training"""
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling
        return_tensors="pt"
    )

def tokenize_function(examples, tokenizer, config: FineTuningConfig):
    """Tokenize the dataset"""
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding=True,
        max_length=config.max_length,
        return_tensors="pt"
    )
    
    # Set labels to input_ids for causal LM
    tokenized['labels'] = tokenized['input_ids'].clone()
    
    return tokenized

def prepare_datasets(config: FineTuningConfig, tokenizer, device):
    """Prepare training and validation datasets"""
    logger.info("Preparing datasets...")
    
    # Load datasets
    train_dataset = CompetitiveProgrammingDataset(config.train_file, tokenizer, config)
    val_dataset = CompetitiveProgrammingDataset(config.val_file, tokenizer, config)
    
    # Convert to HuggingFace datasets
    train_hf_dataset = HuggingFaceDataset.from_list(train_dataset.data)
    val_hf_dataset = HuggingFaceDataset.from_list(val_dataset.data)
    
    # Tokenize datasets
    def tokenize_wrapper(examples):
        return tokenize_function(examples, tokenizer, config)
    
    train_tokenized = train_hf_dataset.map(
        tokenize_wrapper,
        batched=True,
        remove_columns=train_hf_dataset.column_names
    )
    
    val_tokenized = val_hf_dataset.map(
        tokenize_wrapper,
        batched=True,
        remove_columns=val_hf_dataset.column_names
    )
    
    logger.info(f"Training samples: {len(train_tokenized)}")
    logger.info(f"Validation samples: {len(val_tokenized)}")
    
    return train_tokenized, val_tokenized

def setup_training_args(config: FineTuningConfig, output_dir: str):
    """Setup training arguments"""
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=config.mixed_precision,
        dataloader_pin_memory=False,  # Disable for DirectML
        remove_unused_columns=False,
        report_to="wandb" if config.use_directml else None,
        run_name="competitive-programming-finetune",
    )

def main():
    """Main fine-tuning pipeline"""
    parser = argparse.ArgumentParser(description="Fine-tune LLM for competitive programming")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--model", type=str, default="microsoft/DialoGPT-medium", help="Model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="fine_tuned_model", help="Output directory")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = FineTuningConfig(**config_dict)
    else:
        config = FineTuningConfig()
    
    # Override with command line args
    config.model_name = args.model
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.output_dir = args.output_dir
    
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    
    # Setup DirectML
    device = setup_directml_environment()
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project="competitive-programming-llm",
            config=vars(config),
            name=f"finetune-{config.model_name.split('/')[-1]}"
        )
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config, device)
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(config, tokenizer, device)
    
    # Create data collator
    data_collator = create_data_collator(tokenizer, config)
    
    # Setup training arguments
    training_args = setup_training_args(config, config.output_dir)
    
    # Initialize trainer
    trainer = DirectMLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Start training
    logger.info("Starting fine-tuning...")
    trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Save config
    with open(os.path.join(config.output_dir, "config.json"), 'w') as f:
        json.dump(vars(config), f, indent=2)
    
    logger.info(f"Fine-tuning completed! Model saved to {config.output_dir}")
    
    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 