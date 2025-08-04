#!/usr/bin/env python3
"""
Inference Engine for Competitive Programming Assistant
Optimized for AMD RX 6800 XT with DirectML

This script provides fast inference for the fine-tuned model, designed to be
integrated into the invisible overlay system.
"""

import os
import json
import logging
import time
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
import threading
from peft import PeftModel
import onnxruntime as ort
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """Configuration for inference"""
    model_path: str = "fine_tuned_model"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_return_sequences: int = 1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    # DirectML settings
    use_directml: bool = True
    mixed_precision: bool = True

    # Quantisation settings
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    # Performance settings
    max_new_tokens: int = 512
    use_cache: bool = True
    lazy_load: bool = True

class DirectMLInferenceEngine:
    """Inference engine optimized for DirectML on AMD GPUs"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = self._setup_device()
        self.tokenizer, self.model = self._load_model()
        self._warmup()
    
    def _setup_device(self) -> torch.device:
        """Setup device for DirectML inference"""
        logger.info("Setting up DirectML inference environment...")
        
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
        
        # Set device
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using MPS device for AMD GPU inference")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU device with DirectML acceleration")
        
        return device
    
    def _load_model(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """Load the fine-tuned model and tokenizer"""
        logger.info(f"Loading model from {self.config.model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure quantisation if requested
        bnb_config = None
        if self.config.load_in_8bit or self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=self.config.load_in_8bit,
                load_in_4bit=self.config.load_in_4bit,
            )

        # Determine device map for lazy loading
        device_map = 'auto' if self.config.lazy_load else None
        if self.config.use_directml and device_map is None:
            device_map = None  # load on CPU then move

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16 if self.config.mixed_precision else torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=self.config.lazy_load,
            use_cache=self.config.use_cache,
            quantization_config=bnb_config,
            load_in_8bit=self.config.load_in_8bit if not bnb_config else None,
            load_in_4bit=self.config.load_in_4bit if not bnb_config else None,
        )

        # Move to device if needed
        if self.config.use_directml:
            model = model.to(self.device)
        
        # Load LoRA weights if they exist
        lora_path = os.path.join(self.config.model_path, "adapter_config.json")
        if os.path.exists(lora_path):
            logger.info("Loading LoRA adapter weights...")
            model = PeftModel.from_pretrained(model, self.config.model_path)
        
        # Set to evaluation mode
        model.eval()
        
        logger.info("Model loaded successfully!")
        return tokenizer, model
    
    def _warmup(self):
        """Warm up the model with a dummy inference"""
        logger.info("Warming up model...")
        
        dummy_text = "Question: Write a function to find the sum of two numbers.\n\nSolution:"
        try:
            with torch.no_grad():
                self.generate_response(dummy_text, max_new_tokens=10)
            logger.info("Model warmup completed!")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
    
    def generate_response(self, question: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate a response for a competitive programming question"""
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        
        # Format the input
        formatted_input = self._format_input(question)
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length - max_new_tokens,
            padding=True
        )
        
        # Move to device
        if self.config.use_directml:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                do_sample=self.config.do_sample,
                num_return_sequences=self.config.num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=self.config.use_cache
            )

        generation_time = time.time() - start_time
        total_tokens = outputs.shape[-1]
        generated_tokens = total_tokens - inputs['input_ids'].shape[-1]
        tps = generated_tokens / generation_time if generation_time > 0 else float('inf')

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part
        response = generated_text[len(formatted_input):].strip()

        logger.debug(
            f"Generation time: {generation_time:.2f}s, throughput: {tps:.2f} tokens/s"
        )

        return response

    def stream_generate_response(self, question: str, max_new_tokens: Optional[int] = None):
        """Stream tokens as they are generated"""
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens

        formatted_input = self._format_input(question)

        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length - max_new_tokens,
            padding=True
        )

        if self.config.use_directml:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            do_sample=self.config.do_sample,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
            use_cache=self.config.use_cache,
        )

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in streamer:
            yield new_text
        thread.join()
    
    def _format_input(self, question: str) -> str:
        """Format the input for the model"""
        return f"Question: {question}\n\nSolution:"
    
    def extract_code(self, response: str) -> str:
        """Extract code from the model response"""
        # Look for code blocks
        if "```" in response:
            # Extract content between code blocks
            start = response.find("```")
            end = response.find("```", start + 3)
            if end != -1:
                code = response[start + 3:end].strip()
                # Remove language identifier if present
                if code.startswith("python") or code.startswith("cpp") or code.startswith("java"):
                    lines = code.split("\n", 1)
                    if len(lines) > 1:
                        code = lines[1]
                return code
        
        # If no code blocks, return the whole response
        return response
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "model_path": self.config.model_path,
            "device": str(self.device),
            "max_length": self.config.max_length,
            "vocab_size": self.tokenizer.vocab_size,
            "model_type": type(self.model).__name__,
            "use_directml": self.config.use_directml,
            "mixed_precision": self.config.mixed_precision
        }

class BatchInferenceEngine:
    """Batch inference engine for processing multiple questions"""
    
    def __init__(self, config: InferenceConfig, batch_size: int = 4):
        self.config = config
        self.batch_size = batch_size
        self.engine = DirectMLInferenceEngine(config)
    
    def process_batch(self, questions: List[str]) -> List[str]:
        """Process a batch of questions"""
        results = []
        
        for i in range(0, len(questions), self.batch_size):
            batch = questions[i:i + self.batch_size]
            batch_results = []
            
            for question in batch:
                try:
                    response = self.engine.generate_response(question)
                    batch_results.append(response)
                except Exception as e:
                    logger.error(f"Error processing question: {e}")
                    batch_results.append("Error: Could not generate response")
            
            results.extend(batch_results)
        
        return results

def create_inference_config(model_path: str = "fine_tuned_model") -> InferenceConfig:
    """Create inference configuration from model path"""
    config_path = os.path.join(model_path, "config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        
        # Create inference config with saved settings
        config = InferenceConfig(
            model_path=model_path,
            max_length=saved_config.get('max_length', 2048),
            use_directml=saved_config.get('use_directml', True),
            mixed_precision=saved_config.get('mixed_precision', True),
            load_in_8bit=saved_config.get('load_in_8bit', False),
            load_in_4bit=saved_config.get('load_in_4bit', False),
        )
    else:
        config = InferenceConfig(model_path=model_path)
    
    return config

def test_inference():
    """Test the inference engine"""
    logger.info("Testing inference engine...")
    
    # Create config
    config = create_inference_config()
    
    # Create engine
    engine = DirectMLInferenceEngine(config)
    
    # Test questions
    test_questions = [
        "Write a function to find the maximum element in an array.",
        "Implement a binary search algorithm.",
        "Write a function to check if a string is a palindrome."
    ]
    
    for i, question in enumerate(test_questions, 1):
        logger.info(f"\nTest {i}: {question}")
        
        try:
            response = engine.generate_response(question)
            code = engine.extract_code(response)
            
            logger.info(f"Response: {response[:200]}...")
            logger.info(f"Extracted code: {code[:100]}...")
            
        except Exception as e:
            logger.error(f"Error in test {i}: {e}")
    
    # Print model info
    model_info = engine.get_model_info()
    logger.info(f"\nModel info: {model_info}")

def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test inference engine")
    parser.add_argument("--model_path", type=str, default="fine_tuned_model", help="Path to fine-tuned model")
    parser.add_argument("--question", type=str, help="Test question")
    parser.add_argument("--batch", action="store_true", help="Test batch inference")
    
    args = parser.parse_args()
    
    if args.question:
        # Single question test
        config = create_inference_config(args.model_path)
        engine = DirectMLInferenceEngine(config)
        
        response = engine.generate_response(args.question)
        code = engine.extract_code(response)
        
        print(f"\nQuestion: {args.question}")
        print(f"\nResponse:\n{response}")
        print(f"\nExtracted Code:\n{code}")
        
    elif args.batch:
        # Batch test
        config = create_inference_config(args.model_path)
        batch_engine = BatchInferenceEngine(config)
        
        questions = [
            "Write a function to find the sum of two numbers.",
            "Implement a function to reverse a string.",
            "Write a function to check if a number is prime."
        ]
        
        responses = batch_engine.process_batch(questions)
        
        for i, (question, response) in enumerate(zip(questions, responses)):
            print(f"\nQuestion {i+1}: {question}")
            print(f"Response: {response[:200]}...")
    
    else:
        # Full test
        test_inference()

if __name__ == "__main__":
    main() 