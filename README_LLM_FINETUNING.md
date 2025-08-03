# LLM Fine-Tuning Pipeline for Competitive Programming Assistant

## Overview

This project implements a complete pipeline for fine-tuning Large Language Models (LLMs) for competitive programming assistance, optimized for AMD Radeon RX 6800 XT with DirectML support. The system includes dataset consolidation, model fine-tuning, and inference optimization.

## Project Structure

```
Raj_ka_interview_coder/
├── dataset_consolidation.py      # Dataset consolidation pipeline
├── llm_fine_tuning.py           # LLM fine-tuning with DirectML
├── inference_engine.py          # Optimized inference engine
├── dataset_verification.py      # Comprehensive dataset verification
├── test_directml_setup.py       # DirectML compatibility test
├── consolidated_output/         # Consolidated dataset (generated)
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── test.jsonl
│   └── manifest.json
└── fine_tuned_model/           # Fine-tuned model (generated)
```

## Prerequisites

### Environment Setup
- **Conda Environment**: `myenv` (already configured)
- **GPU**: AMD Radeon RX 6800 XT
- **OS**: Windows 10/11 with DirectML support

### Required Libraries
All libraries are installed in the `myenv` conda environment:

```bash
# Core PyTorch with DirectML support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# DirectML Runtime
pip install onnxruntime-directml

# LLM Fine-tuning Libraries
pip install transformers accelerate peft bitsandbytes datasets tokenizers
pip install scipy scikit-learn sentencepiece protobuf safetensors
pip install huggingface-hub wandb

# Data Processing
pip install pandas numpy tqdm beautifulsoup4 lxml pyarrow fastparquet html2text
```

## Quick Start

### 1. Test DirectML Setup
First, verify that DirectML is working correctly with your AMD RX 6800 XT:

```bash
python test_directml_setup.py
```

This will test:
- PyTorch compatibility
- DirectML provider availability
- GPU memory allocation
- Tensor operations

### 2. Verify Dataset Consolidation
Run the comprehensive dataset verification to ensure all datasets are properly processed:

```bash
python dataset_verification.py
```

This will:
- Discover all original dataset files
- Count entries in each dataset
- Verify coverage in consolidated output
- Generate quality metrics
- Create a detailed verification report

### 3. Start Fine-Tuning
Begin the LLM fine-tuning process:

```bash
python llm_fine_tuning.py --model microsoft/DialoGPT-medium --epochs 3 --batch_size 2
```

### 4. Test Inference
Test the fine-tuned model:

```bash
python inference_engine.py --question "Write a function to find the maximum element in an array"
```

## Detailed Usage

### Dataset Consolidation

The dataset consolidation pipeline (`dataset_consolidation.py`) has already processed:
- **573,447 files** (11.13 GB total)
- **6,528 consolidated entries**
- **Train/Validation/Test splits**: 5,874 / 324 / 330

### Fine-Tuning Configuration

The fine-tuning script supports various configurations:

```bash
# Basic fine-tuning
python llm_fine_tuning.py --model microsoft/DialoGPT-medium --epochs 3

# Advanced configuration
python llm_fine_tuning.py \
    --model microsoft/DialoGPT-medium \
    --epochs 5 \
    --batch_size 4 \
    --output_dir my_fine_tuned_model \
    --no_wandb
```

### Model Options

Recommended models for AMD RX 6800 XT:

1. **microsoft/DialoGPT-medium** (345M parameters) - Fast, good for testing
2. **microsoft/DialoGPT-large** (774M parameters) - Better quality
3. **microsoft/DialoGPT-mega** (1.5B parameters) - Best quality, slower

### DirectML Optimization

The fine-tuning pipeline includes several DirectML optimizations:

- **Mixed Precision**: FP16 for faster training
- **LoRA**: Parameter-efficient fine-tuning
- **Gradient Accumulation**: Effective larger batch sizes
- **Memory Optimization**: Low CPU memory usage
- **Device Mapping**: Automatic GPU/CPU placement

## Performance Expectations

### AMD RX 6800 XT Performance

| Model Size | Training Time | Memory Usage | Inference Speed |
|------------|---------------|--------------|-----------------|
| 345M (medium) | ~2-3 hours | 4-6 GB | ~15-20 tokens/s |
| 774M (large) | ~4-6 hours | 6-8 GB | ~10-15 tokens/s |
| 1.5B (mega) | ~8-12 hours | 8-12 GB | ~5-10 tokens/s |

### Optimization Tips

1. **Batch Size**: Start with 2, increase if memory allows
2. **Gradient Accumulation**: Use 4-8 steps for effective larger batches
3. **Mixed Precision**: Always enabled for DirectML
4. **LoRA**: Reduces memory usage by 70-80%

## Monitoring and Logging

### WandB Integration
The fine-tuning process automatically logs to Weights & Biases:

- Training loss curves
- Validation metrics
- Model checkpoints
- System metrics (GPU usage, memory)

### Local Logging
All processes log to console with detailed information:
- Dataset statistics
- Training progress
- Memory usage
- Performance metrics

## Troubleshooting

### Common Issues

1. **DirectML Not Available**
   ```bash
   # Check DirectML installation
   python -c "import onnxruntime as ort; print(ort.get_available_providers())"
   ```

2. **Out of Memory**
   - Reduce batch size
   - Increase gradient accumulation steps
   - Use smaller model
   - Enable mixed precision

3. **Slow Training**
   - Check DirectML provider is being used
   - Verify GPU memory allocation
   - Monitor CPU usage

### Performance Optimization

1. **Memory Optimization**
   ```python
   # In llm_fine_tuning.py
   config.batch_size = 1  # Reduce batch size
   config.gradient_accumulation_steps = 8  # Increase accumulation
   ```

2. **Speed Optimization**
   ```python
   # Enable all optimizations
   config.mixed_precision = True
   config.use_directml = True
   ```

## Next Steps

### 1. Complete Fine-Tuning
Run the fine-tuning process with your preferred model:

```bash
python llm_fine_tuning.py --model microsoft/DialoGPT-large --epochs 5
```

### 2. Evaluate Model Performance
Test the fine-tuned model on various competitive programming problems:

```bash
python inference_engine.py --batch
```

### 3. Integration with Invisible Overlay
The next phase will integrate the fine-tuned model with:
- Screenshot OCR for question capture
- Invisible overlay for answer display
- Anti-detection mechanisms

### 4. Advanced Optimizations
Future improvements:
- Model quantization for faster inference
- Streaming token generation
- Multi-language support
- Real-time question processing

## File Descriptions

### Core Scripts

- **`dataset_consolidation.py`**: Processes heterogeneous datasets into unified format
- **`llm_fine_tuning.py`**: Fine-tunes LLMs with DirectML optimization
- **`inference_engine.py`**: Fast inference for competitive programming questions
- **`dataset_verification.py`**: Comprehensive dataset verification and reporting
- **`test_directml_setup.py`**: DirectML compatibility testing

### Configuration

- **`FineTuningConfig`**: Centralized configuration for fine-tuning
- **`InferenceConfig`**: Configuration for inference optimization
- **`UnifiedSchema`**: Standardized data format for all datasets

### Output Files

- **`consolidated_output/`**: Processed and unified datasets
- **`fine_tuned_model/`**: Fine-tuned model and configuration
- **`verification_report.json`**: Detailed dataset verification report

## Support

For issues or questions:
1. Check the troubleshooting section
2. Run `test_directml_setup.py` to verify environment
3. Review logs for detailed error messages
4. Ensure all libraries are properly installed in `myenv`

## License

This project is for educational and research purposes. Please ensure compliance with all applicable laws and regulations regarding AI assistance in competitive programming contexts. 