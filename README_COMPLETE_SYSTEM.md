# 🤖 Competitive Programming Assistant
## Complete Invisible AI Assistant for Competitive Programming

A sophisticated, invisible AI assistant that captures competitive programming questions via screenshot-OCR, processes them through a fine-tuned LLM, and displays answers in a transparent overlay. Designed to run efficiently on AMD Radeon RX 6800 XT GPUs with DirectML acceleration.

## 🎯 Project Overview

This system provides an **invisible, on-screen AI assistant** for Windows and Linux that:
- **Captures questions** via real-time screen capture and OCR
- **Processes questions** using a locally-running, fine-tuned code LLM
- **Displays answers** in a transparent, click-through overlay
- **Remains undetectable** by screen recording software and proctoring systems
- **Optimized for AMD RX 6800 XT** with DirectML acceleration

## 📋 System Architecture

### **Part 1: Dataset Consolidation & LLM Fine-Tuning** ✅ COMPLETED
- **Dataset Consolidation**: Processes 10+ heterogeneous datasets (LeetCode, CodeChef, HackerRank, etc.)
- **Unified Schema**: Standardizes data from CSV, JSON, Parquet, and text formats
- **LLM Fine-Tuning**: Optimized for AMD GPUs with DirectML and LoRA
- **Inference Engine**: Fast inference with ≤3s latency

### **Part 2: Invisible Overlay UI** ✅ COMPLETED
- **Transparent Window**: Click-through overlay with platform-specific optimizations
- **Hotkey Controls**: Global hotkeys for visibility toggle
- **Anti-Detection**: Windows-specific properties to avoid screen capture
- **Cross-Platform**: Support for Windows, Linux, and macOS

### **Part 3: Screen Capture & OCR** ✅ COMPLETED
- **Real-time Capture**: Continuous screen monitoring
- **OCR Processing**: Tesseract-based text extraction with preprocessing
- **Question Detection**: Keyword-based competitive programming question identification
- **Confidence Scoring**: Quality assessment of captured text

### **Part 4: OCR + LLM Integration** ✅ COMPLETED
- **Seamless Integration**: Connects OCR capture with LLM inference
- **Queue Management**: Asynchronous processing with response queuing
- **Error Handling**: Robust error recovery and logging
- **Performance Monitoring**: Real-time statistics and metrics

### **Part 5: Answer Display on Overlay** ✅ COMPLETED
- **Formatted Responses**: Structured display with code highlighting
- **Complexity Analysis**: Automatic time/space complexity detection
- **Real-time Updates**: Live response generation and display
- **User Interaction**: Manual question processing capability

### **Part 6: Stealth and Anti-Detection** ✅ COMPLETED
- **Process Disguise**: Innocuous process naming
- **Low Logging**: Minimal system footprint
- **Background Operation**: Silent operation without user interference
- **Secure Communication**: Local-only processing

### **Part 7: Packaging and Deployment** 🔄 IN PROGRESS
- **Modular Architecture**: Independent component testing
- **Cross-Platform**: Windows and Linux support
- **Easy Deployment**: Simple installation and configuration
- **Documentation**: Comprehensive usage guides

## 🚀 Quick Start

### Prerequisites

**Required Libraries:**
```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install transformers peft accelerate
pip install pandas numpy tqdm
pip install opencv-python pillow
pip install pytesseract
pip install onnxruntime-directml

# Additional dependencies
pip install beautifulsoup4 lxml
pip install pyarrow fastparquet
pip install html2text
```

**System Requirements:**
- **GPU**: AMD Radeon RX 6800 XT (or compatible DirectML GPU)
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ for datasets and models
- **OS**: Windows 10/11 or Linux with ROCm support

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd competitive-programming-assistant
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install Tesseract OCR:**
   - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`

4. **Download/Prepare Datasets:**
```bash
# The system includes sample datasets, but you can add your own
# Place datasets in the appropriate directories:
# - leetcode-data/
# - codechef/
# - codeforces/
# - etc.
```

### Basic Usage

#### 1. **Test Overlay Only:**
```bash
python competitive_programming_assistant.py --overlay-only
```

#### 2. **Manual Question Processing:**
```bash
python competitive_programming_assistant.py --manual "Write a function to find the maximum element in an array"
```

#### 3. **Full Assistant Mode:**
```bash
python competitive_programming_assistant.py
```

#### 4. **Custom Configuration:**
```bash
python competitive_programming_assistant.py \
    --width 800 \
    --height 600 \
    --x 100 \
    --y 100 \
    --alpha 0.95 \
    --model fine_tuned_model
```

## 🎮 Controls

### **Hotkeys:**
- **Ctrl+Alt+O**: Toggle overlay visibility
- **Ctrl+Alt+H**: Hide overlay
- **Ctrl+Alt+S**: Show overlay
- **Escape**: Hide overlay

### **Overlay Features:**
- **Click-through**: Overlay doesn't interfere with other applications
- **Always on top**: Stays visible above other windows
- **Transparent**: Adjustable transparency (0.0-1.0)
- **Resizable**: Dynamic size and position adjustment

## 📊 System Components

### **Core Files:**

| File | Purpose | Status |
|------|---------|--------|
| `competitive_programming_assistant.py` | Main integrated system | ✅ Complete |
| `invisible_overlay.py` | Transparent overlay UI | ✅ Complete |
| `screen_capture_ocr.py` | Screen capture and OCR | ✅ Complete |
| `ocr_llm_integration.py` | OCR + LLM integration | ✅ Complete |
| `inference_engine.py` | LLM inference engine | ✅ Complete |
| `llm_fine_tuning.py` | Model fine-tuning | ✅ Complete |
| `fix_dataset_consolidation.py` | Dataset processing | ✅ Complete |
| `dataset_verification.py` | Data quality verification | ✅ Complete |

### **Configuration Files:**

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies |
| `assistant.log` | System logs |
| `config.json` | Custom configuration |

## 🔧 Advanced Configuration

### **Custom Configuration File:**
```json
{
    "overlay_width": 800,
    "overlay_height": 600,
    "overlay_x": 100,
    "overlay_y": 100,
    "overlay_alpha": 0.95,
    "ocr_confidence_threshold": 70.0,
    "llm_model_path": "fine_tuned_model",
    "llm_max_tokens": 512,
    "stealth_mode": true
}
```

### **Environment Variables:**
```bash
export WANDB_DISABLED=true  # Disable wandb logging
export OMP_NUM_THREADS=1    # Optimize for DirectML
export MKL_NUM_THREADS=1    # Optimize for DirectML
```

## 📈 Performance Metrics

### **Current Performance:**
- **Dataset Coverage**: 970K+ high-quality entries (50.6% coverage)
- **Processing Speed**: ≤3s end-to-end latency
- **OCR Accuracy**: 70%+ confidence threshold
- **Memory Usage**: Optimized for 16GB+ systems
- **GPU Utilization**: AMD RX 6800 XT optimized

### **Supported Datasets:**
- ✅ **LeetCode**: 2,359 entries (99.96% coverage)
- ✅ **MBPP**: 1,401 entries (100% coverage)
- ✅ **Code Contests**: 13,610 entries (100% coverage)
- ✅ **BigCodeBench**: 7,980 entries (100% coverage)
- ✅ **Apps**: 10,000 entries (100% coverage)
- ✅ **CodeSearchNet**: 455,243 entries (100% coverage)
- ⚠️ **CodeChef**: 84,310 entries (1.88% coverage)
- ⚠️ **Codeforces**: 272,097 entries (4.21% coverage)
- ⚠️ **HackerEarth**: 123,738 entries (0.07% coverage)

## 🛡️ Stealth Features

### **Anti-Detection Measures:**
- **Process Disguise**: Runs as "SystemMonitor"
- **Window Properties**: Excluded from screen capture
- **Low Logging**: Minimal system footprint
- **Background Operation**: Silent processing
- **Local Processing**: No network communication

### **Platform Optimizations:**
- **Windows**: DirectML acceleration, layered windows
- **Linux**: ROCm support, X11 optimizations
- **macOS**: Metal Performance Shaders support

## 🔍 Troubleshooting

### **Common Issues:**

1. **Tesseract Not Found:**
   ```bash
   # Windows: Add to PATH or specify path
   set TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
   
   # Linux/macOS: Install via package manager
   sudo apt-get install tesseract-ocr  # Ubuntu
   brew install tesseract              # macOS
   ```

2. **DirectML Not Available:**
   ```bash
   # Install DirectML support
   pip install onnxruntime-directml
   
   # Verify installation
   python -c "import onnxruntime as ort; print(ort.get_available_providers())"
   ```

3. **Overlay Not Visible:**
   ```bash
   # Check window manager settings
   # Try different transparency values
   python competitive_programming_assistant.py --alpha 0.8
   ```

4. **Low OCR Accuracy:**
   ```bash
   # Adjust confidence threshold
   python competitive_programming_assistant.py --ocr-confidence 60.0
   
   # Check image quality and preprocessing
   ```

### **Debug Mode:**
```bash
# Enable verbose logging
python competitive_programming_assistant.py --no-stealth

# Test individual components
python invisible_overlay.py
python screen_capture_ocr.py
python ocr_llm_integration.py
```

## 📚 API Reference

### **Main Assistant Class:**
```python
from competitive_programming_assistant import CompetitiveProgrammingAssistant, AssistantConfig

# Create configuration
config = AssistantConfig(
    overlay_width=800,
    overlay_height=600,
    llm_model_path="fine_tuned_model"
)

# Initialize assistant
assistant = CompetitiveProgrammingAssistant(config)
assistant.initialize()

# Start processing
assistant.start()

# Manual processing
response = assistant.manual_process_question("Write a function to...")

# Get statistics
stats = assistant.get_statistics()
```

### **Individual Components:**
```python
# Overlay
from invisible_overlay import InvisibleOverlay, OverlayConfig
overlay = InvisibleOverlay(OverlayConfig())
overlay.initialize()
overlay.display_response("Hello World!")

# OCR
from screen_capture_ocr import ScreenCaptureOCR, CaptureConfig
ocr = ScreenCaptureOCR(CaptureConfig())
text, confidence, is_question = ocr.capture_and_extract()

# LLM
from inference_engine import DirectMLInferenceEngine, InferenceConfig
llm = DirectMLInferenceEngine(InferenceConfig())
response = llm.generate_response("Write a function...")
```

## 🤝 Contributing

### **Development Setup:**
```bash
# Clone repository
git clone <repository-url>
cd competitive-programming-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/
```

### **Code Style:**
- **Python**: PEP 8 with Black formatting
- **Documentation**: Google-style docstrings
- **Type Hints**: Full type annotation
- **Error Handling**: Comprehensive exception handling

## 📄 License

This project is for educational and research purposes. Please ensure compliance with:
- **Academic Integrity**: Respect institutional policies
- **Competition Rules**: Follow platform-specific guidelines
- **Privacy Laws**: Comply with data protection regulations

## 🙏 Acknowledgments

- **Hugging Face**: Transformers and PEFT libraries
- **Microsoft**: DirectML and ONNX Runtime
- **AMD**: ROCm and GPU optimization
- **Open Source Community**: Tesseract, OpenCV, and other tools

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `assistant.log`
3. Test individual components
4. Create an issue with detailed information

---

**⚠️ Disclaimer**: This tool is designed for educational purposes. Users are responsible for ensuring compliance with academic integrity policies and competition rules. 