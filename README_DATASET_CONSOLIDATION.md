# Dataset Consolidation & LLM Fine-Tuning Pipeline

This repository contains a comprehensive pipeline for consolidating competitive programming datasets from multiple sources and formats into a unified schema for LLM fine-tuning.

## 🎯 Overview

The pipeline handles heterogeneous competitive programming datasets from various sources:
- **LeetCode** (JSONL format)
- **CodeChef** (various formats)
- **Codeforces** (various formats)
- **MBPP** (JSONL format)
- **Code Contests** (Parquet format)
- **Apps** (various formats)
- **DS1000** (JSONL format)
- **BigCodeBench** (Parquet format)

## 📋 Supported Formats

- **CSV/TSV**: Tabular data with various encodings
- **JSON/JSONL**: Structured data in JSON format
- **Parquet**: Columnar data format
- **TXT**: Plain text files
- **HTML**: Web content with text extraction
- **XML**: Structured markup data
- **Pickle**: Python serialized objects
- **ZIP/TAR**: Compressed archives (automatically extracted)

## 🏗️ Architecture

### Unified Schema

All data is normalized to a unified schema with the following fields:

```python
@dataclass
class UnifiedSchema:
    question_text: str          # Problem statement/description
    answer_text: str           # Text explanation/answer
    solution_code: str         # Code solution
    language: str              # Programming language
    question_type: str         # MCQ, coding, fill-blank, etc.
    source_site: str           # Original platform (leetcode, codechef, etc.)
    difficulty: str            # Easy, medium, hard
    tags: List[str]            # Problem tags/categories
    problem_id: str            # Unique problem identifier
    constraints: str           # Problem constraints
    examples: List[Dict]       # Sample inputs/outputs
    test_cases: List[Dict]     # Test cases
    time_limit: Optional[int]  # Time limit in seconds
    memory_limit: Optional[int] # Memory limit in bytes
```

### Pipeline Stages

1. **File Discovery**: Recursively scan directories and build inventory
2. **Unified Parsing**: Route files to appropriate parsers
3. **Schema Harmonization**: Map disparate schemas to unified format
4. **Quality Filtering**: Remove low-quality entries
5. **Deduplication**: Remove duplicate questions
6. **Language Variants**: Create additional language versions (optional)
7. **Data Splitting**: Create train/validation/test splits
8. **Output Generation**: Save consolidated data

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run consolidation on current directory
python dataset_consolidation.py

# Specify custom directories
python dataset_consolidation.py --root_dir ./datasets --output_dir ./consolidated

# Use more worker threads for faster processing
python dataset_consolidation.py --max_workers 8

# Skip language variant creation
python dataset_consolidation.py --no_variants
```

### Test the Pipeline

```bash
python test_consolidation.py
```

## 📊 Output Structure

The pipeline generates the following output:

```
consolidated_dataset/
├── train.jsonl           # Training data (90% of data)
├── validation.jsonl      # Validation data (5% of data)
├── test.jsonl           # Test data (5% of data)
├── manifest.json        # Dataset statistics and distributions
└── file_inventory.json  # Original file discovery results
```

### Manifest Statistics

The `manifest.json` file contains:
- Total number of entries
- Split sizes (train/validation/test)
- Source distribution (by platform)
- Difficulty distribution
- Question type distribution

## 🔧 Configuration

### Schema Mappings

The pipeline automatically detects data sources and applies appropriate schema mappings:

- **LeetCode**: Maps `content` → `question_text`, `java/python/c++` → `solution_code`
- **MBPP**: Maps `text` → `question_text`, `code` → `solution_code`
- **Code Contests**: Maps `description` → `question_text`, `solutions` → `solution_code`

### Quality Filters

Entries are filtered based on:
- Minimum question text length (20 characters)
- Maximum question text length (50,000 characters)
- Must have either answer text or solution code
- No excessive word repetition (>20% frequency)

### Deduplication

Uses MD5 hashing of normalized question text:
- Converts to lowercase
- Removes punctuation
- Normalizes whitespace

## 🎛️ Advanced Usage

### Custom Schema Mapping

You can extend the schema mappings by modifying the `schema_mappings` dictionary in the `DatasetConsolidator` class:

```python
self.schema_mappings['custom_source'] = {
    'question_text': ['problem', 'description'],
    'solution_code': ['code', 'solution'],
    # ... other mappings
}
```

### Custom Quality Filters

Modify the `_validate_entry` method to add custom validation rules:

```python
def _validate_entry(self, entry: UnifiedSchema) -> bool:
    # Add your custom validation logic here
    if len(entry.question_text) < 50:  # Custom minimum length
        return False
    return True
```

### Parallel Processing

The pipeline uses ThreadPoolExecutor for parallel file processing. Adjust `max_workers` based on your system:

- **CPU-bound**: Set to number of CPU cores
- **I/O-bound**: Set to 2-4x number of CPU cores
- **Memory-constrained**: Reduce to avoid memory issues

## 📈 Performance Considerations

### Memory Usage

- Large parquet files are processed in chunks
- Temporary files are cleaned up automatically
- Consider processing large datasets in batches

### Processing Speed

- Parallel processing for file parsing
- Efficient deduplication using hashing
- Streaming JSONL output (no memory accumulation)

### Storage Requirements

- Output size is typically 20-50% of input size (after deduplication)
- JSONL format provides good compression
- Consider using parquet for very large datasets

## 🔍 Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `max_workers` or process smaller batches
2. **Encoding Errors**: Files are automatically tried with multiple encodings
3. **Large Files**: Parquet files are processed efficiently, but very large files may need chunking

### Debug Mode

Enable detailed logging by modifying the logging level:

```python
logging.basicConfig(level=logging.DEBUG)
```

### File Inventory

Check `file_inventory.json` to see which files were discovered and their estimated sizes.

## 🤝 Contributing

To add support for new data sources:

1. Add schema mapping in `schema_mappings`
2. Add source detection in `_detect_source_site`
3. Add custom parsing logic if needed
4. Update documentation

## 📄 License

This project is part of the Interview Coder Clone system. See the main repository for license information.

## 🆘 Support

For issues and questions:
1. Check the logs in `dataset_consolidation.log`
2. Run the test script to verify basic functionality
3. Review the file inventory to ensure all files are discovered 