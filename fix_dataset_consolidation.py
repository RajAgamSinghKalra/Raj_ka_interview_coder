#!/usr/bin/env python3
"""
Fixed Dataset Consolidation Script
Addresses the issues with the original consolidation script

The main problems were:
1. Parquet files not being parsed correctly
2. Large text files causing memory issues
3. Complex nested structures not being handled
4. File format detection issues
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class UnifiedSchema:
    """Unified schema for all competitive programming data"""
    question_text: str
    answer_text: str = ""
    solution_code: str = ""
    language: str = ""
    question_type: str = ""
    source_site: str = ""
    difficulty: str = ""
    tags: List[str] = field(default_factory=list)
    problem_id: str = ""
    constraints: str = ""
    examples: List[Dict] = field(default_factory=list)
    test_cases: List[Dict] = field(default_factory=list)
    time_limit: Optional[int] = None
    memory_limit: Optional[int] = None

class FixedDatasetConsolidator:
    """Fixed dataset consolidator that properly handles all dataset types"""
    
    def __init__(self, root_dir: str, output_dir: str = "consolidated_output"):
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Supported file extensions
        self.supported_extensions = {'.csv', '.jsonl', '.json', '.parquet', '.txt'}
        
        # Dataset patterns for discovery
        self.dataset_patterns = {
            'leetcode': ['leetcode-data/*.jsonl', 'leetcode-data/*.json'],
            'mbpp': ['mbpp-data/data/*.jsonl', 'mbpp-data/data/*.json'],
            'code_contests': ['code_contests-data/data/*.parquet'],
            'bigcodebench': ['bigcodebench-data/data/*.parquet'],
            'apps': ['apps-data/*.jsonl', 'apps-data/*.json'],
            'ds1000': ['ds1000-data/*.jsonl'],
            'codesearchnet': ['codesearchnet_python/data/*.parquet'],
            'codechef': ['description2code_current/codechef/**/*.txt', 'description2code_current/codechef/**/*.json'],
            'codeforces': ['description2code_current/codeforces/**/*.txt', 'description2code_current/codeforces/**/*.json'],
            'hackerearth': ['description2code_current/hackerearth/**/*.txt', 'description2code_current/hackerearth/**/*.json'],
            'generic': ['*.csv', '*.jsonl', '*.json', '*.parquet']
        }

    def discover_all_files(self) -> Dict[str, List[Path]]:
        """Discover all files using the same patterns as verification"""
        logger.info("Discovering all dataset files...")
        
        dataset_files = {}
        
        for dataset_name, patterns in self.dataset_patterns.items():
            dataset_files[dataset_name] = []
            
            for pattern in patterns:
                files = list(self.root_dir.glob(pattern))
                if files:
                    dataset_files[dataset_name].extend(files)
                    logger.info(f"Found {len(files)} files for {dataset_name}")
            
            # Remove duplicates
            dataset_files[dataset_name] = list(set(dataset_files[dataset_name]))
        
        total_files = sum(len(files) for files in dataset_files.values())
        logger.info(f"Total files discovered: {total_files}")
        
        return dataset_files

    def parse_file(self, file_path: Path, dataset_name: str) -> List[UnifiedSchema]:
        """Parse a single file based on its type and dataset"""
        try:
            extension = file_path.suffix.lower()
            
            if extension == '.parquet':
                return self._parse_parquet_file(file_path, dataset_name)
            elif extension == '.jsonl':
                return self._parse_jsonl_file(file_path, dataset_name)
            elif extension == '.json':
                return self._parse_json_file(file_path, dataset_name)
            elif extension == '.csv':
                return self._parse_csv_file(file_path, dataset_name)
            elif extension == '.txt':
                return self._parse_txt_file(file_path, dataset_name)
            else:
                logger.warning(f"Unsupported file type: {extension} for {file_path}")
                return []
                
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return []

    def _parse_parquet_file(self, file_path: Path, dataset_name: str) -> List[UnifiedSchema]:
        """Parse parquet files"""
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"Parsing parquet file {file_path} with {len(df)} rows")
            
            entries = []
            
            for idx, row in df.iterrows():
                try:
                    entry = self._convert_row_to_unified_schema(row, dataset_name, file_path)
                    if entry:
                        entries.append(entry)
                except Exception as e:
                    logger.warning(f"Error processing row {idx} in {file_path}: {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(entries)} entries from {file_path}")
            return entries
            
        except Exception as e:
            logger.error(f"Error reading parquet file {file_path}: {e}")
            return []

    def _parse_jsonl_file(self, file_path: Path, dataset_name: str) -> List[UnifiedSchema]:
        """Parse JSONL files"""
        try:
            entries = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        if line.strip():
                            data = json.loads(line.strip())
                            entry = self._convert_dict_to_unified_schema(data, dataset_name, file_path)
                            if entry:
                                entries.append(entry)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON at line {line_num} in {file_path}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num} in {file_path}: {e}")
                        continue
            
            logger.info(f"Successfully parsed {len(entries)} entries from {file_path}")
            return entries
            
        except Exception as e:
            logger.error(f"Error reading JSONL file {file_path}: {e}")
            return []

    def _parse_json_file(self, file_path: Path, dataset_name: str) -> List[UnifiedSchema]:
        """Parse JSON files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            entries = []
            
            if isinstance(data, list):
                for item in data:
                    entry = self._convert_dict_to_unified_schema(item, dataset_name, file_path)
                    if entry:
                        entries.append(entry)
            elif isinstance(data, dict):
                entry = self._convert_dict_to_unified_schema(data, dataset_name, file_path)
                if entry:
                    entries.append(entry)
            
            logger.info(f"Successfully parsed {len(entries)} entries from {file_path}")
            return entries
            
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")
            return []

    def _parse_csv_file(self, file_path: Path, dataset_name: str) -> List[UnifiedSchema]:
        """Parse CSV files"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Parsing CSV file {file_path} with {len(df)} rows")
            
            entries = []
            
            for idx, row in df.iterrows():
                try:
                    entry = self._convert_row_to_unified_schema(row, dataset_name, file_path)
                    if entry:
                        entries.append(entry)
                except Exception as e:
                    logger.warning(f"Error processing row {idx} in {file_path}: {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(entries)} entries from {file_path}")
            return entries
            
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            return []

    def _parse_txt_file(self, file_path: Path, dataset_name: str) -> List[UnifiedSchema]:
        """Parse text files (for codechef, codeforces, hackerearth)"""
        try:
            # For text files, we'll create a simple entry with the file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if len(content.strip()) < 10:  # Skip very short files
                return []
            
            # Extract problem name from filename
            problem_name = file_path.stem
            
            entry = UnifiedSchema(
                question_text=content[:2000],  # Limit to first 2000 chars
                solution_code="",  # Text files usually don't contain solutions
                source_site=dataset_name,
                problem_id=problem_name,
                question_type="coding"
            )
            
            return [entry]
            
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            return []

    def _convert_row_to_unified_schema(self, row, dataset_name: str, file_path: Path) -> Optional[UnifiedSchema]:
        """Convert a pandas row to UnifiedSchema"""
        try:
            # Convert row to dict
            if hasattr(row, 'to_dict'):
                data = row.to_dict()
            else:
                data = dict(row)
            
            return self._convert_dict_to_unified_schema(data, dataset_name, file_path)
            
        except Exception as e:
            logger.warning(f"Error converting row to schema: {e}")
            return None

    def _convert_dict_to_unified_schema(self, data: Dict, dataset_name: str, file_path: Path) -> Optional[UnifiedSchema]:
        """Convert a dictionary to UnifiedSchema based on dataset type"""
        try:
            # Extract fields based on dataset type
            question_text = ""
            solution_code = ""
            problem_id = ""
            difficulty = ""
            language = ""
            
            if dataset_name == 'leetcode':
                question_text = data.get('content', data.get('description', ''))
                problem_id = data.get('title', data.get('slug', ''))
                difficulty = data.get('difficulty', '').lower()
                
                # Handle multiple language solutions
                solutions = []
                for lang in ['java', 'python', 'c++', 'javascript']:
                    if lang in data and data[lang]:
                        solutions.append(f"# {lang}\n{data[lang]}")
                solution_code = '\n\n'.join(solutions)
                
            elif dataset_name == 'mbpp':
                question_text = data.get('text', '')
                solution_code = data.get('code', '')
                problem_id = str(data.get('task_id', ''))
                
            elif dataset_name == 'code_contests':
                question_text = data.get('description', '')
                problem_id = data.get('name', '')
                difficulty = str(data.get('difficulty', ''))
                
                # Handle solutions
                if 'solutions' in data and isinstance(data['solutions'], dict):
                    solutions = []
                    for lang_code, solution_data in data['solutions'].items():
                        if 'solution' in solution_data:
                            solutions.append(f"# Language {lang_code}\n{solution_data['solution']}")
                    solution_code = '\n\n'.join(solutions)
                    
            elif dataset_name == 'bigcodebench':
                question_text = data.get('prompt', data.get('description', ''))
                solution_code = data.get('canonical_solution', '')
                problem_id = str(data.get('task_id', ''))
                
            elif dataset_name == 'apps':
                question_text = data.get('question', '')
                solution_code = data.get('answer', '')
                problem_id = str(data.get('id', ''))
                
            elif dataset_name == 'ds1000':
                question_text = data.get('question', '')
                solution_code = data.get('answer', '')
                problem_id = str(data.get('id', ''))
                
            elif dataset_name == 'codesearchnet':
                question_text = data.get('docstring', '')
                solution_code = data.get('code', '')
                problem_id = str(data.get('id', ''))
                
            else:
                # Generic handling for unknown datasets
                question_text = data.get('question', data.get('text', data.get('content', '')))
                solution_code = data.get('answer', data.get('code', data.get('solution', '')))
                problem_id = str(data.get('id', data.get('problem_id', '')))
                difficulty = data.get('difficulty', '')
                language = data.get('language', '')

            # Validate that we have at least some content
            if not question_text.strip() and not solution_code.strip():
                return None

            # Create unified schema entry
            entry = UnifiedSchema(
                question_text=question_text.strip(),
                solution_code=solution_code.strip(),
                source_site=dataset_name,
                problem_id=problem_id,
                difficulty=difficulty,
                language=language,
                question_type=self._infer_question_type(question_text)
            )
            
            return entry
            
        except Exception as e:
            logger.warning(f"Error converting dict to schema: {e}")
            return None

    def _infer_question_type(self, question_text: str) -> str:
        """Infer question type from text"""
        text_lower = question_text.lower()
        
        if any(word in text_lower for word in ['implement', 'write', 'function', 'class']):
            return 'coding'
        elif any(word in text_lower for word in ['multiple choice', 'select', 'choose']):
            return 'mcq'
        elif any(word in text_lower for word in ['fill', 'blank', '_____']):
            return 'fill_blank'
        elif any(word in text_lower for word in ['debug', 'fix', 'error']):
            return 'debug'
        else:
            return 'coding'  # Default to coding

    def consolidate_all_datasets(self) -> List[UnifiedSchema]:
        """Consolidate all discovered datasets"""
        logger.info("Starting comprehensive dataset consolidation...")
        
        # Discover all files
        dataset_files = self.discover_all_files()
        
        all_entries = []
        
        # Process each dataset
        for dataset_name, files in dataset_files.items():
            logger.info(f"Processing {dataset_name} with {len(files)} files...")
            
            dataset_entries = []
            
            for file_path in tqdm(files, desc=f"Processing {dataset_name}"):
                try:
                    file_entries = self.parse_file(file_path, dataset_name)
                    dataset_entries.extend(file_entries)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue
            
            logger.info(f"Processed {dataset_name}: {len(dataset_entries)} entries")
            all_entries.extend(dataset_entries)
        
        logger.info(f"Total consolidated entries: {len(all_entries)}")
        return all_entries

    def save_consolidated_data(self, entries: List[UnifiedSchema]) -> None:
        """Save consolidated data to files"""
        logger.info("Saving consolidated data...")
        
        # Convert to JSON-serializable format
        data_list = []
        for entry in entries:
            data_list.append({
                'question_text': entry.question_text,
                'answer_text': entry.answer_text,
                'solution_code': entry.solution_code,
                'language': entry.language,
                'question_type': entry.question_type,
                'source_site': entry.source_site,
                'difficulty': entry.difficulty,
                'tags': entry.tags,
                'problem_id': entry.problem_id,
                'constraints': entry.constraints,
                'examples': entry.examples,
                'test_cases': entry.test_cases,
                'time_limit': entry.time_limit,
                'memory_limit': entry.memory_limit
            })
        
        # Create train/validation/test splits
        np.random.shuffle(data_list)
        
        total = len(data_list)
        train_size = int(0.9 * total)
        val_size = int(0.05 * total)
        
        train_data = data_list[:train_size]
        val_data = data_list[train_size:train_size + val_size]
        test_data = data_list[train_size + val_size:]
        
        # Save splits
        with open(self.output_dir / 'train.jsonl', 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        with open(self.output_dir / 'validation.jsonl', 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        with open(self.output_dir / 'test.jsonl', 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Save manifest
        manifest = {
            'total_entries': total,
            'train_entries': len(train_data),
            'validation_entries': len(val_data),
            'test_entries': len(test_data),
            'datasets_processed': list(self.dataset_patterns.keys())
        }
        
        with open(self.output_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Saved {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test entries")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fixed dataset consolidation")
    parser.add_argument("--root_dir", type=str, default=".", help="Root directory")
    parser.add_argument("--output_dir", type=str, default="consolidated_output", help="Output directory")
    
    args = parser.parse_args()
    
    # Create consolidator
    consolidator = FixedDatasetConsolidator(args.root_dir, args.output_dir)
    
    # Consolidate all datasets
    entries = consolidator.consolidate_all_datasets()
    
    # Save consolidated data
    consolidator.save_consolidated_data(entries)
    
    logger.info("Dataset consolidation completed successfully!")

if __name__ == "__main__":
    main() 