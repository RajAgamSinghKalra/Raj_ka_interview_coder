#!/usr/bin/env python3
"""
Dataset Consolidation & LLM Fine-Tuning Pipeline
================================================

This script consolidates competitive programming datasets from multiple sources
and formats into a unified schema for LLM fine-tuning.

Supported formats:
- CSV, TSV, JSON, JSONL, TXT, HTML, XML, Pickle
- Parquet files
- Nested ZIP/TAR archives
- Various competitive programming platforms (LeetCode, CodeChef, etc.)

Author: AI Assistant
Date: 2024
"""

import os
import json
import pandas as pd
import numpy as np
import hashlib
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import zipfile
import tarfile
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from tqdm import tqdm
import pickle
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
try:
    import html2text
except ImportError:
    html2text = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_consolidation.log'),
        logging.StreamHandler()
    ]
)
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
    tags: List[str] = None
    problem_id: str = ""
    constraints: str = ""
    examples: List[Dict] = None
    test_cases: List[Dict] = None
    time_limit: Optional[int] = None
    memory_limit: Optional[int] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.examples is None:
            self.examples = []
        if self.test_cases is None:
            self.test_cases = [] 

class DatasetConsolidator:
    """Main class for consolidating datasets from multiple sources"""
    
    def __init__(self, root_dir: str, output_dir: str = "consolidated_dataset"):
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # File type mappings
        self.file_extensions = {
            '.csv': self._parse_csv,
            '.tsv': self._parse_tsv,
            '.json': self._parse_json,
            '.jsonl': self._parse_jsonl,
            '.txt': self._parse_txt,
            '.html': self._parse_html,
            '.xml': self._parse_xml,
            '.pkl': self._parse_pickle,
            '.parquet': self._parse_parquet,
            '.zip': self._extract_archive,
            '.tar': self._extract_archive,
            '.tar.gz': self._extract_archive,
            '.tgz': self._extract_archive
        }
        
        # Schema mappings for different sources
        self.schema_mappings = {
            'unknown': {
                'question_text': ['question', 'Question', 'text', 'content', 'description', 'problem'],
                'answer_text': ['answer', 'Answer', 'explanation', 'solution_text'],
                'solution_code': ['code', 'Code', 'solution', 'Solution', 'implementation'],
                'language': ['language', 'lang', 'Language'],
                'question_type': ['type', 'category', 'Type'],
                'source_site': ['source', 'platform', 'Source'],
                'difficulty': ['difficulty', 'level', 'Difficulty'],
                'tags': ['tags', 'categories', 'topics', 'Tags'],
                'problem_id': ['id', 'problem_id', 'task_id', 'Id'],
                'constraints': ['constraints', 'requirements', 'Constraints'],
                'examples': ['examples', 'sample_inputs', 'Examples'],
                'test_cases': ['test_cases', 'tests', 'test_list', 'TestCases'],
                'time_limit': ['time_limit', 'time', 'TimeLimit'],
                'memory_limit': ['memory_limit', 'memory', 'MemoryLimit']
            },
            'leetcode': {
                'question_text': ['content', 'description', 'problem', 'question'],
                'answer_text': ['explanation', 'answer'],
                'solution_code': ['java', 'python', 'c++', 'javascript', 'solution'],
                'language': ['language', 'lang'],
                'question_type': ['type', 'category'],
                'source_site': ['source', 'platform'],
                'difficulty': ['difficulty', 'level'],
                'tags': ['tags', 'categories', 'topics'],
                'problem_id': ['id', 'problem_id', 'slug'],
                'constraints': ['constraints', 'requirements'],
                'examples': ['examples', 'sample_inputs'],
                'test_cases': ['test_cases', 'tests'],
                'time_limit': ['time_limit', 'time'],
                'memory_limit': ['memory_limit', 'memory']
            },
            'codechef': {
                'question_text': ['description', 'problem', 'question'],
                'solution_code': ['solution', 'code', 'answer'],
                'language': ['language', 'lang'],
                'difficulty': ['difficulty', 'level'],
                'tags': ['tags', 'categories'],
                'problem_id': ['id', 'problem_id', 'code'],
                'constraints': ['constraints', 'requirements'],
                'examples': ['examples', 'sample_inputs'],
                'test_cases': ['test_cases', 'tests']
            },
            'codeforces': {
                'question_text': ['description', 'problem', 'question'],
                'solution_code': ['solution', 'code', 'answer'],
                'language': ['language', 'lang'],
                'difficulty': ['difficulty', 'level', 'rating'],
                'tags': ['tags', 'categories'],
                'problem_id': ['id', 'problem_id', 'index'],
                'constraints': ['constraints', 'requirements'],
                'examples': ['examples', 'sample_inputs'],
                'test_cases': ['test_cases', 'tests']
            },
            'mbpp': {
                'question_text': ['text', 'description', 'prompt'],
                'solution_code': ['code', 'solution', 'implementation'],
                'language': ['language', 'lang'],
                'question_type': ['type', 'category'],
                'difficulty': ['difficulty', 'level'],
                'problem_id': ['task_id', 'id'],
                'test_cases': ['test_list', 'tests']
            }
        }
        
        # Question type detection patterns
        self.question_patterns = {
            'mcq': r'\b(multiple choice|select|choose|option|a\)|b\)|c\)|d\))\b',
            'coding': r'\b(implement|write|function|class|algorithm|code|program)\b',
            'fill_blank': r'\b(fill|blank|_____|___|__)\b',
            'debug': r'\b(debug|fix|error|bug|correct)\b',
            'optimize': r'\b(optimize|efficient|performance|time complexity)\b'
        }
        
        self.consolidated_data = []
        self.inventory = []
    
    def discover_files(self) -> List[Dict]:
        """Recursively scan all sub-folders and build inventory"""
        logger.info("Starting file discovery...")
        
        inventory = []
        total_size = 0
        
        for file_path in tqdm(list(self.root_dir.rglob('*')), desc="Discovering files"):
            if file_path.is_file():
                try:
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    
                    # Detect file type
                    detected_format = self._detect_file_format(file_path)
                    
                    # Estimate rows (rough approximation)
                    estimated_rows = self._estimate_rows(file_path, detected_format)
                    
                    inventory.append({
                        'path': str(file_path),
                        'detected_format': detected_format,
                        'size': file_size,
                        'estimated_rows': estimated_rows,
                        'relative_path': str(file_path.relative_to(self.root_dir))
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
        
        self.inventory = inventory
        logger.info(f"Discovered {len(inventory)} files, total size: {total_size / (1024**3):.2f} GB")
        
        # Save inventory
        with open(self.output_dir / 'file_inventory.json', 'w') as f:
            json.dump(inventory, f, indent=2)
        
        return inventory
    
    def _detect_file_format(self, file_path: Path) -> str:
        """Detect file format using extension and MIME sniffing"""
        suffix = file_path.suffix.lower()
        
        if suffix in self.file_extensions:
            return suffix[1:]  # Remove dot
        
        # Try MIME detection for unknown extensions
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            if 'json' in mime_type:
                return 'json'
            elif 'xml' in mime_type:
                return 'xml'
            elif 'text' in mime_type:
                return 'txt'
        
        return 'unknown'
    
    def _estimate_rows(self, file_path: Path, format_type: str) -> int:
        """Estimate number of rows in a file"""
        try:
            if format_type == 'csv':
                return sum(1 for _ in open(file_path, 'r', encoding='utf-8', errors='ignore'))
            elif format_type == 'jsonl':
                return sum(1 for _ in open(file_path, 'r', encoding='utf-8', errors='ignore'))
            elif format_type == 'parquet':
                df = pd.read_parquet(file_path)
                return len(df)
            elif format_type == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return len(data)
                    else:
                        return 1
            else:
                return 1
        except:
            return 1 
    
    def consolidate_datasets(self, max_workers: int = 4) -> List[UnifiedSchema]:
        """Main consolidation pipeline"""
        logger.info("Starting dataset consolidation...")
        
        # Discover files
        inventory = self.discover_files()
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for file_info in inventory:
                if '.' + file_info['detected_format'] in self.file_extensions:
                    future = executor.submit(self._process_file, file_info)
                    futures.append(future)
            
            # Collect results
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                try:
                    result = future.result()
                    if result:
                        self.consolidated_data.extend(result)
                except Exception as e:
                    logger.error(f"Error in file processing: {e}")
        
        logger.info(f"Consolidated {len(self.consolidated_data)} entries")
        return self.consolidated_data
    
    def _process_file(self, file_info: Dict) -> List[UnifiedSchema]:
        """Process a single file and return unified schema entries"""
        file_path = Path(file_info['path'])
        format_type = file_info['detected_format']
        
        try:
            parser_func = self.file_extensions.get('.' + format_type)
            if parser_func:
                raw_data = parser_func(file_path)
                if raw_data:
                    return self._normalize_data(raw_data, file_info)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
        
        return []
    
    def _parse_csv(self, file_path: Path) -> List[Dict]:
        """Parse CSV files"""
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                    return df.to_dict('records')
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            logger.error(f"Error parsing CSV {file_path}: {e}")
        return []
    
    def _parse_tsv(self, file_path: Path) -> List[Dict]:
        """Parse TSV files"""
        try:
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, sep='\t', encoding=encoding, low_memory=False)
                    return df.to_dict('records')
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            logger.error(f"Error parsing TSV {file_path}: {e}")
        return []
    
    def _parse_json(self, file_path: Path) -> List[Dict]:
        """Parse JSON files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    return [data]
        except Exception as e:
            logger.error(f"Error parsing JSON {file_path}: {e}")
        return []
    
    def _parse_jsonl(self, file_path: Path) -> List[Dict]:
        """Parse JSONL files"""
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            return data
        except Exception as e:
            logger.error(f"Error parsing JSONL {file_path}: {e}")
        return []
    
    def _parse_txt(self, file_path: Path) -> List[Dict]:
        """Parse text files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                return [{'content': content, 'file_path': str(file_path)}]
        except Exception as e:
            logger.error(f"Error parsing TXT {file_path}: {e}")
        return []
    
    def _parse_html(self, file_path: Path) -> List[Dict]:
        """Parse HTML files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                # Extract text content
                text_content = soup.get_text()
                return [{'content': text_content, 'file_path': str(file_path)}]
        except Exception as e:
            logger.error(f"Error parsing HTML {file_path}: {e}")
        return []
    
    def _parse_xml(self, file_path: Path) -> List[Dict]:
        """Parse XML files"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            def xml_to_dict(element):
                result = {}
                for child in element:
                    if len(child) == 0:
                        result[child.tag] = child.text
                    else:
                        result[child.tag] = xml_to_dict(child)
                return result
            
            return [xml_to_dict(root)]
        except Exception as e:
            logger.error(f"Error parsing XML {file_path}: {e}")
        return []
    
    def _parse_pickle(self, file_path: Path) -> List[Dict]:
        """Parse pickle files"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    return data
                else:
                    return [data]
        except Exception as e:
            logger.error(f"Error parsing pickle {file_path}: {e}")
        return []
    
    def _parse_parquet(self, file_path: Path) -> List[Dict]:
        """Parse parquet files"""
        try:
            df = pd.read_parquet(file_path)
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Error parsing parquet {file_path}: {e}")
        return []
    
    def _extract_archive(self, file_path: Path) -> List[Dict]:
        """Extract and process archive files"""
        extracted_data = []
        temp_dir = self.output_dir / 'temp_extract' / file_path.stem
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if file_path.suffix == '.zip':
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
            else:
                with tarfile.open(file_path, 'r:*') as tar_ref:
                    tar_ref.extractall(temp_dir)
            
            # Process extracted files
            for extracted_file in temp_dir.rglob('*'):
                if extracted_file.is_file():
                    format_type = self._detect_file_format(extracted_file)
                    if format_type in self.file_extensions:
                        parser_func = self.file_extensions[format_type]
                        if parser_func != self._extract_archive:  # Avoid recursion
                            data = parser_func(extracted_file)
                            extracted_data.extend(data)
        
        except Exception as e:
            logger.error(f"Error extracting archive {file_path}: {e}")
        finally:
            # Clean up
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        
        return extracted_data 
    
    def _normalize_data(self, raw_data: List[Dict], file_info: Dict) -> List[UnifiedSchema]:
        """Normalize raw data to unified schema"""
        normalized_entries = []
        
        # Detect source site from file path
        source_site = self._detect_source_site(file_info['path'])
        
        for item in raw_data:
            try:
                # Map fields based on source
                mapped_data = self._map_schema_fields(item, source_site)
                
                # Create unified schema entry
                unified_entry = UnifiedSchema(
                    question_text=mapped_data.get('question_text', ''),
                    answer_text=mapped_data.get('answer_text', ''),
                    solution_code=mapped_data.get('solution_code', ''),
                    language=mapped_data.get('language', ''),
                    question_type=mapped_data.get('question_type', ''),
                    source_site=source_site,
                    difficulty=mapped_data.get('difficulty', ''),
                    tags=mapped_data.get('tags', []),
                    problem_id=mapped_data.get('problem_id', ''),
                    constraints=mapped_data.get('constraints', ''),
                    examples=mapped_data.get('examples', []),
                    test_cases=mapped_data.get('test_cases', []),
                    time_limit=mapped_data.get('time_limit'),
                    memory_limit=mapped_data.get('memory_limit')
                )
                
                # Clean and validate
                if self._validate_entry(unified_entry):
                    normalized_entries.append(unified_entry)
            
            except Exception as e:
                logger.warning(f"Error normalizing item: {e}")
                continue
        
        return normalized_entries
    
    def _detect_source_site(self, file_path: str) -> str:
        """Detect the source site from file path"""
        file_path_lower = file_path.lower()
        
        if 'leetcode' in file_path_lower:
            return 'leetcode'
        elif 'codechef' in file_path_lower:
            return 'codechef'
        elif 'codeforces' in file_path_lower:
            return 'codeforces'
        elif 'hackerrank' in file_path_lower:
            return 'hackerrank'
        elif 'mbpp' in file_path_lower:
            return 'mbpp'
        elif 'apps' in file_path_lower:
            return 'apps'
        elif 'ds1000' in file_path_lower:
            return 'ds1000'
        elif 'bigcodebench' in file_path_lower:
            return 'bigcodebench'
        elif 'code_contests' in file_path_lower:
            return 'code_contests'
        else:
            return 'unknown'
    
    def _map_schema_fields(self, item: Dict, source_site: str) -> Dict:
        """Map raw data fields to unified schema"""
        mapped_data = {}
        
        # Get schema mapping for this source
        schema_mapping = self.schema_mappings.get(source_site, {})
        
        for unified_field, possible_fields in schema_mapping.items():
            for field in possible_fields:
                if field in item and item[field]:
                    if unified_field == 'solution_code':
                        # Handle multiple language solutions
                        if isinstance(item[field], dict):
                            # Combine all language solutions
                            solutions = []
                            for lang, code in item[field].items():
                                if code and isinstance(code, str):
                                    solutions.append(f"# {lang}\n{code}")
                            mapped_data[unified_field] = '\n\n'.join(solutions)
                        else:
                            mapped_data[unified_field] = str(item[field])
                    elif unified_field == 'tags':
                        # Handle tags as list
                        if isinstance(item[field], list):
                            mapped_data[unified_field] = item[field]
                        elif isinstance(item[field], str):
                            mapped_data[unified_field] = [tag.strip() for tag in item[field].split(',')]
                    else:
                        mapped_data[unified_field] = str(item[field])
                    break
        
        # Infer question type if not provided
        if not mapped_data.get('question_type'):
            mapped_data['question_type'] = self._infer_question_type(
                mapped_data.get('question_text', '')
            )
        
        # Handle special cases for different sources
        if source_site == 'leetcode':
            # LeetCode specific mapping
            if 'content' in item:
                mapped_data['question_text'] = item['content']
            if 'title' in item:
                mapped_data['problem_id'] = item['title']
            if 'difficulty' in item:
                mapped_data['difficulty'] = item['difficulty'].lower()
        
        elif source_site == 'mbpp':
            # MBPP specific mapping
            if 'text' in item:
                mapped_data['question_text'] = item['text']
            if 'code' in item:
                mapped_data['solution_code'] = item['code']
            if 'task_id' in item:
                mapped_data['problem_id'] = str(item['task_id'])
            if 'test_list' in item:
                mapped_data['test_cases'] = item['test_list']
        
        elif source_site == 'code_contests':
            # Code contests specific mapping
            if 'description' in item:
                mapped_data['question_text'] = item['description']
            if 'name' in item:
                mapped_data['problem_id'] = item['name']
            if 'solutions' in item and isinstance(item['solutions'], dict):
                solutions = []
                for lang_code, solution_data in item['solutions'].items():
                    if 'solution' in solution_data:
                        solutions.append(f"# Language {lang_code}\n{solution_data['solution']}")
                mapped_data['solution_code'] = '\n\n'.join(solutions)
        
        return mapped_data
    
    def _infer_question_type(self, question_text: str) -> str:
        """Infer question type from text content"""
        question_lower = question_text.lower()
        
        for qtype, pattern in self.question_patterns.items():
            if re.search(pattern, question_lower):
                return qtype
        
        # Default to coding if code-related keywords are found
        if any(keyword in question_lower for keyword in ['function', 'class', 'algorithm', 'implement']):
            return 'coding'
        
        return 'unknown'
    
    def _validate_entry(self, entry: UnifiedSchema) -> bool:
        """Validate if an entry meets quality criteria"""
        # Must have question text
        if not entry.question_text or len(entry.question_text.strip()) < 10:
            return False
        
        # Must have either answer text or solution code
        if not entry.answer_text and not entry.solution_code:
            return False
        
        # Remove entries that are too short or too long
        if len(entry.question_text) < 20 or len(entry.question_text) > 50000:
            return False
        
        return True 
    
    def deduplicate_data(self) -> List[UnifiedSchema]:
        """Remove duplicate entries based on question text similarity"""
        logger.info("Starting deduplication...")
        
        # Create hash-based deduplication
        seen_hashes = set()
        unique_entries = []
        
        for entry in tqdm(self.consolidated_data, desc="Deduplicating"):
            # Create hash of normalized question text
            normalized_text = self._normalize_text(entry.question_text)
            text_hash = hashlib.md5(normalized_text.encode()).hexdigest()
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_entries.append(entry)
        
        logger.info(f"Deduplication complete: {len(self.consolidated_data)} -> {len(unique_entries)} entries")
        self.consolidated_data = unique_entries
        return unique_entries
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for deduplication"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
    
    def quality_filter(self) -> List[UnifiedSchema]:
        """Apply quality filtering"""
        logger.info("Applying quality filters...")
        
        filtered_entries = []
        
        for entry in tqdm(self.consolidated_data, desc="Quality filtering"):
            # Skip entries with excessive chatter
            if self._has_excessive_chatter(entry.question_text):
                continue
            
            # Keep only primary solutions
            if entry.solution_code:
                entry.solution_code = self._extract_primary_solution(entry.solution_code)
            
            filtered_entries.append(entry)
        
        logger.info(f"Quality filtering complete: {len(self.consolidated_data)} -> {len(filtered_entries)} entries")
        self.consolidated_data = filtered_entries
        return filtered_entries
    
    def _has_excessive_chatter(self, text: str) -> bool:
        """Check if text has excessive chatter"""
        # Check for repeated patterns
        words = text.split()
        if len(words) < 10:
            return False
        
        # Check for excessive repetition
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # If any word appears more than 20% of the time, it's likely chatter
        max_freq = max(word_counts.values()) if word_counts else 0
        if max_freq > len(words) * 0.2:
            return True
        
        return False
    
    def _extract_primary_solution(self, solution_code: str) -> str:
        """Extract the primary solution from multiple solutions"""
        if not solution_code:
            return ""
        
        # Split by language markers
        solutions = re.split(r'#\s*\w+', solution_code)
        
        if len(solutions) > 1:
            # Return the longest solution (likely the most complete)
            return max(solutions, key=len).strip()
        
        return solution_code.strip()
    
    def create_language_variants(self) -> List[UnifiedSchema]:
        """Create additional language variants for coding problems"""
        logger.info("Creating language variants...")
        
        variants = []
        
        for entry in tqdm(self.consolidated_data, desc="Creating variants"):
            if entry.question_type == 'coding' and entry.solution_code:
                # Create variants for different languages
                for target_lang in ['python', 'java', 'cpp', 'javascript']:
                    if target_lang not in entry.language.lower():
                        variant = UnifiedSchema(
                            question_text=entry.question_text,
                            answer_text=entry.answer_text,
                            solution_code=entry.solution_code,  # Keep original for now
                            language=target_lang,
                            question_type=entry.question_type,
                            source_site=entry.source_site,
                            difficulty=entry.difficulty,
                            tags=entry.tags.copy(),
                            problem_id=f"{entry.problem_id}_{target_lang}",
                            constraints=entry.constraints,
                            examples=entry.examples.copy() if entry.examples else [],
                            test_cases=entry.test_cases.copy() if entry.test_cases else []
                        )
                        variants.append(variant)
        
        self.consolidated_data.extend(variants)
        logger.info(f"Created {len(variants)} language variants")
        return self.consolidated_data
    
    def create_train_val_test_splits(self, train_ratio: float = 0.9, val_ratio: float = 0.05) -> Dict[str, List[UnifiedSchema]]:
        """Create train/validation/test splits"""
        logger.info("Creating train/validation/test splits...")
        
        # Stratify by source_site and difficulty
        splits = {
            'train': [],
            'validation': [],
            'test': []
        }
        
        # Group by source and difficulty
        grouped_data = {}
        for entry in self.consolidated_data:
            key = (entry.source_site, entry.difficulty)
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append(entry)
        
        # Create splits for each group
        for group_key, group_entries in grouped_data.items():
            np.random.shuffle(group_entries)
            
            n_total = len(group_entries)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            splits['train'].extend(group_entries[:n_train])
            splits['validation'].extend(group_entries[n_train:n_train + n_val])
            splits['test'].extend(group_entries[n_train + n_val:])
        
        # Shuffle each split
        for split_name in splits:
            np.random.shuffle(splits[split_name])
        
        logger.info(f"Split sizes - Train: {len(splits['train'])}, Val: {len(splits['validation'])}, Test: {len(splits['test'])}")
        
        return splits
    
    def save_consolidated_data(self, splits: Dict[str, List[UnifiedSchema]]) -> None:
        """Save consolidated data to files"""
        logger.info("Saving consolidated data...")
        
        # Save as JSONL
        for split_name, entries in splits.items():
            output_file = self.output_dir / f'{split_name}.jsonl'
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for entry in entries:
                    f.write(json.dumps(asdict(entry), ensure_ascii=False) + '\n')
            
            logger.info(f"Saved {len(entries)} entries to {output_file}")
        
        # Save manifest
        manifest = {
            'total_entries': sum(len(entries) for entries in splits.values()),
            'split_sizes': {name: len(entries) for name, entries in splits.items()},
            'source_distribution': {},
            'difficulty_distribution': {},
            'question_type_distribution': {}
        }
        
        # Calculate distributions
        for split_name, entries in splits.items():
            for entry in entries:
                manifest['source_distribution'][entry.source_site] = manifest['source_distribution'].get(entry.source_site, 0) + 1
                manifest['difficulty_distribution'][entry.difficulty] = manifest['difficulty_distribution'].get(entry.difficulty, 0) + 1
                manifest['question_type_distribution'][entry.question_type] = manifest['question_type_distribution'].get(entry.question_type, 0) + 1
        
        with open(self.output_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Saved manifest to {self.output_dir / 'manifest.json'}")
    
    def run_pipeline(self, max_workers: int = 4, create_variants: bool = True) -> Dict[str, List[UnifiedSchema]]:
        """Run the complete consolidation pipeline"""
        logger.info("Starting dataset consolidation pipeline...")
        
        # Step 1: Consolidate datasets
        self.consolidate_datasets(max_workers)
        
        # Step 2: Deduplicate
        self.deduplicate_data()
        
        # Step 3: Quality filter
        self.quality_filter()
        
        # Step 4: Create language variants (optional)
        if create_variants:
            self.create_language_variants()
        
        # Step 5: Create splits
        splits = self.create_train_val_test_splits()
        
        # Step 6: Save data
        self.save_consolidated_data(splits)
        
        logger.info("Dataset consolidation pipeline completed successfully!")
        return splits

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Dataset Consolidation Pipeline')
    parser.add_argument('--root_dir', type=str, default='.', help='Root directory containing datasets')
    parser.add_argument('--output_dir', type=str, default='consolidated_dataset', help='Output directory')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of worker threads')
    parser.add_argument('--no_variants', action='store_true', help='Skip creating language variants')
    
    args = parser.parse_args()
    
    # Initialize consolidator
    consolidator = DatasetConsolidator(args.root_dir, args.output_dir)
    
    # Run pipeline
    splits = consolidator.run_pipeline(
        max_workers=args.max_workers,
        create_variants=not args.no_variants
    )
    
    print(f"\nConsolidation complete!")
    print(f"Total entries: {sum(len(entries) for entries in splits.values())}")
    print(f"Train: {len(splits['train'])}")
    print(f"Validation: {len(splits['validation'])}")
    print(f"Test: {len(splits['test'])}")

if __name__ == "__main__":
    main() 