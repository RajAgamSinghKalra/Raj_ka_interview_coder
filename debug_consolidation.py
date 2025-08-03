#!/usr/bin/env python3
"""
Debug script for dataset consolidation pipeline
"""

import json
import tempfile
from pathlib import Path
from dataset_consolidation import DatasetConsolidator

def create_test_data():
    """Create test data files"""
    test_dir = Path("debug_test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create test CSV file
    csv_content = """Question,Answer
"Write a function to find the minimum cost path","def min_cost_path(cost_matrix):
    # Implementation here
    pass"
"Implement binary search","def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1"
"""
    
    with open(test_dir / "test.csv", "w") as f:
        f.write(csv_content)
    
    # Create test JSONL file (MBPP format)
    jsonl_content = [
        {
            "text": "Write a function to reverse a string",
            "code": "def reverse_string(s):\n    return s[::-1]",
            "task_id": 1,
            "test_list": ["assert reverse_string('hello') == 'olleh'"]
        },
        {
            "text": "Find the maximum element in an array",
            "code": "def find_max(arr):\n    return max(arr) if arr else None",
            "task_id": 2,
            "test_list": ["assert find_max([1, 5, 3, 9, 2]) == 9"]
        }
    ]
    
    with open(test_dir / "test.jsonl", "w") as f:
        for item in jsonl_content:
            f.write(json.dumps(item) + "\n")
    
    return test_dir

def debug_consolidation():
    """Debug the consolidation pipeline step by step"""
    print("Creating test data...")
    test_dir = create_test_data()
    
    print("Initializing consolidator...")
    with tempfile.TemporaryDirectory() as temp_output:
        consolidator = DatasetConsolidator(str(test_dir), temp_output)
        
        # Step 1: File discovery
        print("\n=== Step 1: File Discovery ===")
        inventory = consolidator.discover_files()
        print(f"Discovered {len(inventory)} files:")
        for file_info in inventory:
            print(f"  - {file_info['path']} ({file_info['detected_format']})")
        
        # Step 2: Process files manually
        print("\n=== Step 2: File Processing ===")
        for file_info in inventory:
            print(f"\nProcessing: {file_info['path']}")
            try:
                # Test raw parsing first
                file_path = Path(file_info['path'])
                format_type = file_info['detected_format']
                
                print(f"  Format: {format_type}")
                print(f"  Available parsers: {list(consolidator.file_extensions.keys())}")
                parser_func = consolidator.file_extensions.get('.' + format_type)
                print(f"  Parser found: {parser_func is not None}")
                
                if parser_func:
                    print(f"  Parsing raw data...")
                    raw_data = parser_func(file_path)
                    print(f"  Raw data: {len(raw_data)} items")
                    if raw_data:
                        print(f"  Sample raw item: {raw_data[0]}")
                        
                        # Test normalization
                        print(f"  Normalizing data...")
                        normalized = consolidator._normalize_data(raw_data, file_info)
                        print(f"  Normalized: {len(normalized)} entries")
                        
                        if normalized:
                            print(f"  Sample normalized: {normalized[0]}")
                        else:
                            print(f"  No normalized entries - checking validation...")
                            # Test validation on a sample
                            if raw_data:
                                print(f"  Raw item keys: {list(raw_data[0].keys())}")
                                mapped_data = consolidator._map_schema_fields(raw_data[0], 'unknown')
                                print(f"  Mapped data: {mapped_data}")
                                
                                from dataset_consolidation import UnifiedSchema
                                unified_entry = UnifiedSchema(
                                    question_text=mapped_data.get('question_text', ''),
                                    answer_text=mapped_data.get('answer_text', ''),
                                    solution_code=mapped_data.get('solution_code', ''),
                                    language=mapped_data.get('language', ''),
                                    question_type=mapped_data.get('question_type', ''),
                                    source_site='unknown',
                                    difficulty=mapped_data.get('difficulty', ''),
                                    tags=mapped_data.get('tags', []),
                                    problem_id=mapped_data.get('problem_id', ''),
                                    constraints=mapped_data.get('constraints', ''),
                                    examples=mapped_data.get('examples', []),
                                    test_cases=mapped_data.get('test_cases', []),
                                    time_limit=mapped_data.get('time_limit'),
                                    memory_limit=mapped_data.get('memory_limit')
                                )
                                
                                is_valid = consolidator._validate_entry(unified_entry)
                                print(f"  Is valid: {is_valid}")
                                print(f"  Question text length: {len(unified_entry.question_text)}")
                                print(f"  Has answer: {bool(unified_entry.answer_text)}")
                                print(f"  Has solution: {bool(unified_entry.solution_code)}")
                
                result = consolidator._process_file(file_info)
                print(f"  Final result: {len(result)} entries")
                
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
        
        # Step 3: Check consolidated data
        print(f"\n=== Step 3: Consolidated Data ===")
        print(f"Total entries: {len(consolidator.consolidated_data)}")
        
        if consolidator.consolidated_data:
            sample = consolidator.consolidated_data[0]
            print(f"Sample entry:")
            print(f"  Question: {sample.question_text[:100]}...")
            print(f"  Source: {sample.source_site}")
            print(f"  Type: {sample.question_type}")
            print(f"  Language: {sample.language}")
            print(f"  Solution: {sample.solution_code[:100]}...")

if __name__ == "__main__":
    debug_consolidation() 