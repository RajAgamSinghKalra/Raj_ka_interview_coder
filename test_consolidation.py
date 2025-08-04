#!/usr/bin/env python3
"""
Test script for dataset consolidation pipeline
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from dataset_consolidation import DatasetConsolidator

def create_test_data():
    """Create test data files to verify the consolidation pipeline"""
    test_dir = Path("test_data")
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
    
    # Create test JSONL file
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

def test_consolidation():
    """Test the consolidation pipeline"""
    print("Creating test data...")
    test_dir = create_test_data()
    
    print("Running consolidation pipeline...")
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_output:
        consolidator = DatasetConsolidator(str(test_dir), temp_output)
        
        # Run pipeline with minimal processing
        splits = consolidator.run_pipeline(
            max_workers=1,
            create_variants=False,
            shard_size=1,
            output_formats=("jsonl", "arrow"),
        )
        
        print(f"\nTest Results:")
        print(f"Total entries: {sum(len(entries) for entries in splits.values())}")
        print(f"Train: {len(splits['train'])}")
        print(f"Validation: {len(splits['validation'])}")
        print(f"Test: {len(splits['test'])}")
        
        # Print sample entries
        if splits['train']:
            print(f"\nSample train entry:")
            sample = splits['train'][0]
            print(f"Question: {sample.question_text[:100]}...")
            print(f"Source: {sample.source_site}")
            print(f"Type: {sample.question_type}")
            print(f"Language: {sample.language}")
        
        # Check output files
        output_files = list(Path(temp_output).glob("*"))
        print(f"\nOutput files created: {[f.name for f in output_files]}")
        
        # Clean up test data
        shutil.rmtree(test_dir)
        
        return len(splits['train']) > 0

if __name__ == "__main__":
    success = test_consolidation()
    if success:
        print("\n✅ Test passed! Consolidation pipeline is working correctly.")
    else:
        print("\n❌ Test failed! Check the logs for errors.") 