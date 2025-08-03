#!/usr/bin/env python3
"""
Dataset Verification Script
Comprehensive verification that all datasets are properly processed

This script verifies that every entry from every individual dataset is present
in the consolidated output, with detailed reporting and statistics.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetVerifier:
    """Comprehensive dataset verification system"""
    
    def __init__(self, root_dir: str = ".", consolidated_dir: str = "consolidated_output"):
        self.root_dir = Path(root_dir)
        self.consolidated_dir = Path(consolidated_dir)
        self.verification_results = {}
        self.dataset_stats = {}
        
    def discover_original_datasets(self) -> Dict[str, List[Path]]:
        """Discover all original dataset files"""
        logger.info("Discovering original datasets...")
        
        dataset_files = defaultdict(list)
        
        # Define dataset patterns
        dataset_patterns = {
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
        
        for dataset_name, patterns in dataset_patterns.items():
            for pattern in patterns:
                files = list(self.root_dir.glob(pattern))
                if files:
                    dataset_files[dataset_name].extend(files)
                    logger.info(f"Found {len(files)} files for {dataset_name}")
        
        # Remove duplicates
        for dataset_name in dataset_files:
            dataset_files[dataset_name] = list(set(dataset_files[dataset_name]))
        
        total_files = sum(len(files) for files in dataset_files.values())
        logger.info(f"Total original dataset files found: {total_files}")
        
        return dataset_files
    
    def count_entries_in_file(self, file_path: Path) -> int:
        """Count entries in a single file"""
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
                return len(df)
            elif file_path.suffix.lower() == '.jsonl':
                count = 0
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            count += 1
                return count
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return len(data)
                elif isinstance(data, dict):
                    return 1
                else:
                    return 0
            elif file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
                return len(df)
            else:
                # Try to read as text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                return len([line for line in lines if line.strip()])
        except Exception as e:
            logger.warning(f"Error counting entries in {file_path}: {e}")
            return 0
    
    def analyze_original_datasets(self, dataset_files: Dict[str, List[Path]]) -> Dict:
        """Analyze original datasets and count entries"""
        logger.info("Analyzing original datasets...")
        
        analysis = {}
        total_entries = 0
        
        for dataset_name, files in dataset_files.items():
            dataset_entries = 0
            file_details = []
            
            for file_path in tqdm(files, desc=f"Analyzing {dataset_name}"):
                entry_count = self.count_entries_in_file(file_path)
                file_details.append({
                    'file': str(file_path),
                    'entries': entry_count,
                    'size_mb': file_path.stat().st_size / (1024 * 1024)
                })
                dataset_entries += entry_count
            
            analysis[dataset_name] = {
                'files': len(files),
                'total_entries': dataset_entries,
                'file_details': file_details
            }
            total_entries += dataset_entries
            
            logger.info(f"{dataset_name}: {dataset_entries} entries in {len(files)} files")
        
        analysis['total'] = {
            'datasets': len(dataset_files),
            'files': sum(len(files) for files in dataset_files.values()),
            'entries': total_entries
        }
        
        return analysis
    
    def analyze_consolidated_data(self) -> Dict:
        """Analyze the consolidated dataset"""
        logger.info("Analyzing consolidated dataset...")
        
        if not self.consolidated_dir.exists():
            raise FileNotFoundError(f"Consolidated directory {self.consolidated_dir} not found")
        
        analysis = {}
        
        # Analyze each split
        for split in ['train', 'validation', 'test']:
            file_path = self.consolidated_dir / f"{split}.jsonl"
            if file_path.exists():
                entries = 0
                source_sites = Counter()
                difficulties = Counter()
                languages = Counter()
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                entry = json.loads(line.strip())
                                entries += 1
                                source_sites[entry.get('source_site', 'unknown')] += 1
                                difficulties[entry.get('difficulty', 'unknown')] += 1
                                languages[entry.get('language', 'unknown')] += 1
                            except json.JSONDecodeError:
                                continue
                
                analysis[split] = {
                    'entries': entries,
                    'source_sites': dict(source_sites),
                    'difficulties': dict(difficulties),
                    'languages': dict(languages)
                }
                
                logger.info(f"{split}: {entries} entries")
            else:
                analysis[split] = {'entries': 0, 'source_sites': {}, 'difficulties': {}, 'languages': {}}
        
        # Analyze manifest
        manifest_path = self.consolidated_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            analysis['manifest'] = manifest
        
        return analysis
    
    def verify_coverage(self, original_analysis: Dict, consolidated_analysis: Dict) -> Dict:
        """Verify that all original entries are covered in consolidated data"""
        logger.info("Verifying coverage...")
        
        verification = {
            'coverage_analysis': {},
            'missing_entries': {},
            'quality_metrics': {}
        }
        
        # Calculate total entries in original vs consolidated
        original_total = original_analysis['total']['entries']
        consolidated_total = sum(
            consolidated_analysis[split]['entries'] 
            for split in ['train', 'validation', 'test']
        )
        
        coverage_percentage = (consolidated_total / original_total * 100) if original_total > 0 else 0
        
        verification['coverage_analysis'] = {
            'original_total': original_total,
            'consolidated_total': consolidated_total,
            'coverage_percentage': coverage_percentage,
            'missing_entries': original_total - consolidated_total
        }
        
        # Analyze by source site
        for dataset_name, dataset_info in original_analysis.items():
            if dataset_name == 'total':
                continue
            
            original_count = dataset_info['total_entries']
            consolidated_count = 0
            
            # Count entries from this source in consolidated data
            for split in ['train', 'validation', 'test']:
                if split in consolidated_analysis:
                    source_sites = consolidated_analysis[split].get('source_sites', {})
                    # Map dataset names to source sites
                    source_mapping = {
                        'leetcode': 'leetcode',
                        'mbpp': 'mbpp',
                        'code_contests': 'code_contests',
                        'bigcodebench': 'bigcodebench',
                        'apps': 'apps',
                        'ds1000': 'ds1000',
                        'codesearchnet': 'codesearchnet',
                        'codechef': 'codechef',
                        'codeforces': 'codeforces',
                        'hackerearth': 'hackerearth'
                    }
                    
                    mapped_source = source_mapping.get(dataset_name, dataset_name)
                    consolidated_count += source_sites.get(mapped_source, 0)
            
            dataset_coverage = (consolidated_count / original_count * 100) if original_count > 0 else 0
            
            verification['coverage_analysis'][dataset_name] = {
                'original': original_count,
                'consolidated': consolidated_count,
                'coverage_percentage': dataset_coverage,
                'missing': original_count - consolidated_count
            }
        
        return verification
    
    def generate_quality_report(self, consolidated_analysis: Dict) -> Dict:
        """Generate quality metrics for the consolidated dataset"""
        logger.info("Generating quality report...")
        
        quality_metrics = {
            'split_distribution': {},
            'source_diversity': {},
            'difficulty_distribution': {},
            'language_distribution': {},
            'data_quality': {}
        }
        
        # Split distribution
        total_entries = sum(
            consolidated_analysis[split]['entries'] 
            for split in ['train', 'validation', 'test']
        )
        
        for split in ['train', 'validation', 'test']:
            entries = consolidated_analysis[split]['entries']
            percentage = (entries / total_entries * 100) if total_entries > 0 else 0
            quality_metrics['split_distribution'][split] = {
                'entries': entries,
                'percentage': percentage
            }
        
        # Source diversity
        all_sources = set()
        for split in ['train', 'validation', 'test']:
            all_sources.update(consolidated_analysis[split].get('source_sites', {}).keys())
        
        quality_metrics['source_diversity'] = {
            'unique_sources': len(all_sources),
            'sources': list(all_sources)
        }
        
        # Difficulty distribution
        all_difficulties = Counter()
        for split in ['train', 'validation', 'test']:
            all_difficulties.update(consolidated_analysis[split].get('difficulties', {}))
        
        quality_metrics['difficulty_distribution'] = dict(all_difficulties)
        
        # Language distribution
        all_languages = Counter()
        for split in ['train', 'validation', 'test']:
            all_languages.update(consolidated_analysis[split].get('languages', {}))
        
        quality_metrics['language_distribution'] = dict(all_languages)
        
        return quality_metrics
    
    def run_full_verification(self) -> Dict:
        """Run the complete verification process"""
        logger.info("Starting full dataset verification...")
        
        # Step 1: Discover original datasets
        dataset_files = self.discover_original_datasets()
        
        # Step 2: Analyze original datasets
        original_analysis = self.analyze_original_datasets(dataset_files)
        
        # Step 3: Analyze consolidated data
        consolidated_analysis = self.analyze_consolidated_data()
        
        # Step 4: Verify coverage
        coverage_verification = self.verify_coverage(original_analysis, consolidated_analysis)
        
        # Step 5: Generate quality report
        quality_report = self.generate_quality_report(consolidated_analysis)
        
        # Compile final results
        results = {
            'original_datasets': original_analysis,
            'consolidated_data': consolidated_analysis,
            'coverage_verification': coverage_verification,
            'quality_report': quality_report,
            'summary': self._generate_summary(original_analysis, consolidated_analysis, coverage_verification)
        }
        
        return results
    
    def _generate_summary(self, original_analysis: Dict, consolidated_analysis: Dict, coverage_verification: Dict) -> Dict:
        """Generate a summary of the verification results"""
        original_total = original_analysis['total']['entries']
        consolidated_total = sum(
            consolidated_analysis[split]['entries'] 
            for split in ['train', 'validation', 'test']
        )
        
        return {
            'total_original_entries': original_total,
            'total_consolidated_entries': consolidated_total,
            'overall_coverage_percentage': coverage_verification['coverage_analysis']['coverage_percentage'],
            'datasets_processed': original_analysis['total']['datasets'],
            'files_processed': original_analysis['total']['files'],
            'consolidation_success': consolidated_total > 0,
            'quality_score': self._calculate_quality_score(consolidated_analysis)
        }
    
    def _calculate_quality_score(self, consolidated_analysis: Dict) -> float:
        """Calculate a quality score for the consolidated dataset"""
        # Simple quality scoring based on various metrics
        score = 0.0
        
        # Check if all splits have data
        splits_with_data = sum(
            1 for split in ['train', 'validation', 'test']
            if consolidated_analysis[split]['entries'] > 0
        )
        score += (splits_with_data / 3) * 30  # 30 points for proper splits
        
        # Check source diversity
        all_sources = set()
        for split in ['train', 'validation', 'test']:
            all_sources.update(consolidated_analysis[split].get('source_sites', {}).keys())
        
        source_diversity = min(len(all_sources) / 10, 1.0)  # Normalize to 10 sources
        score += source_diversity * 30  # 30 points for source diversity
        
        # Check total volume
        total_entries = sum(
            consolidated_analysis[split]['entries'] 
            for split in ['train', 'validation', 'test']
        )
        volume_score = min(total_entries / 5000, 1.0)  # Normalize to 5000 entries
        score += volume_score * 40  # 40 points for volume
        
        return min(score, 100.0)
    
    def save_verification_report(self, results: Dict, output_file: str = "verification_report.json"):
        """Save the verification report to a file"""
        logger.info(f"Saving verification report to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info("Verification report saved successfully!")
    
    def print_summary(self, results: Dict):
        """Print a human-readable summary of the verification results"""
        summary = results['summary']
        
        print("\n" + "="*80)
        print("DATASET VERIFICATION SUMMARY")
        print("="*80)
        
        print(f"Total Original Entries: {summary['total_original_entries']:,}")
        print(f"Total Consolidated Entries: {summary['total_consolidated_entries']:,}")
        print(f"Overall Coverage: {summary['overall_coverage_percentage']:.1f}%")
        print(f"Datasets Processed: {summary['datasets_processed']}")
        print(f"Files Processed: {summary['files_processed']}")
        print(f"Consolidation Success: {'✓' if summary['consolidation_success'] else '✗'}")
        print(f"Quality Score: {summary['quality_score']:.1f}/100")
        
        print("\n" + "-"*80)
        print("DETAILED COVERAGE BY DATASET")
        print("-"*80)
        
        coverage = results['coverage_verification']['coverage_analysis']
        for dataset, info in coverage.items():
            if dataset not in ['original_total', 'consolidated_total', 'coverage_percentage', 'missing_entries']:
                print(f"{dataset:15} | {info['original']:8,} → {info['consolidated']:8,} | {info['coverage_percentage']:5.1f}% | Missing: {info['missing']:6,}")
        
        print("\n" + "-"*80)
        print("CONSOLIDATED DATA SPLITS")
        print("-"*80)
        
        consolidated = results['consolidated_data']
        for split in ['train', 'validation', 'test']:
            entries = consolidated[split]['entries']
            print(f"{split:12} | {entries:8,} entries")
        
        print("="*80)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Verify dataset consolidation")
    parser.add_argument("--root_dir", type=str, default=".", help="Root directory containing datasets")
    parser.add_argument("--consolidated_dir", type=str, default="consolidated_output", help="Directory with consolidated data")
    parser.add_argument("--output", type=str, default="verification_report.json", help="Output report file")
    parser.add_argument("--no_print", action="store_true", help="Don't print summary")
    
    args = parser.parse_args()
    
    # Create verifier
    verifier = DatasetVerifier(args.root_dir, args.consolidated_dir)
    
    # Run verification
    results = verifier.run_full_verification()
    
    # Save report
    verifier.save_verification_report(results, args.output)
    
    # Print summary
    if not args.no_print:
        verifier.print_summary(results)
    
    # Return success/failure
    summary = results['summary']
    if summary['consolidation_success'] and summary['overall_coverage_percentage'] > 50:
        logger.info("✓ Dataset verification completed successfully!")
        return 0
    else:
        logger.error("✗ Dataset verification failed or coverage is too low!")
        return 1

if __name__ == "__main__":
    exit(main()) 