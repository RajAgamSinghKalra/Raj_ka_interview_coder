#!/usr/bin/env python3
"""
Quick analysis of the verification report to understand consolidation failure
"""

import json
import sys

def analyze_verification_report():
    """Analyze the verification report to understand the issue"""
    
    try:
        with open('verification_report.json', 'r') as f:
            data = json.load(f)
        
        print("="*80)
        print("VERIFICATION REPORT ANALYSIS")
        print("="*80)
        
        # Summary
        summary = data.get('summary', {})
        print(f"\nSUMMARY:")
        print(f"Total Original Entries: {summary.get('total_original_entries', 0):,}")
        print(f"Total Consolidated Entries: {summary.get('total_consolidated_entries', 0):,}")
        print(f"Coverage Percentage: {summary.get('overall_coverage_percentage', 0):.4f}%")
        print(f"Consolidation Success: {summary.get('consolidation_success', False)}")
        print(f"Quality Score: {summary.get('quality_score', 0):.1f}/100")
        
        # Coverage by dataset
        print(f"\nCOVERAGE BY DATASET:")
        coverage = data.get('coverage_verification', {}).get('coverage_analysis', {})
        
        for dataset, info in coverage.items():
            if dataset not in ['original_total', 'consolidated_total', 'coverage_percentage', 'missing_entries']:
                original = info.get('original', 0)
                consolidated = info.get('consolidated', 0)
                coverage_pct = info.get('coverage_percentage', 0)
                missing = info.get('missing', 0)
                
                print(f"{dataset:15} | {original:12,} → {consolidated:8,} | {coverage_pct:6.2f}% | Missing: {missing:10,}")
        
        # Original dataset analysis
        print(f"\nORIGINAL DATASET ANALYSIS:")
        original = data.get('original_datasets', {})
        
        for dataset, info in original.items():
            if dataset != 'total':
                files = info.get('files', 0)
                entries = info.get('total_entries', 0)
                print(f"{dataset:15} | {files:6,} files | {entries:12,} entries")
        
        # Consolidated data analysis
        print(f"\nCONSOLIDATED DATA ANALYSIS:")
        consolidated = data.get('consolidated_data', {})
        
        for split in ['train', 'validation', 'test']:
            entries = consolidated.get(split, {}).get('entries', 0)
            print(f"{split:12} | {entries:8,} entries")
        
        # Source sites in consolidated data
        print(f"\nSOURCE SITES IN CONSOLIDATED DATA:")
        for split in ['train', 'validation', 'test']:
            source_sites = consolidated.get(split, {}).get('source_sites', {})
            if source_sites:
                print(f"{split}: {source_sites}")
        
    except Exception as e:
        print(f"Error analyzing report: {e}")
        return False
    
    return True

if __name__ == "__main__":
    analyze_verification_report() 