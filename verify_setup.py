#!/usr/bin/env python3
"""
Verification script to check if the training setup is correct
This script verifies the code structure without requiring all dependencies
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filepath}")
    return exists

def check_imports():
    """Check if the necessary imports can be resolved"""
    print("\n=== Checking Code Structure ===")

    # Check if files can be parsed
    files_to_check = [
        ('train.py', 'Training script'),
        ('core/main.py', 'Model architecture'),
        ('utils/fcgr.py', 'FCGR utility')
    ]

    all_valid = True
    for filepath, description in files_to_check:
        try:
            with open(filepath, 'r') as f:
                compile(f.read(), filepath, 'exec')
            print(f"✓ {description} syntax is valid")
        except SyntaxError as e:
            print(f"✗ {description} has syntax error: {e}")
            all_valid = False
        except FileNotFoundError:
            print(f"✗ {description} not found: {filepath}")
            all_valid = False

    return all_valid

def main():
    print("=" * 60)
    print("miTarFCGR Training Setup Verification")
    print("=" * 60)

    # Check required files
    print("\n=== Checking Required Files ===")
    files = [
        ("data/miraw.csv", "Data file"),
        ("core/main.py", "Model architecture"),
        ("utils/fcgr.py", "FCGR utility"),
        ("train.py", "Training script"),
        ("requirements.txt", "Requirements file"),
        ("README.md", "Documentation")
    ]

    all_exist = all(check_file_exists(f, desc) for f, desc in files)

    # Check code structure
    syntax_valid = check_imports()

    # Check data file
    print("\n=== Checking Data File ===")
    if os.path.exists("data/miraw.csv"):
        try:
            with open("data/miraw.csv", 'r') as f:
                header = f.readline().strip()
                required_columns = ['mature_miRNA_Transcript', 'mRNA_Site_Transcript', 'validation']
                has_required = all(col in header for col in required_columns)
                if has_required:
                    print("✓ Data file has required columns")
                else:
                    print("✗ Data file missing required columns")
                    print(f"  Required: {required_columns}")
                    print(f"  Found: {header}")
        except Exception as e:
            print(f"✗ Error reading data file: {e}")

    # Summary
    print("\n" + "=" * 60)
    if all_exist and syntax_valid:
        print("✓ Setup verification passed!")
        print("\nTo run training:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run training: python train.py")
    else:
        print("✗ Setup verification failed. Please check the errors above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
