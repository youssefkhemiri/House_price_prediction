#!/usr/bin/env python3
"""
Test runner script for the Real Estate Scraper API
This script provides easy commands to run different types of tests
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and handle its output"""
    print(f"\n🔄 {description}")
    print("=" * 60)
    
    result = subprocess.run(command, shell=True, capture_output=False)
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully")
    else:
        print(f"❌ {description} failed with exit code {result.returncode}")
    
    return result.returncode == 0

def install_test_dependencies():
    """Install test dependencies"""
    return run_command(
        "pip install -r requirements-test.txt",
        "Installing test dependencies"
    )

def run_unit_tests():
    """Run unit tests only"""
    return run_command(
        'pytest tests/ -m "not integration and not slow" -v',
        "Running unit tests"
    )

def run_integration_tests():
    """Run integration tests"""
    return run_command(
        'pytest tests/ -m "integration" -v',
        "Running integration tests"
    )

def run_all_tests():
    """Run all tests"""
    return run_command(
        "pytest tests/ -v",
        "Running all tests"
    )

def run_tests_with_coverage():
    """Run tests with coverage report"""
    return run_command(
        "pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v",
        "Running tests with coverage"
    )

def run_api_tests_only():
    """Run API tests only"""
    return run_command(
        "pytest tests/test_api.py -v",
        "Running API tests"
    )

def run_scraper_tests_only():
    """Run scraper tests only"""
    return run_command(
        "pytest tests/test_scrapers.py -v",
        "Running scraper tests"
    )

def run_prediction_tests_only():
    """Run prediction tests only"""
    return run_command(
        "pytest tests/test_prediction.py -v",
        "Running prediction tests"
    )

def run_fast_tests():
    """Run fast tests only (exclude slow tests)"""
    return run_command(
        'pytest tests/ -m "not slow" -v',
        "Running fast tests"
    )

def generate_test_report():
    """Generate HTML test report"""
    return run_command(
        "pytest tests/ --html=test_report.html --self-contained-html -v",
        "Generating HTML test report"
    )

def check_code_quality():
    """Run code quality checks"""
    success = True
    
    # Run flake8
    if not run_command("flake8 src/ tests/", "Running flake8 linting"):
        success = False
    
    # Run black check
    if not run_command("black --check src/ tests/", "Checking code formatting"):
        success = False
    
    # Run isort check
    if not run_command("isort --check-only src/ tests/", "Checking import sorting"):
        success = False
    
    return success

def fix_code_formatting():
    """Fix code formatting issues"""
    success = True
    
    if not run_command("black src/ tests/", "Fixing code formatting with black"):
        success = False
    
    if not run_command("isort src/ tests/", "Fixing import sorting with isort"):
        success = False
    
    return success

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Test runner for Real Estate Scraper API")
    parser.add_argument(
        "command",
        choices=[
            "install",
            "unit",
            "integration", 
            "all",
            "coverage",
            "api",
            "scrapers",
            "prediction",
            "fast",
            "report",
            "quality",
            "format"
        ],
        help="Test command to run"
    )
    
    args = parser.parse_args()
    
    print("🏠 Real Estate Scraper API - Test Runner")
    print("=" * 50)
    
    # Change to project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    success = True
    
    if args.command == "install":
        success = install_test_dependencies()
    elif args.command == "unit":
        success = run_unit_tests()
    elif args.command == "integration":
        success = run_integration_tests()
    elif args.command == "all":
        success = run_all_tests()
    elif args.command == "coverage":
        success = run_tests_with_coverage()
    elif args.command == "api":
        success = run_api_tests_only()
    elif args.command == "scrapers":
        success = run_scraper_tests_only()
    elif args.command == "prediction":
        success = run_prediction_tests_only()
    elif args.command == "fast":
        success = run_fast_tests()
    elif args.command == "report":
        success = generate_test_report()
    elif args.command == "quality":
        success = check_code_quality()
    elif args.command == "format":
        success = fix_code_formatting()
    
    if success:
        print(f"\n🎉 Command '{args.command}' completed successfully!")
        sys.exit(0)
    else:
        print(f"\n💥 Command '{args.command}' failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
