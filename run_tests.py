#!/usr/bin/env python3
import unittest
import os
import sys

def run_tests():
    """Run all token analysis tests"""
    # Create test data directory if it doesn't exist
    if not os.path.exists('test_data'):
        os.makedirs('test_data')
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
