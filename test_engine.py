#!/usr/bin/env python3
"""
Test script for Cluster-then-Search Engine
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import ClusterThenSearchEngine, preprocess
from datasets import load_dataset, concatenate_datasets

def test_inspec_dataset():
    """Test with Inspec dataset"""
    print("🧪 Testing with Inspec dataset...")
    
    try:
        # Load dataset
        full_dataset_dict = load_dataset("taln-ls2n/inspec")
        dataset_to_eval = concatenate_datasets([
            full_dataset_dict['train'], 
            full_dataset_dict['validation'], 
            full_dataset_dict['test']
        ])
        
        # Initialize engine
        engine = ClusterThenSearchEngine(dataset_to_eval, n_clusters=5)
        
        # Test search
        query = "machine learning algorithms"
        print(f"\n🔍 Testing search: '{query}'")
        results = engine.search(query, top_n=3)
        
        print(f"\n📋 Results:")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc['title']}")
            print(f"   Score: {doc['score']:.4f}")
            print(f"   Cluster: {doc['cluster_id']}")
            print(f"   Abstract: {doc['abstract'][:100]}...")
            print()
        
        print("✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_preprocessing():
    """Test preprocessing function"""
    print("🧪 Testing preprocessing...")
    
    test_text = "Computer Vision and Machine Learning algorithms"
    result = preprocess(test_text)
    expected = ['comput', 'vision', 'machin', 'learn', 'algorithm']
    
    if result == expected:
        print("✅ Preprocessing test passed!")
        return True
    else:
        print(f"❌ Preprocessing test failed!")
        print(f"Expected: {expected}")
        print(f"Got: {result}")
        return False

if __name__ == "__main__":
    print("🔍 Cluster-then-Search Engine Test Suite")
    print("=" * 40)
    
    # Test preprocessing
    if not test_preprocessing():
        sys.exit(1)
    
    # Test with Inspec dataset
    if not test_inspec_dataset():
        sys.exit(1)
    
    print("\n🎉 All tests passed!")
    print("✅ Engine is ready to use!")
