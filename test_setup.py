#!/usr/bin/env python3
"""
Simple test script to verify the Financial LLM Framework setup.
Run this to debug installation and configuration issues.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test all critical imports."""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  - CUDA available: {torch.cuda.get_device_name()}")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"  - MPS available")
    except Exception as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except Exception as e:
        print(f"❌ Transformers: {e}")
        return False
    
    try:
        import datasets
        print(f"✓ Datasets")
    except Exception as e:
        print(f"❌ Datasets: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except Exception as e:
        print(f"❌ Pandas: {e}")
        return False
    
    return True

def test_paths():
    """Test critical file paths."""
    print("\n📁 Testing file structure...")
    
    critical_paths = [
        "src",
        "src/core",
        "src/core/config.py",
        "src/data/processor.py",
        "src/models/llama_adapter.py",
        "config",
        "config/config.yaml"
    ]
    
    all_exist = True
    for path in critical_paths:
        if Path(path).exists():
            print(f"✓ {path}")
        else:
            print(f"❌ Missing: {path}")
            all_exist = False
    
    return all_exist

def test_configuration():
    """Test configuration loading."""
    print("\n⚙️  Testing configuration...")
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        # Test basic imports first
        try:
            from src.core.config import ConfigManager
            print("✓ ConfigManager import successful")
        except Exception as e:
            print(f"❌ ConfigManager import failed: {e}")
            return False
        
        # Test minimal configuration loading without full initialization
        try:
            from omegaconf import OmegaConf
            config = OmegaConf.load("config/config.yaml")
            print("✓ Basic YAML loading works")
            print(f"  - Model: {config.model.base_model}")
        except Exception as e:
            print(f"❌ YAML loading failed: {e}")
            return False
        
        # Test full configuration manager (this might be where it hangs)
        try:
            print("  - Testing full ConfigManager initialization...")
            config_manager = ConfigManager(
                config_path="config/config.yaml",
                environment="local"
            )
            print("✓ Configuration loaded successfully")
            print(f"  - Environment: {config_manager.environment}")
            return True
        except Exception as e:
            print(f"❌ Full configuration loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_files():
    """Test if data files exist."""
    print("\n📊 Testing data files...")
    
    data_files = ["Train dataset.csv", "Val dataset.csv"]
    files_exist = True
    
    for file in data_files:
        if Path(file).exists():
            print(f"✓ {file}")
            # Check file size
            size_mb = Path(file).stat().st_size / (1024 * 1024)
            print(f"  - Size: {size_mb:.1f} MB")
        else:
            print(f"❌ Missing: {file}")
            files_exist = False
    
    if not files_exist:
        print("\n💡 Tip: Make sure your data files are in the correct location.")
        print("   You can also update the paths in config/config.yaml")
    
    return files_exist

def main():
    """Run all tests."""
    print("🧪 Financial LLM Framework - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("File Structure Test", test_paths),
        ("Configuration Test", test_configuration),
        ("Data Files Test", test_data_files)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    
    for i, (test_name, _) in enumerate(tests):
        status = "✓ PASS" if results[i] else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results)
    
    if all_passed:
        print("\n🎉 All tests passed! You can run the main framework.")
        print("\nNext steps:")
        print("  python main.py --environment local")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
