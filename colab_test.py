"""
Minimal test for Google Colab to debug framework issues.

Run this in Colab to isolate and identify the problem.
"""

def test_basic_imports():
    """Test basic Python imports."""
    print("Testing basic imports...")
    
    try:
        import sys
        import os
        from pathlib import Path
        print("✓ Basic Python modules")
    except Exception as e:
        print(f"❌ Basic Python: {e}")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name()}")
    except Exception as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except Exception as e:
        print(f"❌ Transformers: {e}")
        return False
    
    return True

def test_framework_imports():
    """Test framework-specific imports."""
    print("\nTesting framework imports...")
    
    try:
        # Add src to path
        import sys
        from pathlib import Path
        src_path = Path("src")
        if src_path.exists():
            sys.path.insert(0, str(src_path))
        else:
            print("❌ src directory not found")
            return False
        
        print("✓ src directory found")
    except Exception as e:
        print(f"❌ Path setup: {e}")
        return False
    
    try:
        from src.core.config import ConfigManager
        print("✓ ConfigManager import")
    except Exception as e:
        print(f"❌ ConfigManager: {e}")
        return False
    
    try:
        from src.utils.logger import initialize_logging
        print("✓ Logger import")
    except Exception as e:
        print(f"❌ Logger: {e}")
        return False
    
    return True

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        from src.core.config import ConfigManager
        
        # Check if config file exists
        config_path = Path("config/config.yaml")
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return False
        
        print("✓ Config file exists")
        
        # Try to load config
        config_manager = ConfigManager(
            config_path="config/config.yaml",
            environment="colab"
        )
        
        print("✓ Configuration loaded")
        print(f"  Model: {config_manager.config.model.base_model}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_logger_init():
    """Test logger initialization."""
    print("\nTesting logger initialization...")
    
    try:
        from src.utils.logger import initialize_logging, get_logger
        
        # Initialize logging
        log_manager = initialize_logging(
            log_dir="test_logs",
            structured_logging=False
        )
        
        print("✓ Logger manager created")
        
        # Get a logger
        logger = get_logger("test")
        logger.info("Test log message")
        
        print("✓ Logger working")
        return True
        
    except Exception as e:
        print(f"❌ Logger init: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_manager():
    """Test memory manager initialization."""
    print("\nTesting memory manager...")
    
    try:
        from src.utils.memory import MemoryManager
        
        # Create memory manager without monitoring to avoid psutil issues
        memory_manager = MemoryManager(
            auto_cleanup=True,
            enable_monitoring=False  # Disable to avoid dependency issues
        )
        
        print("✓ Memory manager created")
        
        # Test basic functionality
        stats = memory_manager.get_memory_stats()
        print(f"✓ Memory stats: {len(stats)} metrics")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory manager: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run diagnostic tests."""
    print("🧪 Colab Diagnostic Test")
    print("=" * 40)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Framework Imports", test_framework_imports),
        ("Config Loading", test_config_loading),
        ("Logger Init", test_logger_init),
        ("Memory Manager", test_memory_manager)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 20)
        try:
            result = test_func()
            results.append(result)
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    print("📊 Test Summary")
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Framework should work.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        
    return 0 if passed == total else 1

if __name__ == "__main__":
    main()
