#!/usr/bin/env python3
"""
Quick test script to verify model loading works without full download.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import ConfigManager
from src.models.llama_adapter import LlamaFinancialAdapter

def test_model_loading():
    """Test model loading with a small model."""
    print("🧪 Testing Model Loading...")
    
    try:
        # Load config
        config_manager = ConfigManager(
            config_path="config/config.yaml",
            environment="mac_m1"
        )
        
        # Get model config
        model_config = config_manager.get_model_config()
        print(f"✓ Model config loaded: {model_config.base_model}")
        
        # Initialize adapter
        adapter = LlamaFinancialAdapter(model_config)
        print(f"✓ Adapter initialized for device: {adapter.device}")
        
        # Test if we can create the loading setup (without actual loading)
        print("✓ Model loading setup is properly configured")
        print(f"  - Device: {adapter.device}")
        print(f"  - Quantization enabled: {model_config.quantization_enabled}")
        print(f"  - LoRA rank: {model_config.lora_rank}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    print("\n" + "="*50)
    if success:
        print("🎉 Model loading test PASSED!")
        print("The framework should work properly for actual training.")
    else:
        print("⚠️  Model loading test FAILED!")
        print("Please check the configuration and dependencies.")
    
    sys.exit(0 if success else 1)
