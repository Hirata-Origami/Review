"""
Llama Model Adapter Module

Advanced adapter for Llama family models with QLoRA optimization,
memory-efficient training, and financial domain specialization.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Union
from pathlib import Path
import warnings
from dataclasses import dataclass

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig, 
    TaskType, 
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except (ImportError, AssertionError) as e:
    UNSLOTH_AVAILABLE = False
    FastLanguageModel = None

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    bnb = None

from ..core.config import ModelConfig
from ..utils.logger import get_logger
from ..utils.memory import MemoryManager

logger = get_logger(__name__)

@dataclass
class ModelMetrics:
    """Container for model metrics and statistics"""
    total_parameters: int
    trainable_parameters: int
    memory_usage_mb: float
    quantization_enabled: bool
    lora_rank: int
    
    @property
    def trainable_percentage(self) -> float:
        return (self.trainable_parameters / self.total_parameters) * 100
    
    def __str__(self) -> str:
        return f"""Model Metrics:
  Total parameters: {self.total_parameters:,}
  Trainable parameters: {self.trainable_parameters:,} ({self.trainable_percentage:.2f}%)
  Memory usage: {self.memory_usage_mb:.1f} MB
  Quantization: {'Enabled' if self.quantization_enabled else 'Disabled'}
  LoRA rank: {self.lora_rank}"""

class LlamaFinancialAdapter:
    """
    Advanced Llama model adapter for financial domain fine-tuning.
    
    Features:
    - QLoRA optimization for memory efficiency
    - Unsloth integration for accelerated training
    - Comprehensive model metrics tracking
    - Memory management and optimization
    - Domain-specific adaptations
    """
    
    def __init__(self, config: ModelConfig, device: Optional[torch.device] = None):
        """
        Initialize Llama adapter.
        
        Args:
            config: Model configuration
            device: Target device (auto-detected if None)
        """
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_manager = MemoryManager()
        
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
        logger.info(f"Initialized LlamaFinancialAdapter for device: {self.device}")
    
    def load_base_model(self, use_unsloth: bool = True) -> Tuple[nn.Module, AutoTokenizer]:
        """
        Load base Llama model with optimizations.
        
        Args:
            use_unsloth: Whether to use Unsloth for optimization
            
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading base model: {self.config.base_model}")
        
        try:
            if use_unsloth and self.device.type == "cuda" and UNSLOTH_AVAILABLE:
                # Use Unsloth for optimal performance
                model, tokenizer = self._load_with_unsloth()
            else:
                # Fallback to standard HuggingFace loading
                if use_unsloth and not UNSLOTH_AVAILABLE:
                    logger.warning("Unsloth requested but not available, using transformers")
                model, tokenizer = self._load_with_transformers()
            
            self.model = model
            self.tokenizer = tokenizer
            
            # Log model metrics
            metrics = self._compute_model_metrics()
            logger.info(f"Model loaded successfully:\n{metrics}")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise
    
    def _load_with_unsloth(self) -> Tuple[nn.Module, AutoTokenizer]:
        """Load model using Unsloth optimization."""
        try:
            # Unsloth works best with CUDA
            if self.device.type != "cuda":
                logger.warning("Unsloth optimization requires CUDA. Falling back to transformers.")
                return self._load_with_transformers()
            
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.base_model,
                max_seq_length=self.config.max_sequence_length,
                dtype=getattr(torch, self.config.compute_dtype, torch.float16),
                load_in_4bit=self.config.quantization_enabled,
                trust_remote_code=True
            )
            
            logger.info("Model loaded with Unsloth optimization")
            return model, tokenizer
            
        except Exception as e:
            logger.warning(f"Unsloth loading failed: {e}. Falling back to transformers.")
            return self._load_with_transformers()
    
    def _load_with_transformers(self) -> Tuple[nn.Module, AutoTokenizer]:
        """Load model using standard transformers library."""
        logger.info(f"Loading model with transformers from: {self.config.base_model}")
        
        # Configure quantization (only for CUDA)
        quantization_config = None
        if self.config.quantization_enabled and self.device.type == "cuda" and BITSANDBYTES_AVAILABLE:
            logger.info("Setting up 4-bit quantization for CUDA device")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, self.config.compute_dtype, torch.float16),
            )
        elif self.config.quantization_enabled:
            logger.warning("Quantization requested but not supported on this device or bitsandbytes not available. Disabling quantization.")
        
        # Load tokenizer first
        logger.info("Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                trust_remote_code=True,
                padding_side="right"
            )
            logger.info("✓ Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info("Added missing pad token")
        
        # Determine device map and dtype based on device
        if self.device.type == "cuda":
            device_map = "auto"
            torch_dtype = getattr(torch, self.config.compute_dtype, torch.float16)
            logger.info(f"Using CUDA with device_map='auto' and torch_dtype={torch_dtype}")
        elif self.device.type == "mps":
            device_map = None  # MPS doesn't support device_map="auto"
            torch_dtype = torch.float32  # MPS works better with float32
            logger.info("Using MPS with manual device placement and float32")
        else:
            device_map = None
            torch_dtype = torch.float32
            logger.info("Using CPU with manual device placement and float32")
        
        # Load model with error handling
        logger.info("Loading model...")
        try:
            # Simplified loading for Mac M1/MPS to avoid configuration issues
            if self.device.type == "mps":
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True
                )
            logger.info("✓ Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Attempting fallback model loading...")
            
            # Fallback: try without optional parameters
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True
                )
                logger.info("✓ Model loaded with fallback method")
            except Exception as fallback_e:
                logger.error(f"Fallback loading also failed: {fallback_e}")
                raise
        
        # Move to device if not using device_map
        if device_map is None:
            model = model.to(self.device)
        
        # Prepare for k-bit training if quantized
        if quantization_config and BITSANDBYTES_AVAILABLE:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True
            )
        
        logger.info("Model loaded with transformers")
        return model, tokenizer
    
    def setup_lora_adaptation(self) -> nn.Module:
        """
        Setup LoRA adaptation for efficient fine-tuning.
        
        Returns:
            LoRA-adapted model
        """
        if self.model is None:
            raise ValueError("Base model must be loaded first")
        
        logger.info("Setting up LoRA adaptation...")
        
        try:
            # Check if using Unsloth
            if hasattr(self.model, 'get_peft_model'):
                # Unsloth path
                peft_model = FastLanguageModel.get_peft_model(
                    self.model,
                    r=self.config.lora_rank,
                    target_modules=self.config.lora_target_modules,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    bias=self.config.lora_bias,
                    use_gradient_checkpointing="unsloth",
                    random_state=42,
                    use_rslora=False,
                    loftq_config=None,
                )
            else:
                # Standard PEFT path
                lora_config = LoraConfig(
                    r=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    target_modules=self.config.lora_target_modules,
                    lora_dropout=self.config.lora_dropout,
                    bias=self.config.lora_bias,
                    task_type=TaskType.CAUSAL_LM,
                )
                
                peft_model = get_peft_model(self.model, lora_config)
            
            self.peft_model = peft_model
            
            # Log adaptation metrics
            metrics = self._compute_model_metrics()
            logger.info(f"LoRA adaptation complete:\n{metrics}")
            
            return peft_model
            
        except Exception as e:
            logger.error(f"Failed to setup LoRA adaptation: {e}")
            raise
    
    def _compute_model_metrics(self) -> ModelMetrics:
        """Compute comprehensive model metrics."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        
        if self.peft_model:
            trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        else:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate memory usage
        memory_usage = self.memory_manager.get_model_memory_usage(self.model)
        
        return ModelMetrics(
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            memory_usage_mb=memory_usage,
            quantization_enabled=self.config.quantization_enabled,
            lora_rank=self.config.lora_rank
        )
    
    def _supports_flash_attention(self) -> bool:
        """Check if Flash Attention 2 is supported."""
        try:
            import flash_attn
            return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        except ImportError:
            return False
    
    def save_adapter(self, output_dir: Union[str, Path]) -> None:
        """
        Save LoRA adapter weights.
        
        Args:
            output_dir: Directory to save adapter
        """
        if self.peft_model is None:
            raise ValueError("PEFT model not initialized")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save adapter
        self.peft_model.save_pretrained(output_path)
        
        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_path)
        
        # Save configuration
        config_dict = {
            "base_model": self.config.base_model,
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "target_modules": self.config.lora_target_modules,
            "max_sequence_length": self.config.max_sequence_length
        }
        
        import json
        with open(output_path / "adapter_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Adapter saved to {output_path}")
    
    def load_adapter(self, adapter_path: Union[str, Path]) -> nn.Module:
        """
        Load pre-trained LoRA adapter.
        
        Args:
            adapter_path: Path to adapter directory
            
        Returns:
            Model with loaded adapter
        """
        adapter_path = Path(adapter_path)
        
        if not adapter_path.exists():
            raise ValueError(f"Adapter path does not exist: {adapter_path}")
        
        logger.info(f"Loading adapter from {adapter_path}")
        
        try:
            # Load base model if not already loaded
            if self.model is None:
                self.load_base_model()
            
            # Load adapter
            model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                is_trainable=True
            )
            
            self.peft_model = model
            logger.info("Adapter loaded successfully")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load adapter: {e}")
            raise
    
    def merge_and_save_full_model(self, output_dir: Union[str, Path]) -> None:
        """
        Merge LoRA weights with base model and save full model.
        
        Args:
            output_dir: Directory to save merged model
        """
        if self.peft_model is None:
            raise ValueError("PEFT model not initialized")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Merging LoRA weights with base model...")
        
        try:
            # Merge weights
            merged_model = self.peft_model.merge_and_unload()
            
            # Save merged model
            merged_model.save_pretrained(
                output_path,
                safe_serialization=True,
                max_shard_size="5GB"
            )
            
            # Save tokenizer
            if self.tokenizer:
                self.tokenizer.save_pretrained(output_path)
            
            logger.info(f"Merged model saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to merge and save model: {e}")
            raise
    
    def get_model_for_training(self) -> nn.Module:
        """Get the model ready for training."""
        if self.peft_model is not None:
            return self.peft_model
        elif self.model is not None:
            return self.model
        else:
            raise ValueError("No model loaded")
    
    def optimize_for_inference(self) -> None:
        """Optimize model for inference."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        self.model.eval()
        
        # Enable inference optimizations
        if hasattr(self.model, 'fuse_qkv'):
            self.model.fuse_qkv()
        
        # Compile model if using PyTorch 2.0+
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Model compiled for optimized inference")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        return self.memory_manager.get_memory_stats()
    
    def cleanup_memory(self) -> None:
        """Clean up GPU memory."""
        self.memory_manager.cleanup_memory()
        
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup_memory()