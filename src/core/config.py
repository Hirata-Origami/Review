"""
Configuration Management Module

Provides centralized configuration management with validation, environment-specific overrides,
and type safety for the financial LLM fine-tuning framework.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig
import torch
from rich.console import Console

console = Console()

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    base_model: str = "unsloth/Meta-Llama-3-8B-bnb-4bit"
    model_family: str = "llama3"
    max_sequence_length: int = 2048
    quantization_enabled: bool = True
    quantization_bits: int = 4
    compute_dtype: str = "float16"
    
    # LoRA parameters
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_bias: str = "none"

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    num_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    gradient_checkpointing: bool = True
    fp16: bool = True
    bf16: bool = False
    
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 3
    
    logging_steps: int = 100

@dataclass
class DataConfig:
    """Data processing configuration"""
    train_path: str = "./Train dataset.csv"
    val_path: str = "./Val dataset.csv"
    max_samples: Optional[int] = None
    random_seed: int = 42
    test_size: float = 0.1
    
    # Preprocessing
    lowercase: bool = False
    remove_special_chars: bool = False
    normalize_whitespace: bool = True
    max_transcript_length: int = 1024
    min_transcript_length: int = 10

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    metrics: list = field(default_factory=lambda: [
        "perplexity", "rouge", "bert_score", "bleu", "word_overlap"
    ])
    quick_eval_samples: int = 100
    full_eval_samples: int = 1000
    qualitative_eval_samples: int = 20
    
    # Generation parameters
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1

class ConfigManager:
    """
    Centralized configuration manager for the financial LLM framework.
    
    Provides:
    - Environment-specific configuration loading
    - Configuration validation
    - Runtime configuration updates
    - Type-safe access to configuration parameters
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None, 
                 environment: str = "local"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
            environment: Environment name (local, kaggle, colab, production)
        """
        self.environment = environment
        self.config_path = config_path or Path("config/config.yaml")
        self.config = self._load_config()
        self._validate_config()
        self._setup_paths()
        
    def _load_config(self) -> DictConfig:
        """Load configuration from YAML files with environment overrides."""
        try:
            # Load base configuration
            base_config = OmegaConf.load(self.config_path)
            
            # Load environment-specific overrides
            env_config_path = Path(self.config_path).parent / "environment" / f"{self.environment}.yaml"
            if env_config_path.exists():
                env_config = OmegaConf.load(env_config_path)
                base_config = OmegaConf.merge(base_config, env_config)
            
            # Resolve interpolations
            OmegaConf.resolve(base_config)
            
            console.print(f"[green]✓[/green] Configuration loaded for environment: {self.environment}")
            return base_config
            
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to load configuration: {e}")
            raise
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        try:
            # Validate model configuration
            assert self.config.model.max_sequence_length > 0
            assert self.config.model.lora.rank > 0
            assert 0 <= self.config.model.lora.dropout <= 1
            
            # Validate training configuration
            assert self.config.training.num_epochs > 0
            assert self.config.training.learning_rate > 0
            assert self.config.training.per_device_train_batch_size > 0
            
            # Validate data paths exist (if not using environment variables)
            train_path = Path(self.config.data.train_path)
            val_path = Path(self.config.data.val_path)
            
            if not train_path.exists() and not train_path.as_posix().startswith('/kaggle/'):
                console.print(f"[yellow]⚠[/yellow] Training data path not found: {train_path}")
            
            if not val_path.exists() and not val_path.as_posix().startswith('/kaggle/'):
                console.print(f"[yellow]⚠[/yellow] Validation data path not found: {val_path}")
            
            console.print("[green]✓[/green] Configuration validation passed")
            
        except AssertionError as e:
            console.print(f"[red]✗[/red] Configuration validation failed: {e}")
            raise
    
    def _setup_paths(self) -> None:
        """Create necessary directories."""
        dirs_to_create = [
            self.config.output.base_dir,
            self.config.output.model_dir,
            self.config.output.logs_dir,
            self.config.output.results_dir,
            self.config.output.checkpoints_dir
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def get_model_config(self) -> ModelConfig:
        """Get typed model configuration."""
        return ModelConfig(
            base_model=self.config.model.base_model,
            model_family=self.config.model.model_family,
            max_sequence_length=self.config.model.max_sequence_length,
            quantization_enabled=self.config.model.quantization.enabled,
            quantization_bits=self.config.model.quantization.bits,
            compute_dtype=self.config.model.quantization.compute_dtype,
            lora_rank=self.config.model.lora.rank,
            lora_alpha=self.config.model.lora.alpha,
            lora_dropout=self.config.model.lora.dropout,
            lora_target_modules=self.config.model.lora.target_modules,
            lora_bias=self.config.model.lora.bias
        )
    
    def get_training_config(self) -> TrainingConfig:
        """Get typed training configuration."""
        return TrainingConfig(
            num_epochs=self.config.training.num_epochs,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_ratio=self.config.training.warmup_ratio,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            evaluation_strategy=self.config.training.evaluation_strategy,
            eval_steps=self.config.training.eval_steps,
            save_strategy=self.config.training.save_strategy,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            logging_steps=self.config.training.logging_steps
        )
    
    def get_data_config(self) -> DataConfig:
        """Get typed data configuration."""
        return DataConfig(
            train_path=self.config.data.train_path,
            val_path=self.config.data.val_path,
            max_samples=self.config.data.max_samples,
            random_seed=self.config.data.random_seed,
            test_size=self.config.data.test_size,
            lowercase=self.config.data.preprocessing.lowercase,
            remove_special_chars=self.config.data.preprocessing.remove_special_chars,
            normalize_whitespace=self.config.data.preprocessing.normalize_whitespace,
            max_transcript_length=self.config.data.preprocessing.max_transcript_length,
            min_transcript_length=self.config.data.preprocessing.min_transcript_length
        )
    
    def get_evaluation_config(self) -> EvaluationConfig:
        """Get typed evaluation configuration."""
        return EvaluationConfig(
            metrics=self.config.evaluation.metrics,
            quick_eval_samples=self.config.evaluation.quick_eval_samples,
            full_eval_samples=self.config.evaluation.full_eval_samples,
            qualitative_eval_samples=self.config.evaluation.qualitative_eval_samples,
            max_new_tokens=self.config.evaluation.generation.max_new_tokens,
            temperature=self.config.evaluation.generation.temperature,
            top_p=self.config.evaluation.generation.top_p,
            do_sample=self.config.evaluation.generation.do_sample,
            repetition_penalty=self.config.evaluation.generation.repetition_penalty
        )
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration at runtime."""
        self.config = OmegaConf.merge(self.config, OmegaConf.create(updates))
        self._validate_config()
    
    def get_device(self) -> torch.device:
        """Get the appropriate device for computation."""
        if self.config.hardware.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.hardware.device)
    
    def save_config(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to file."""
        save_path = path or Path(self.config.output.base_dir) / "config_used.yaml"
        OmegaConf.save(self.config, save_path)
        console.print(f"[green]✓[/green] Configuration saved to: {save_path}")
    
    def print_config(self) -> None:
        """Print configuration in a readable format."""
        console.print("\n[bold blue]Configuration Summary[/bold blue]")
        console.print("=" * 50)
        console.print(f"Environment: [cyan]{self.environment}[/cyan]")
        console.print(f"Model: [cyan]{self.config.model.base_model}[/cyan]")
        console.print(f"Max Sequence Length: [cyan]{self.config.model.max_sequence_length}[/cyan]")
        console.print(f"Training Epochs: [cyan]{self.config.training.num_epochs}[/cyan]")
        console.print(f"Batch Size: [cyan]{self.config.training.per_device_train_batch_size}[/cyan]")
        console.print(f"Learning Rate: [cyan]{self.config.training.learning_rate}[/cyan]")
        console.print(f"Output Directory: [cyan]{self.config.output.base_dir}[/cyan]")
        console.print("=" * 50)

# Global configuration instance
_config_manager: Optional[ConfigManager] = None

def get_config_manager(config_path: Optional[Union[str, Path]] = None, 
                      environment: str = "local") -> ConfigManager:
    """Get or create global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path, environment)
    return _config_manager

def reset_config_manager() -> None:
    """Reset global configuration manager (useful for testing)."""
    global _config_manager
    _config_manager = None