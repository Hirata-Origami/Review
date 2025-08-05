"""
Advanced Training Module

Enterprise-grade training framework for financial domain LLM fine-tuning
with comprehensive monitoring, optimization, and fault tolerance.
"""

import os
import time
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, asdict
import warnings
from contextlib import contextmanager

from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    get_scheduler
)
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers.training_args import OptimizerNames
import wandb
from torch.utils.tensorboard import SummaryWriter

from ..core.config import TrainingConfig
from ..models.llama_adapter import LlamaFinancialAdapter
from ..utils.logger import get_logger
from ..utils.memory import MemoryManager
from ..utils.callbacks import MetricsCallback, MemoryCallback, TimeCallback

logger = get_logger(__name__)

@dataclass
class TrainingMetrics:
    """Container for training metrics and statistics"""
    epoch: int
    step: int
    train_loss: float
    eval_loss: Optional[float]
    learning_rate: float
    grad_norm: float
    memory_usage_mb: float
    samples_per_second: float
    time_elapsed: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class AdvancedTrainingCallback(TrainerCallback):
    """Advanced callback for comprehensive training monitoring."""
    
    def __init__(self, memory_manager: MemoryManager, log_dir: str):
        self.memory_manager = memory_manager
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = None
        self.step_times = []
        self.metrics_history = []
        
        # Initialize tensorboard writer
        self.tb_writer = SummaryWriter(self.log_dir / "tensorboard")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.start_time = time.time()
        logger.info("Training started")
        
        # Log initial memory state
        memory_stats = self.memory_manager.get_memory_stats()
        self.tb_writer.add_scalar("Memory/Initial_Usage", memory_stats.get("used_gb", 0), 0)
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        current_time = time.time()
        step_time = current_time - (self.step_times[-1] if self.step_times else self.start_time)
        self.step_times.append(current_time)
        
        # Collect metrics
        metrics = TrainingMetrics(
            epoch=state.epoch,
            step=state.global_step,
            train_loss=state.log_history[-1].get("train_loss", 0.0) if state.log_history else 0.0,
            eval_loss=state.log_history[-1].get("eval_loss") if state.log_history else None,
            learning_rate=state.log_history[-1].get("learning_rate", 0.0) if state.log_history else 0.0,
            grad_norm=state.log_history[-1].get("train_grad_norm", 0.0) if state.log_history else 0.0,
            memory_usage_mb=self.memory_manager.get_memory_stats().get("used_mb", 0),
            samples_per_second=args.per_device_train_batch_size * args.gradient_accumulation_steps / step_time,
            time_elapsed=current_time - self.start_time
        )
        
        self.metrics_history.append(metrics)
        
        # Log to tensorboard
        self._log_metrics_to_tensorboard(metrics)
        
        # Memory cleanup every 100 steps
        if state.global_step % 100 == 0:
            self.memory_manager.cleanup_memory()
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Called after evaluation."""
        if state.log_history:
            eval_metrics = state.log_history[-1]
            for key, value in eval_metrics.items():
                if key.startswith("eval_"):
                    self.tb_writer.add_scalar(f"Evaluation/{key}", value, state.global_step)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        total_time = time.time() - self.start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save training metrics
        self._save_training_summary()
        
        # Close tensorboard writer
        self.tb_writer.close()
    
    def _log_metrics_to_tensorboard(self, metrics: TrainingMetrics):
        """Log metrics to tensorboard."""
        self.tb_writer.add_scalar("Training/Loss", metrics.train_loss, metrics.step)
        self.tb_writer.add_scalar("Training/Learning_Rate", metrics.learning_rate, metrics.step)
        self.tb_writer.add_scalar("Training/Grad_Norm", metrics.grad_norm, metrics.step)
        self.tb_writer.add_scalar("Performance/Memory_Usage_MB", metrics.memory_usage_mb, metrics.step)
        self.tb_writer.add_scalar("Performance/Samples_Per_Second", metrics.samples_per_second, metrics.step)
        
        if metrics.eval_loss is not None:
            self.tb_writer.add_scalar("Evaluation/Loss", metrics.eval_loss, metrics.step)
    
    def _save_training_summary(self):
        """Save comprehensive training summary."""
        summary = {
            "total_steps": len(self.metrics_history),
            "total_time": self.metrics_history[-1].time_elapsed if self.metrics_history else 0,
            "final_train_loss": self.metrics_history[-1].train_loss if self.metrics_history else 0,
            "final_eval_loss": self.metrics_history[-1].eval_loss if self.metrics_history else None,
            "avg_samples_per_second": np.mean([m.samples_per_second for m in self.metrics_history]),
            "peak_memory_usage_mb": max([m.memory_usage_mb for m in self.metrics_history]),
            "metrics_history": [m.to_dict() for m in self.metrics_history]
        }
        
        with open(self.log_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

class FinancialLLMTrainer:
    """
    Advanced trainer for financial domain LLM fine-tuning.
    
    Features:
    - Comprehensive monitoring and logging
    - Memory optimization and management
    - Fault tolerance and checkpointing
    - Multiple evaluation metrics
    - Integration with W&B and TensorBoard
    """
    
    def __init__(self, 
                 config: TrainingConfig,
                 model_adapter: LlamaFinancialAdapter,
                 output_dir: str,
                 wandb_project: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
            model_adapter: Model adapter instance
            output_dir: Output directory for checkpoints and logs
            wandb_project: Weights & Biases project name
        """
        self.config = config
        self.model_adapter = model_adapter
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_manager = MemoryManager()
        self.trainer = None
        self.training_args = None
        
        # Initialize monitoring
        self._setup_monitoring(wandb_project)
        
        logger.info(f"Initialized FinancialLLMTrainer with output dir: {self.output_dir}")
    
    def _setup_monitoring(self, wandb_project: Optional[str]):
        """Setup monitoring tools."""
        # Initialize W&B if project specified
        if wandb_project:
            try:
                wandb.init(
                    project=wandb_project,
                    config=asdict(self.config),
                    tags=["llama3", "financial", "qlora"]
                )
                logger.info(f"W&B initialized for project: {wandb_project}")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
    
    def setup_training_args(self, 
                           train_dataset_size: int,
                           eval_dataset_size: int) -> TrainingArguments:
        """
        Setup comprehensive training arguments.
        
        Args:
            train_dataset_size: Size of training dataset
            eval_dataset_size: Size of evaluation dataset
            
        Returns:
            TrainingArguments object
        """
        # Calculate steps
        steps_per_epoch = train_dataset_size // (
            self.config.per_device_train_batch_size * 
            self.config.gradient_accumulation_steps
        )
        total_steps = steps_per_epoch * self.config.num_epochs
        
        # Determine eval steps
        eval_steps = min(self.config.eval_steps, steps_per_epoch // 2)
        
        # Setup report_to
        report_to = []
        if wandb.run is not None:
            report_to.append("wandb")
        report_to.append("tensorboard")
        
        self.training_args = TrainingArguments(
            # Basic settings
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            
            # Training hyperparameters
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            
            # Batch settings
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            
            # Optimization settings
            gradient_checkpointing=self.config.gradient_checkpointing,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            
            # Evaluation and saving
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Logging
            logging_strategy="steps",
            logging_steps=self.config.logging_steps,
            logging_first_step=True,
            report_to=report_to,
            
            # Performance
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            
            # Stability
            seed=42,
            data_seed=42,
            
            # Advanced settings
            ddp_find_unused_parameters=False,
            group_by_length=True,
            length_column_name="length",
            
            # Tensorboard settings
            logging_dir=str(self.output_dir / "logs"),
        )
        
        logger.info(f"Training arguments configured for {total_steps} total steps")
        return self.training_args
    
    def create_trainer(self, 
                      train_dataset,
                      eval_dataset,
                      data_collator,
                      compute_metrics: Optional[Callable] = None) -> Trainer:
        """
        Create Trainer instance with advanced callbacks.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator
            compute_metrics: Optional metrics computation function
            
        Returns:
            Configured Trainer instance
        """
        if self.training_args is None:
            raise ValueError("Training arguments not setup. Call setup_training_args first.")
        
        # Get model for training
        model = self.model_adapter.get_model_for_training()
        tokenizer = self.model_adapter.tokenizer
        
        # Setup callbacks
        callbacks = [
            AdvancedTrainingCallback(
                memory_manager=self.memory_manager,
                log_dir=str(self.output_dir / "logs")
            ),
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001
            )
        ]
        
        # Create trainer
        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
        
        logger.info("Trainer created with advanced monitoring")
        return self.trainer
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute training with comprehensive monitoring.
        
        Args:
            resume_from_checkpoint: Path to checkpoint for resuming
            
        Returns:
            Training results dictionary
        """
        if self.trainer is None:
            raise ValueError("Trainer not created. Call create_trainer first.")
        
        logger.info("Starting training...")
        
        try:
            # Pre-training memory check
            initial_memory = self.memory_manager.get_memory_stats()
            logger.info(f"Initial memory usage: {initial_memory}")
            
            # Start training
            train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            # Post-training memory check
            final_memory = self.memory_manager.get_memory_stats()
            logger.info(f"Final memory usage: {final_memory}")
            
            # Save final model
            self._save_final_model()
            
            # Generate training report
            training_report = self._generate_training_report(train_result)
            
            logger.info("Training completed successfully")
            return training_report
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self._save_error_checkpoint()
            raise
        
        finally:
            # Cleanup
            self.memory_manager.cleanup_memory()
    
    def _save_final_model(self):
        """Save the final trained model."""
        try:
            # Save the adapter
            adapter_dir = self.output_dir / "final_adapter"
            self.model_adapter.save_adapter(adapter_dir)
            
            # Save merged model if requested
            merged_dir = self.output_dir / "final_merged_model"
            self.model_adapter.merge_and_save_full_model(merged_dir)
            
            logger.info("Final models saved")
            
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")
    
    def _save_error_checkpoint(self):
        """Save checkpoint in case of error for debugging."""
        try:
            error_dir = self.output_dir / "error_checkpoint"
            error_dir.mkdir(exist_ok=True)
            
            if self.trainer and self.trainer.state:
                self.trainer.save_state()
                logger.info(f"Error checkpoint saved to {error_dir}")
        except:
            pass  # Best effort
    
    def _generate_training_report(self, train_result) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        # Load training summary
        summary_path = self.output_dir / "logs" / "training_summary.json"
        training_summary = {}
        if summary_path.exists():
            with open(summary_path) as f:
                training_summary = json.load(f)
        
        # Compute model metrics
        model_metrics = self.model_adapter._compute_model_metrics()
        
        report = {
            "training_completed": True,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "final_train_loss": train_result.metrics.get("train_loss", 0),
            "model_metrics": {
                "total_parameters": model_metrics.total_parameters,
                "trainable_parameters": model_metrics.trainable_parameters,
                "trainable_percentage": model_metrics.trainable_percentage,
                "memory_usage_mb": model_metrics.memory_usage_mb
            },
            "training_summary": training_summary,
            "config": asdict(self.config)
        }
        
        # Save report
        with open(self.output_dir / "training_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def evaluate_model(self, eval_dataset, metric_key_prefix: str = "eval") -> Dict[str, float]:
        """
        Evaluate model on given dataset.
        
        Args:
            eval_dataset: Dataset for evaluation
            metric_key_prefix: Prefix for metric keys
            
        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer not created")
        
        logger.info(f"Evaluating model on {len(eval_dataset)} samples...")
        
        eval_results = self.trainer.evaluate(
            eval_dataset=eval_dataset,
            metric_key_prefix=metric_key_prefix
        )
        
        logger.info(f"Evaluation completed. Loss: {eval_results.get(f'{metric_key_prefix}_loss', 'N/A')}")
        return eval_results
    
    def save_checkpoint(self, checkpoint_name: str = "manual_checkpoint") -> str:
        """
        Save manual checkpoint.
        
        Args:
            checkpoint_name: Name for the checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        if self.trainer is None:
            raise ValueError("Trainer not created")
        
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(exist_ok=True)
        
        self.trainer.save_model(str(checkpoint_dir))
        self.trainer.save_state()
        
        logger.info(f"Manual checkpoint saved to {checkpoint_dir}")
        return str(checkpoint_dir)
    
    @contextmanager
    def memory_monitoring(self):
        """Context manager for memory monitoring."""
        initial_stats = self.memory_manager.get_memory_stats()
        logger.info(f"Memory monitoring started: {initial_stats}")
        
        try:
            yield
        finally:
            final_stats = self.memory_manager.get_memory_stats()
            logger.info(f"Memory monitoring ended: {final_stats}")
            self.memory_manager.cleanup_memory()
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress."""
        if self.trainer is None or self.trainer.state is None:
            return {"status": "not_started"}
        
        state = self.trainer.state
        return {
            "status": "training",
            "epoch": state.epoch,
            "global_step": state.global_step,
            "max_steps": state.max_steps,
            "progress_percentage": (state.global_step / state.max_steps) * 100 if state.max_steps > 0 else 0,
            "best_metric": state.best_metric,
            "best_model_checkpoint": state.best_model_checkpoint
        }
    
    def cleanup(self):
        """Cleanup resources."""
        if wandb.run is not None:
            wandb.finish()
        
        self.memory_manager.cleanup_memory()
        logger.info("Trainer cleanup completed")