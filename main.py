"""
Financial Domain LLM Fine-tuning Framework - Main Orchestrator

Enterprise-grade orchestrator for fine-tuning Llama-3 8B on financial domain data
with comprehensive monitoring, evaluation, and deployment capabilities.

This is the main entry point for the financial LLM fine-tuning framework,
designed to handle the complete pipeline from data processing to model deployment.

Author: Bharath Pranav S
Version: 1.0.0
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Core imports
from src.core.config import get_config_manager, ConfigManager
from src.data.processor import FinancialDataProcessor
from src.models.llama_adapter import LlamaFinancialAdapter
from src.training.trainer import FinancialLLMTrainer
from src.evaluation.evaluator import FinancialLLMEvaluator
from src.utils.logger import initialize_logging, get_logger, get_performance_logger
from src.utils.memory import MemoryManager
from src.utils.validators import DataValidator

# Filter warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} available")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__} available")
    except ImportError:
        missing_deps.append("transformers")
    
    try:
        import datasets
        print(f"✓ Datasets available")
    except ImportError:
        missing_deps.append("datasets")
    
    try:
        import pandas
        print(f"✓ Pandas {pandas.__version__} available")
    except ImportError:
        missing_deps.append("pandas")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {missing_deps}")
        print("Please install them with: pip install " + " ".join(missing_deps))
        return False
    
    print("✓ All core dependencies available")
    return True

class FinancialLLMOrchestrator:
    """
    Main orchestrator for the financial LLM fine-tuning pipeline.
    
    This class coordinates the entire pipeline:
    1. Data loading, validation, and preprocessing
    2. Model loading and adaptation setup
    3. Training with comprehensive monitoring
    4. Evaluation and comparative analysis
    5. Model export and deployment preparation
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 environment: str = "local",
                 log_dir: Optional[str] = None,
                 wandb_project: Optional[str] = None):
        """
        Initialize the orchestrator.
        
        Args:
            config_path: Path to configuration file
            environment: Environment name (local, kaggle, colab, production, mac_m1)
            log_dir: Directory for logs (auto-generated if None)
            wandb_project: Weights & Biases project name
        """
        # Initialize logging first
        self.log_dir = Path(log_dir) if log_dir else Path("outputs/logs")
        self.log_manager = initialize_logging(
            log_dir=self.log_dir,
            structured_logging=True
        )
        
        self.logger = get_logger("orchestrator")
        self.perf_logger = get_performance_logger("orchestrator")
        
        self.logger.info("Initializing Financial LLM Orchestrator...")
        
        try:
            # Initialize configuration
            self.config_manager = get_config_manager(config_path, environment)
            self.logger.info("Configuration loaded successfully")
            self.config_manager.print_config()
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}", exc_info=True)
            raise
        
        # Initialize components
        try:
            self.memory_manager = MemoryManager(auto_cleanup=True, enable_monitoring=True)
            self.logger.info("Memory manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize memory manager: {e}", exc_info=True)
            # Continue without memory manager
            self.memory_manager = None
            
        self.data_processor = None
        self.model_adapter = None
        self.trainer = None
        self.evaluator = None
        
        # Training state
        self.datasets = None
        self.training_completed = False
        self.evaluation_results = {}
        
        # Wandb project
        self.wandb_project = wandb_project
        
        self.logger.info("Orchestrator initialization complete")
    
    def run_complete_pipeline(self, 
                            validate_data: bool = True,
                            train_model: bool = True,
                            evaluate_model: bool = True,
                            export_model: bool = True) -> Dict[str, Any]:
        """
        Run the complete fine-tuning pipeline.
        
        Args:
            validate_data: Whether to validate input data
            train_model: Whether to train the model
            evaluate_model: Whether to evaluate the model
            export_model: Whether to export the final model
            
        Returns:
            Dictionary with pipeline results
        """
        self.logger.info("Starting complete fine-tuning pipeline")
        pipeline_start = time.time()
        
        results = {
            "pipeline_completed": False,
            "stages_completed": [],
            "errors": [],
            "warnings": []
        }
        
        try:
            with self.perf_logger.timer("complete_pipeline"):
                # Stage 1: Data Processing
                if validate_data or train_model:
                    self.logger.info("=" * 50)
                    self.logger.info("STAGE 1: DATA PROCESSING")
                    self.logger.info("=" * 50)
                    
                    data_result = self._run_data_stage(validate_data)
                    results["data_processing"] = data_result
                    results["stages_completed"].append("data_processing")
                
                # Stage 2: Model Training
                if train_model:
                    self.logger.info("=" * 50)
                    self.logger.info("STAGE 2: MODEL TRAINING")
                    self.logger.info("=" * 50)
                    
                    training_result = self._run_training_stage()
                    results["training"] = training_result
                    results["stages_completed"].append("training")
                    self.training_completed = True
                
                # Stage 3: Model Evaluation
                if evaluate_model:
                    self.logger.info("=" * 50)
                    self.logger.info("STAGE 3: MODEL EVALUATION")
                    self.logger.info("=" * 50)
                    
                    evaluation_result = self._run_evaluation_stage()
                    results["evaluation"] = evaluation_result
                    results["stages_completed"].append("evaluation")
                
                # Stage 4: Model Export
                if export_model and self.training_completed:
                    self.logger.info("=" * 50)
                    self.logger.info("STAGE 4: MODEL EXPORT")
                    self.logger.info("=" * 50)
                    
                    export_result = self._run_export_stage()
                    results["export"] = export_result
                    results["stages_completed"].append("export")
            
            # Pipeline completion
            pipeline_duration = time.time() - pipeline_start
            results["pipeline_completed"] = True
            results["pipeline_duration_seconds"] = pipeline_duration
            
            self.logger.info(f"Pipeline completed successfully in {pipeline_duration:.2f} seconds")
            self._generate_final_report(results)
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            results["errors"].append(str(e))
            results["pipeline_completed"] = False
        
        finally:
            # Cleanup
            self._cleanup_resources()
        
        return results
    
    def _run_data_stage(self, validate_data: bool) -> Dict[str, Any]:
        """Run data processing stage."""
        with self.perf_logger.timer("data_processing"):
            # Initialize data processor
            data_config = self.config_manager.get_data_config()
            self.data_processor = FinancialDataProcessor(data_config)
            
            # Load and validate data
            if validate_data:
                validator = DataValidator(
                    min_transcript_length=data_config.min_transcript_length,
                    max_transcript_length=data_config.max_transcript_length
                )
                
                # Validate files
                train_validation = validator.validate_file(data_config.train_path)
                val_validation = validator.validate_file(data_config.val_path)
                
                self.logger.info(f"Training data validation: {'PASSED' if train_validation.is_valid else 'FAILED'}")
                self.logger.info(f"Validation data validation: {'PASSED' if val_validation.is_valid else 'FAILED'}")
                
                if not (train_validation.is_valid and val_validation.is_valid):
                    raise ValueError("Data validation failed")
            
            # Process datasets
            self.datasets, data_stats = self.data_processor.process_dataset(
                data_config.train_path,
                data_config.val_path
            )
            
            # Save processed data
            self.data_processor.save_processed_data(
                self.datasets,
                self.config_manager.config.output.base_dir
            )
            
            return {
                "status": "completed",
                "train_samples": len(self.datasets['train']),
                "validation_samples": len(self.datasets['validation']),
                "data_statistics": data_stats.__dict__
            }
    
    def _run_training_stage(self) -> Dict[str, Any]:
        """Run model training stage."""
        with self.perf_logger.timer("model_training"):
            # Initialize model adapter
            model_config = self.config_manager.get_model_config()
            self.model_adapter = LlamaFinancialAdapter(model_config)
            
            # Load base model
            model, tokenizer = self.model_adapter.load_base_model()
            
            # Setup LoRA adaptation
            peft_model = self.model_adapter.setup_lora_adaptation()
            
            # Tokenize datasets
            tokenized_datasets = self.data_processor.tokenize_instructions(self.datasets)
            
            # Initialize trainer
            training_config = self.config_manager.get_training_config()
            self.trainer = FinancialLLMTrainer(
                config=training_config,
                model_adapter=self.model_adapter,
                output_dir=self.config_manager.config.output.model_dir,
                wandb_project=self.wandb_project
            )
            
            # Setup training arguments
            training_args = self.trainer.setup_training_args(
                train_dataset_size=len(tokenized_datasets['train']),
                eval_dataset_size=len(tokenized_datasets['validation'])
            )
            
            # Create data collator
            data_collator = self.data_processor.create_data_collator()
            
            # Create trainer instance
            trainer_instance = self.trainer.create_trainer(
                train_dataset=tokenized_datasets['train'],
                eval_dataset=tokenized_datasets['validation'],
                data_collator=data_collator
            )
            
            # Start training
            training_result = self.trainer.train()
            
            return {
                "status": "completed",
                "training_result": training_result,
                "model_metrics": self.model_adapter._compute_model_metrics().__dict__
            }
    
    def _run_evaluation_stage(self) -> Dict[str, Any]:
        """Run model evaluation stage."""
        with self.perf_logger.timer("model_evaluation"):
            # Initialize evaluator
            eval_config = self.config_manager.get_evaluation_config()
            self.evaluator = FinancialLLMEvaluator(eval_config)
            
            # Evaluate baseline model (if possible)
            baseline_results = None
            try:
                baseline_adapter = LlamaFinancialAdapter(self.config_manager.get_model_config())
                baseline_adapter.load_base_model(use_unsloth=False)
                
                baseline_individual, baseline_agg = self.evaluator.evaluate_model(
                    baseline_adapter,
                    self.datasets['validation'],
                    "baseline"
                )
                baseline_results = (baseline_individual, baseline_agg)
                
                self.logger.info("Baseline model evaluation completed")
                
            except Exception as e:
                self.logger.warning(f"Baseline evaluation failed: {e}")
            
            # Evaluate fine-tuned model
            finetuned_individual, finetuned_agg = self.evaluator.evaluate_model(
                self.model_adapter,
                self.datasets['validation'],
                "finetuned"
            )
            finetuned_results = (finetuned_individual, finetuned_agg)
            
            self.logger.info("Fine-tuned model evaluation completed")
            
            # Comparative analysis
            comparison_result = None
            if baseline_results:
                comparison_result = self.evaluator.compare_models(
                    baseline_results,
                    finetuned_results,
                    output_dir=self.config_manager.config.output.results_dir
                )
                
                self.logger.info("Comparative analysis completed")
            
            # Store results
            self.evaluation_results = {
                "baseline": baseline_results,
                "finetuned": finetuned_results,
                "comparison": comparison_result
            }
            
            return {
                "status": "completed",
                "baseline_available": baseline_results is not None,
                "finetuned_metrics": finetuned_agg.to_dict(),
                "comparison_available": comparison_result is not None
            }
    
    def _run_export_stage(self) -> Dict[str, Any]:
        """Run model export stage."""
        with self.perf_logger.timer("model_export"):
            export_dir = Path(self.config_manager.config.output.model_dir) / "final_export"
            
            # Export adapter
            adapter_dir = export_dir / "adapter"
            self.model_adapter.save_adapter(adapter_dir)
            
            # Export merged model
            merged_dir = export_dir / "merged_model"
            self.model_adapter.merge_and_save_full_model(merged_dir)
            
            # Save configuration
            self.config_manager.save_config(export_dir / "config.yaml")
            
            self.logger.info(f"Model exported to {export_dir}")
            
            return {
                "status": "completed",
                "export_directory": str(export_dir),
                "adapter_directory": str(adapter_dir),
                "merged_model_directory": str(merged_dir)
            }
    
    def _generate_final_report(self, pipeline_results: Dict[str, Any]):
        """Generate comprehensive final report."""
        report_path = Path(self.config_manager.config.output.base_dir) / "final_report.md"
        
        report = f"""# Financial LLM Fine-tuning Report

## Pipeline Summary
- **Completed**: {pipeline_results['pipeline_completed']}
- **Duration**: {pipeline_results.get('pipeline_duration_seconds', 0):.2f} seconds
- **Stages Completed**: {', '.join(pipeline_results['stages_completed'])}

## Configuration
- **Environment**: {self.config_manager.environment}
- **Model**: {self.config_manager.config.model.base_model}
- **Max Sequence Length**: {self.config_manager.config.model.max_sequence_length}
- **Training Epochs**: {self.config_manager.config.training.num_epochs}

"""
        
        # Add data processing results
        if "data_processing" in pipeline_results:
            data_result = pipeline_results["data_processing"]
            report += f"""## Data Processing
- **Training Samples**: {data_result['train_samples']:,}
- **Validation Samples**: {data_result['validation_samples']:,}

"""
        
        # Add training results
        if "training" in pipeline_results:
            training_result = pipeline_results["training"]
            report += f"""## Training Results
- **Status**: {training_result['status']}
- **Model Parameters**: {training_result['model_metrics']['total_parameters']:,}
- **Trainable Parameters**: {training_result['model_metrics']['trainable_parameters']:,}
- **Training Efficiency**: {training_result['model_metrics']['trainable_percentage']:.2f}% trainable

"""
        
        # Add evaluation results
        if "evaluation" in pipeline_results:
            eval_result = pipeline_results["evaluation"]
            report += f"""## Evaluation Results
- **Baseline Available**: {eval_result['baseline_available']}
- **Comparison Available**: {eval_result['comparison_available']}

### Key Metrics
"""
            if "finetuned_metrics" in eval_result:
                metrics = eval_result["finetuned_metrics"]["metrics_summary"]
                for metric_name, stats in list(metrics.items())[:5]:  # Top 5 metrics
                    report += f"- **{metric_name}**: {stats['mean']:.4f} ± {stats['std']:.4f}\n"
        
        # Add export information
        if "export" in pipeline_results:
            export_result = pipeline_results["export"]
            report += f"""
## Model Export
- **Export Directory**: {export_result['export_directory']}
- **Adapter**: {export_result['adapter_directory']}
- **Merged Model**: {export_result['merged_model_directory']}
"""
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Final report saved to {report_path}")
    
    def _cleanup_resources(self):
        """Cleanup resources and finalize logging."""
        self.logger.info("Cleaning up resources...")
        
        # Memory cleanup
        if self.memory_manager:
            final_stats = self.memory_manager.cleanup_memory(force=True)
            self.logger.info(f"Final memory cleanup: {final_stats}")
        
        # Close trainer resources
        if self.trainer:
            self.trainer.cleanup()
        
        # Log final memory stats
        if self.memory_manager:
            memory_summary = self.memory_manager.get_memory_summary()
            self.logger.info(f"Final memory state:\n{memory_summary}")
        
        self.logger.info("Resource cleanup completed")
    
    def quick_evaluation(self) -> Dict[str, Any]:
        """Run quick evaluation for fast feedback."""
        if not self.datasets or not self.model_adapter:
            raise ValueError("Pipeline must be run first or data/model not available")
        
        eval_config = self.config_manager.get_evaluation_config()
        evaluator = FinancialLLMEvaluator(eval_config)
        
        quick_metrics = evaluator.quick_evaluate(
            self.model_adapter,
            self.datasets['validation']
        )
        
        self.logger.info(f"Quick evaluation completed: {quick_metrics}")
        return quick_metrics

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Financial Domain LLM Fine-tuning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py --config config/config.yaml --environment local
  
  # Run with Kaggle environment
  python main.py --environment kaggle --wandb-project financial-llm
  
  # Run with Mac M1 Pro optimization
  python main.py --environment mac_m1 --wandb-project financial-llm
  
  # Quick evaluation only
  python main.py --quick-eval --no-train
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (default: config/config.yaml)"
    )
    
    parser.add_argument(
        "--environment", "-e",
        type=str,
        default="local",
        choices=["local", "kaggle", "colab", "production", "mac_m1"],
        help="Environment configuration to use"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Directory for log files (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="Weights & Biases project name for monitoring"
    )
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip data validation"
    )
    
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip model training"
    )
    
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip model evaluation"
    )
    
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip model export"
    )
    
    parser.add_argument(
        "--quick-eval",
        action="store_true",
        help="Run quick evaluation only"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser

def main():
    """Main entry point."""
    print("🚀 Financial LLM Fine-tuning Framework")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        return 1
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Initialize orchestrator
        orchestrator = FinancialLLMOrchestrator(
            config_path=args.config,
            environment=args.environment,
            log_dir=args.log_dir,
            wandb_project=args.wandb_project
        )
        
        if args.quick_eval:
            # Quick evaluation mode
            # First need to load data and model
            orchestrator._run_data_stage(validate_data=not args.no_validate)
            # Load pre-trained model (would need path specification)
            results = {"quick_evaluation": orchestrator.quick_evaluation()}
        else:
            # Full pipeline
            results = orchestrator.run_complete_pipeline(
                validate_data=not args.no_validate,
                train_model=not args.no_train,
                evaluate_model=not args.no_eval,
                export_model=not args.no_export
            )
        
        # Print final status
        if results.get("pipeline_completed", False):
            print("\n🎉 Pipeline completed successfully!")
            print(f"Results saved to: {orchestrator.config_manager.config.output.base_dir}")
        else:
            print("\n❌ Pipeline failed!")
            if "errors" in results:
                for error in results["errors"]:
                    print(f"  Error: {error}")
        
        return 0 if results.get("pipeline_completed", False) else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())