# Financial LLM Fine-tuning Implementation Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Data Preparation](#data-preparation)
5. [Training Pipeline](#training-pipeline)
6. [Evaluation and Analysis](#evaluation-and-analysis)
7. [Model Deployment](#model-deployment)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Usage](#advanced-usage)

## Quick Start

### 1. Setup and Installation

```bash
# Clone the repository
git clone https://github.com/your-org/financial-llm-framework.git
cd financial-llm-framework

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 2. Basic Training Run

```bash
# Local environment with default settings
python main.py --environment local

# Kaggle environment
python main.py --environment kaggle --wandb-project financial-llm-demo

# Quick evaluation only (no training)
python main.py --quick-eval --no-train
```

### 3. Expected Outputs

After successful completion, you'll find:
```
outputs/
├── models/
│   ├── final_adapter/          # LoRA adapter weights
│   ├── final_merged_model/     # Complete merged model
│   └── checkpoints/           # Training checkpoints
├── results/
│   ├── comparison_report.json # Evaluation results
│   ├── metrics_comparison.csv # Performance metrics
│   └── visualizations/        # Charts and plots
├── logs/
│   ├── training_summary.json  # Training metrics
│   └── tensorboard/          # TensorBoard logs
└── final_report.md           # Comprehensive report
```

## Installation

### 1. System Requirements

**Minimum Requirements**:
- Python 3.8+
- 16GB RAM
- 8GB GPU VRAM (for training)
- 50GB disk space

**Recommended Setup**:
- Python 3.10+
- 32GB+ RAM
- NVIDIA GPU with 24GB+ VRAM
- 100GB+ SSD storage

### 2. Environment Setup

**Option A: Conda Environment**
```bash
# Create conda environment
conda create -n financial-llm python=3.10
conda activate financial-llm

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining dependencies
pip install -r requirements.txt
```

**Option B: Docker Environment**
```bash
# Build Docker image
docker build -t financial-llm:latest .

# Run container with GPU support
docker run --gpus all -v $(pwd):/workspace -it financial-llm:latest
```

**Option C: Google Colab**
```python
# In Colab notebook
!git clone https://github.com/your-org/financial-llm-framework.git
%cd financial-llm-framework
!pip install -r requirements.txt

# Run with Colab-specific settings
!python main.py --environment colab
```

### 3. Dependency Installation

**Core Dependencies**:
```bash
# Essential ML libraries
pip install torch>=2.1.0 transformers>=4.36.0 accelerate>=0.25.0

# Unsloth for optimization
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Data processing
pip install datasets pandas numpy scikit-learn

# Evaluation metrics
pip install rouge-score sacrebleu bert-score

# Configuration and logging
pip install hydra-core omegaconf wandb rich

# Statistical analysis
pip install scipy statsmodels matplotlib seaborn
```

**Optional Dependencies**:
```bash
# For advanced features
pip install flash-attn  # Flash Attention 2 (if supported)
pip install deepspeed   # For distributed training
pip install tensorboard # Alternative to W&B
```

## Configuration

### 1. Configuration Files

The framework uses YAML configuration files with environment-specific overrides:

**Main Configuration** (`config/config.yaml`):
```yaml
# Core settings
project:
  name: "financial-llm-finetune"
  version: "1.0.0"

model:
  base_model: "unsloth/Meta-Llama-3-8B-bnb-4bit"
  max_sequence_length: 2048
  lora:
    rank: 16
    alpha: 32
    dropout: 0.1

training:
  num_epochs: 3
  learning_rate: 2e-4
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4

data:
  train_path: "./Train dataset.csv"
  val_path: "./Val dataset.csv"
```

**Environment Overrides** (`config/environment/kaggle.yaml`):
```yaml
# @package _global_
# Kaggle-specific overrides
training:
  per_device_train_batch_size: 2  # Reduced for memory constraints
  gradient_accumulation_steps: 8   # Maintain effective batch size

data:
  train_path: "/kaggle/input/spgispeech/train.csv.bz2"
  val_path: "/kaggle/input/spgispeech/val.csv.bz2"

output:
  base_dir: "/kaggle/working/outputs"
```

### 2. Configuration Management

**Loading Configuration**:
```python
from src.core.config import get_config_manager

# Load configuration for specific environment
config_manager = get_config_manager(
    config_path="config/config.yaml",
    environment="kaggle"
)

# Access typed configuration
model_config = config_manager.get_model_config()
training_config = config_manager.get_training_config()
```

**Runtime Configuration Updates**:
```python
# Update configuration at runtime
config_manager.update_config({
    "training": {
        "learning_rate": 1e-4,
        "num_epochs": 5
    }
})
```

### 3. Environment Variables

Set environment variables for sensitive information:
```bash
export WANDB_API_KEY="your-wandb-key"
export HUGGINGFACE_TOKEN="your-hf-token"
export TRAIN_CSV_PATH="/path/to/train.csv"
export VAL_CSV_PATH="/path/to/val.csv"
```

## Data Preparation

### 1. Expected Data Format

**CSV Structure**:
```csv
wav_filename,wav_filesize,transcript
audio1.wav,123456,"The company reported strong quarterly earnings..."
audio2.wav,234567,"Our revenue growth exceeded expectations..."
```

**Required Columns**:
- `transcript`: The text content for training
- Additional columns (like `wav_filename`) are preserved but optional

### 2. Data Validation

**Automatic Validation**:
```python
from src.utils.validators import DataValidator

validator = DataValidator(
    min_transcript_length=10,
    max_transcript_length=1024,
    required_columns=['transcript']
)

# Validate dataset
validation_result = validator.validate_dataset(dataframe)
print(validation_result)
```

**Manual Data Inspection**:
```python
import pandas as pd

# Load and inspect data
df = pd.read_csv("Train dataset.csv", sep='|')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Sample transcripts:")
for i, row in df.head(3).iterrows():
    print(f"{i}: {row['transcript'][:100]}...")
```

### 3. Data Preprocessing

**Text Preprocessing Options**:
```yaml
# In config.yaml
data:
  preprocessing:
    lowercase: false              # Preserve capitalization
    remove_special_chars: false  # Keep punctuation
    normalize_whitespace: true   # Clean up spacing
    max_transcript_length: 1024  # Token limit
    min_transcript_length: 10    # Quality filter
```

**Custom Preprocessing**:
```python
def custom_preprocess(text):
    # Custom preprocessing logic
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()               # Remove leading/trailing space
    return text

# Apply during data processing
processor = FinancialDataProcessor(config)
processor.custom_preprocessor = custom_preprocess
```

## Training Pipeline

### 1. Complete Pipeline Execution

**Basic Training**:
```python
from main import FinancialLLMOrchestrator

# Initialize orchestrator
orchestrator = FinancialLLMOrchestrator(
    config_path="config/config.yaml",
    environment="local",
    wandb_project="financial-llm-experiment"
)

# Run complete pipeline
results = orchestrator.run_complete_pipeline(
    validate_data=True,
    train_model=True,
    evaluate_model=True,
    export_model=True
)

print(f"Pipeline completed: {results['pipeline_completed']}")
```

**Stage-by-Stage Execution**:
```python
# Data processing only
data_result = orchestrator._run_data_stage(validate_data=True)

# Training only (requires data to be processed first)
training_result = orchestrator._run_training_stage()

# Evaluation only
evaluation_result = orchestrator._run_evaluation_stage()

# Export only
export_result = orchestrator._run_export_stage()
```

### 2. Training Monitoring

**Real-time Monitoring**:
```bash
# Monitor with TensorBoard
tensorboard --logdir outputs/logs/tensorboard

# Monitor with W&B (automatic if configured)
# Visit https://wandb.ai/your-project
```

**Progress Tracking**:
```python
# Check training progress
progress = trainer.get_training_progress()
print(f"Epoch: {progress['epoch']}")
print(f"Progress: {progress['progress_percentage']:.1f}%")
```

**Memory Monitoring**:
```python
# Monitor memory usage
memory_stats = orchestrator.memory_manager.get_memory_stats()
print(f"GPU Memory: {memory_stats['gpu_allocated_gb']:.2f}GB")
print(f"CPU Memory: {memory_stats['used_gb']:.2f}GB")
```

### 3. Training Optimization

**Memory Optimization**:
```python
# Enable memory optimizations
model_adapter = LlamaFinancialAdapter(model_config)
optimizations = model_adapter.optimize_memory_for_training(model)
print(f"Applied optimizations: {optimizations}")
```

**Batch Size Optimization**:
```python
# Find optimal batch size
def find_optimal_batch_size(start_size=1, max_size=16):
    for batch_size in range(start_size, max_size + 1):
        try:
            # Test batch size
            config.training.per_device_train_batch_size = batch_size
            # Run small training test
            return batch_size
        except torch.cuda.OutOfMemoryError:
            continue
    return start_size
```

## Evaluation and Analysis

### 1. Comprehensive Evaluation

**Full Evaluation**:
```python
from src.evaluation.evaluator import FinancialLLMEvaluator

evaluator = FinancialLLMEvaluator(eval_config)

# Evaluate both models
baseline_results = evaluator.evaluate_model(baseline_adapter, eval_dataset, "baseline")
finetuned_results = evaluator.evaluate_model(finetuned_adapter, eval_dataset, "finetuned")

# Compare models
comparison = evaluator.compare_models(
    baseline_results,
    finetuned_results,
    output_dir="outputs/results"
)
```

**Quick Evaluation**:
```python
# Fast evaluation for development
quick_metrics = evaluator.quick_evaluate(model_adapter, eval_dataset)
print(f"Quick ROUGE-L: {quick_metrics['rougeL_f']:.4f}")
print(f"Quick BLEU: {quick_metrics['bleu']:.4f}")
```

### 2. Statistical Analysis

**Significance Testing**:
```python
from src.utils.statistics import StatisticalAnalyzer

analyzer = StatisticalAnalyzer(alpha=0.05)

# Perform comprehensive comparison
comparison_results = analyzer.comprehensive_comparison(
    baseline_scores=[...],
    treatment_scores=[...]
)

# Generate statistical report
report = analyzer.create_statistical_report(comparison_results)
print(report)
```

**Custom Metrics**:
```python
def financial_accuracy_metric(predictions, references):
    """Custom metric for financial accuracy"""
    financial_terms = ['revenue', 'profit', 'growth', 'margin']
    
    pred_terms = set(pred.lower().split()) & set(financial_terms)
    ref_terms = set(ref.lower().split()) & set(financial_terms)
    
    if not ref_terms:
        return 1.0 if not pred_terms else 0.0
    
    return len(pred_terms & ref_terms) / len(ref_terms)

# Add to evaluator
evaluator.custom_metrics['financial_accuracy'] = financial_accuracy_metric
```

### 3. Visualization and Reporting

**Generate Visualizations**:
```python
# Automatically generated during comparison
comparison = evaluator.compare_models(baseline_results, finetuned_results, "outputs/results")

# Manual visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Plot improvement metrics
metrics = comparison['statistical_comparison']
improvements = [metrics[m]['relative_improvement'] for m in metrics]
metric_names = list(metrics.keys())

plt.figure(figsize=(12, 6))
plt.bar(metric_names, improvements)
plt.title('Performance Improvements by Metric')
plt.ylabel('Relative Improvement (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/improvements.png')
```

## Model Deployment

### 1. Model Export

**Export Options**:
```python
# Export LoRA adapter only (lightweight)
model_adapter.save_adapter("outputs/adapter")

# Export merged model (standalone)
model_adapter.merge_and_save_full_model("outputs/merged_model")

# Export for specific deployment formats
export_config = {
    "formats": ["huggingface", "gguf", "onnx"],
    "quantization": ["int8", "int4"]
}
```

### 2. Inference Setup

**Local Inference**:
```python
from transformers import pipeline

# Load fine-tuned model
generator = pipeline(
    "text-generation",
    model="outputs/merged_model",
    tokenizer="outputs/merged_model",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Generate predictions
prompt = "Analyze the following financial performance:"
result = generator(prompt, max_new_tokens=256, temperature=0.7)
print(result[0]['generated_text'])
```

**API Deployment**:
```python
# FastAPI server example
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class FinancialQuery(BaseModel):
    text: str
    max_tokens: int = 256

@app.post("/analyze")
async def analyze_financial_text(query: FinancialQuery):
    result = generator(
        f"Analyze this financial text: {query.text}",
        max_new_tokens=query.max_tokens,
        temperature=0.7
    )
    return {"analysis": result[0]['generated_text']}

# Run with: uvicorn deploy:app --host 0.0.0.0 --port 8000
```

### 3. Production Considerations

**Model Optimization**:
```python
# Optimize for inference
model_adapter.optimize_for_inference()

# Compile model (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode="reduce-overhead")
```

**Monitoring Setup**:
```python
import time
from collections import deque

class InferenceMonitor:
    def __init__(self):
        self.response_times = deque(maxlen=1000)
        self.error_count = 0
    
    def log_request(self, start_time, success=True):
        duration = time.time() - start_time
        self.response_times.append(duration)
        if not success:
            self.error_count += 1
    
    def get_stats(self):
        if not self.response_times:
            return {}
        return {
            "avg_response_time": sum(self.response_times) / len(self.response_times),
            "p95_response_time": sorted(self.response_times)[int(0.95 * len(self.response_times))],
            "error_rate": self.error_count / len(self.response_times)
        }
```

## Troubleshooting

### 1. Common Issues

**Out of Memory Errors**:
```python
# Solution 1: Reduce batch size
config.training.per_device_train_batch_size = 1
config.training.gradient_accumulation_steps = 16

# Solution 2: Enable gradient checkpointing
config.training.gradient_checkpointing = True

# Solution 3: Use CPU offloading
model_adapter.model = model_adapter.model.cpu()
```

**Training Instability**:
```python
# Solution 1: Lower learning rate
config.training.learning_rate = 1e-4

# Solution 2: Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Solution 3: Increase warmup
config.training.warmup_ratio = 0.1
```

**Data Loading Issues**:
```python
# Check data format
try:
    df = pd.read_csv(data_path, sep='|')
except:
    df = pd.read_csv(data_path, sep=',')  # Try different separator

# Verify required columns
required_cols = ['transcript']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"Missing columns: {missing_cols}")
```

### 2. Performance Issues

**Slow Training**:
```python
# Enable optimizations
config.hardware.use_flash_attention = True
config.hardware.compile_model = True
config.training.dataloader_num_workers = 4
```

**Memory Leaks**:
```python
# Regular cleanup
if step % 100 == 0:
    memory_manager.cleanup_memory()
    torch.cuda.empty_cache()
```

### 3. Debugging Tools

**Logging Configuration**:
```python
# Enable debug logging
from src.utils.logger import initialize_logging
logger_manager = initialize_logging(
    console_level=logging.DEBUG,
    structured_logging=True
)
```

**Memory Profiling**:
```python
# Profile memory usage
with memory_manager.memory_tracking("training_step"):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
```

**Performance Profiling**:
```python
# Profile with PyTorch profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Training step
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Advanced Usage

### 1. Custom Components

**Custom Data Processor**:
```python
class CustomFinancialProcessor(FinancialDataProcessor):
    def _convert_to_instructions(self, df):
        # Custom instruction formatting
        instructions = []
        for _, row in df.iterrows():
            instruction = {
                'instruction': 'Custom financial analysis task:',
                'input': row['transcript'],
                'output': self._generate_custom_output(row['transcript'])
            }
            instructions.append(instruction)
        return instructions
    
    def _generate_custom_output(self, transcript):
        # Custom output generation logic
        return f"Analysis: {transcript[:100]}..."
```

**Custom Evaluation Metrics**:
```python
class CustomMetricCalculator(MetricCalculator):
    def calculate_domain_coherence(self, ground_truth, prediction):
        # Custom domain-specific metric
        financial_keywords = ['revenue', 'profit', 'loss', 'margin']
        
        gt_keywords = [w for w in ground_truth.lower().split() if w in financial_keywords]
        pred_keywords = [w for w in prediction.lower().split() if w in financial_keywords]
        
        if not gt_keywords:
            return 1.0 if not pred_keywords else 0.0
        
        return len(set(gt_keywords) & set(pred_keywords)) / len(set(gt_keywords))
```

### 2. Distributed Training

**Multi-GPU Training**:
```python
# Enable distributed training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = DDP(model, device_ids=[local_rank])

# Use DistributedSampler
from torch.utils.data.distributed import DistributedSampler
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
```

### 3. Hyperparameter Optimization

**Grid Search**:
```python
from itertools import product

# Define hyperparameter grid
param_grid = {
    'learning_rate': [1e-4, 2e-4, 5e-4],
    'lora_rank': [8, 16, 32],
    'lora_alpha': [16, 32, 64]
}

best_config = None
best_score = 0

for lr, rank, alpha in product(*param_grid.values()):
    config.training.learning_rate = lr
    config.model.lora.rank = rank
    config.model.lora.alpha = alpha
    
    # Train and evaluate
    score = train_and_evaluate(config)
    
    if score > best_score:
        best_score = score
        best_config = config.copy()
```

**Bayesian Optimization**:
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

def objective_function(params):
    config.training.learning_rate = params[0]
    config.model.lora.rank = int(params[1])
    return train_and_evaluate(config)

# Bayesian optimization implementation
# (Using libraries like scikit-optimize or Optuna)
```

This implementation guide provides comprehensive coverage of the framework usage, from basic setup to advanced customizations, ensuring users can effectively leverage the system for their financial LLM fine-tuning needs.