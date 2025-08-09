# Financial Domain LLM Fine-tuning Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An enterprise-grade framework for fine-tuning large language models on financial domain data, specifically designed for the SPGISpeech dataset containing business call transcripts. This framework provides comprehensive tools for domain adaptation, evaluation, and deployment of financial AI models.

## 🚀 Key Features

- **🎯 Domain-Specific Fine-tuning**: Specialized for financial language understanding
- **💾 Memory-Efficient Training**: QLoRA optimization for resource-constrained environments
- **📊 Comprehensive Evaluation**: Multiple metrics with statistical significance testing
- **🔧 Enterprise Architecture**: Modular, scalable, and maintainable design
- **📈 Advanced Monitoring**: Real-time training metrics and resource monitoring
- **🚀 Production-Ready**: Complete deployment utilities and model export
- **🔍 Quality Assurance**: Extensive data validation and quality checks

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## 🏃‍♂️ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Hirata-Origami/financial-llm-framework.git
cd financial-llm-framework

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 2. Prepare Your Data

Ensure your data is in CSV format with the following structure:
```csv
wav_filename,wav_filesize,transcript
audio1.wav,123456,"The company reported strong quarterly earnings..."
audio2.wav,234567,"Our revenue growth exceeded expectations..."
```

### 3. Run Training

```bash
# Basic training with default settings
python main.py --environment local

# Training with Weights & Biases monitoring
python main.py --environment local --wandb-project financial-llm-demo

# Kaggle environment (optimized for resource constraints)
python main.py --environment kaggle
```

### 4. Expected Results

After completion, you'll find:
- **Fine-tuned model**: `outputs/models/final_merged_model/`
- **Evaluation results**: `outputs/results/comparison_report.json`
- **Training logs**: `outputs/logs/`
- **Comprehensive report**: `outputs/final_report.md`

## 🛠 Installation

### System Requirements

**Minimum Requirements:**
- Python 3.8+
- 16GB RAM
- 8GB GPU VRAM
- 50GB disk space

**Recommended Setup:**
- Python 3.10+
- 32GB+ RAM
- NVIDIA GPU with 24GB+ VRAM
- 100GB+ SSD storage

### Environment Setup

**Option 1: Conda (Recommended)**
```bash
conda create -n financial-llm python=3.10
conda activate financial-llm
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

**Option 2: Virtual Environment**
```bash
python -m venv financial-llm
source financial-llm/bin/activate  # On Windows: financial-llm\Scripts\activate
pip install -r requirements.txt
```

**Option 3: Docker**
```bash
docker build -t financial-llm:latest .
docker run --gpus all -v $(pwd):/workspace -it financial-llm:latest
```

## 🎯 Usage

### Basic Usage

```python
from main import FinancialLLMOrchestrator

# Initialize orchestrator
orchestrator = FinancialLLMOrchestrator(
    config_path="config/config.yaml",
    environment="local",
    wandb_project="financial-llm-experiment"
)

# Run complete pipeline
results = orchestrator.run_complete_pipeline()
print(f"Success: {results['pipeline_completed']}")
```

### Advanced Usage

```python
# Custom configuration
config_updates = {
    "training": {
        "learning_rate": 1e-4,
        "num_epochs": 5
    },
    "model": {
        "lora": {
            "rank": 32,
            "alpha": 64
        }
    }
}

orchestrator.config_manager.update_config(config_updates)

# Stage-by-stage execution
data_result = orchestrator._run_data_stage(validate_data=True)
training_result = orchestrator._run_training_stage()
evaluation_result = orchestrator._run_evaluation_stage()
```

### Command Line Interface

```bash
# Complete pipeline with validation
python main.py --config config/config.yaml --environment local

# Skip data validation (faster startup)
python main.py --no-validate --environment kaggle

# Evaluation only (no training)
python main.py --no-train --no-export

# Quick evaluation for development
python main.py --quick-eval

# Verbose logging
python main.py --verbose --environment local
```

## 🏗 Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Financial LLM Framework                      │
├─────────────────────────────────────────────────────────────────┤
│  🎯 Orchestrator Layer                                         │
│  ├─ Pipeline Coordination  ├─ Error Handling                   │
│  └─ Resource Management    └─ Progress Monitoring              │
├─────────────────────────────────────────────────────────────────┤
│  ⚙️ Core Services                                               │
│  ├─ Configuration Mgmt     ├─ Logging System                   │
│  ├─ Memory Management      ├─ Validation Framework             │
│  └─ Statistical Analysis   └─ Performance Monitoring           │
├─────────────────────────────────────────────────────────────────┤
│  🔧 Business Logic                                              │
│  ├─ Data Processing        ├─ Model Adaptation                 │
│  ├─ Training Framework     └─ Evaluation System                │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Data Processing Pipeline**: Advanced preprocessing with validation and instruction formatting
2. **Model Adaptation**: QLoRA-optimized Llama-3 fine-tuning with memory management
3. **Training Framework**: Enterprise-grade training with comprehensive monitoring
4. **Evaluation System**: Multi-metric evaluation with statistical significance testing
5. **Configuration Management**: Environment-aware configuration with validation

### Design Principles

- **Separation of Concerns**: Clear module boundaries with single responsibilities
- **Dependency Injection**: Configurable components with loose coupling
- **Scalability**: Designed for both resource-constrained and high-performance environments
- **Maintainability**: Comprehensive logging, error handling, and documentation
- **Extensibility**: Plugin architecture for custom components and metrics

## ⚙️ Configuration

### Configuration Structure

```yaml
# config/config.yaml
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

evaluation:
  metrics: ["rouge", "bleu", "bert_score", "semantic_similarity"]
  full_eval_samples: 1000
```

### Environment-Specific Overrides

Create environment-specific configurations in `config/environment/`:

- `local.yaml`: Local development settings
- `kaggle.yaml`: Kaggle platform optimizations
- `colab.yaml`: Google Colab configurations
- `production.yaml`: Production deployment settings

### Runtime Configuration

```python
# Update configuration at runtime
orchestrator.config_manager.update_config({
    "training": {"learning_rate": 1e-4},
    "model": {"max_sequence_length": 1024}
})
```

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Architecture Guide](docs/ARCHITECTURE.md)**: Detailed system architecture and design patterns
- **[Theory & Methodology](docs/THEORY_AND_METHODOLOGY.md)**: Mathematical foundations and research insights
- **[Implementation Guide](docs/IMPLEMENTATION_GUIDE.md)**: Step-by-step usage instructions
- **[API Reference](docs/API_REFERENCE.md)**: Complete API documentation
- **[Troubleshooting](docs/TROUBLESHOOTING.md)**: Common issues and solutions

## 📊 Evaluation Metrics

The framework provides comprehensive evaluation capabilities:

### Automated Metrics
- **ROUGE-1, ROUGE-2, ROUGE-L**: N-gram overlap metrics
- **BLEU Score**: Precision-focused evaluation
- **BERTScore**: Contextual embedding similarity
- **Semantic Similarity**: Sentence-level cosine similarity

### Financial Domain Metrics
- **Financial Term Recall**: Domain vocabulary coverage
- **Number Accuracy**: Numerical information preservation
- **Business Context Coherence**: Financial reasoning evaluation

### Statistical Analysis
- **Significance Testing**: Multiple statistical tests (t-test, Wilcoxon, permutation)
- **Effect Size Calculation**: Cohen's d and confidence intervals
- **Bootstrap Analysis**: Robust uncertainty quantification

## 🚀 Performance & Optimization

### Memory Optimization
- **QLoRA**: 4-bit quantization with LoRA adaptation
- **Gradient Checkpointing**: Trade computation for memory
- **Mixed Precision**: FP16/BF16 training support
- **Dynamic Memory Management**: Automatic cleanup and monitoring

### Training Optimization
- **Unsloth Integration**: Accelerated training framework
- **Flash Attention**: Memory-efficient attention computation
- **Gradient Accumulation**: Simulate larger batch sizes
- **Advanced Scheduling**: Cosine annealing with warmup

### Resource Monitoring
- **Real-time Memory Tracking**: GPU and CPU usage monitoring
- **Performance Metrics**: Training speed and efficiency analysis
- **Automatic Resource Cleanup**: Prevent memory leaks

## 🧪 Testing & Quality Assurance

### Data Validation
- **Schema Validation**: Ensure data format consistency
- **Content Quality Checks**: Detect and filter low-quality samples
- **Statistical Analysis**: Distribution and quality metrics
- **Financial Relevance**: Domain-specific content validation

### Model Validation
- **Training Stability**: Monitor for convergence issues
- **Performance Regression**: Compare against baselines
- **Statistical Significance**: Rigorous improvement validation
- **Cross-validation**: Robust performance estimation

## 🌟 Examples

### Financial Analysis Task
```python
# Load fine-tuned model
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="outputs/merged_model",
    torch_dtype=torch.float16
)

# Analyze financial text
prompt = """Analyze the following business call transcript:
"We achieved record quarterly revenue of $2.1B, representing 15% year-over-year growth..."

Key insights:"""

result = generator(prompt, max_new_tokens=256, temperature=0.7)
print(result[0]['generated_text'])
```

### Custom Evaluation Metric
```python
def financial_accuracy_metric(predictions, references):
    """Custom metric for financial term accuracy"""
    financial_terms = ['revenue', 'profit', 'growth', 'EBITDA', 'margin']
    
    def extract_terms(text):
        return set(word.lower() for word in text.split() if word.lower() in financial_terms)
    
    pred_terms = extract_terms(predictions)
    ref_terms = extract_terms(references)
    
    if not ref_terms:
        return 1.0 if not pred_terms else 0.0
    
    return len(pred_terms & ref_terms) / len(ref_terms)

# Add to evaluator
evaluator.custom_metrics['financial_accuracy'] = financial_accuracy_metric
```

## 🔧 Troubleshooting

### Common Issues

**Out of Memory Errors**:
```bash
# Reduce batch size
python main.py --config config/config.yaml
# Edit config to set smaller batch_size and enable gradient_checkpointing
```

**Training Instability**:
```python
# Lower learning rate and add gradient clipping
config_updates = {
    "training": {
        "learning_rate": 1e-4,
        "gradient_clipping": 1.0
    }
}
```

**Data Loading Issues**:
```python
# Verify data format
import pandas as pd
df = pd.read_csv("your_data.csv", sep='|')  # Try different separators
print(df.columns.tolist())
print(df.head())
```

### Performance Optimization

**Improve Training Speed**:
- Enable Flash Attention 2
- Use mixed precision training
- Increase dataloader workers
- Enable model compilation (PyTorch 2.0+)

**Reduce Memory Usage**:
- Decrease batch size, increase gradient accumulation
- Enable gradient checkpointing
- Use CPU offloading for large models

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/your-username/financial-llm-framework.git
cd financial-llm-framework

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

### Code Style
We use Black for code formatting and follow PEP 8 guidelines:
```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face Transformers**: For the excellent model and training infrastructure
- **Unsloth**: For memory-efficient training optimizations
- **SPGISpeech Dataset**: For providing high-quality financial conversation data
- **Meta AI**: For the Llama-3 model family

## 📞 Support

- **Documentation**: Check the `docs/` directory for detailed guides
- **Issues**: Create an issue on GitHub for bug reports or feature requests
- **Discussions**: Use GitHub Discussions for questions and community interaction

## 🗺 Roadmap

### Short-term (v1.1)
- [ ] Support for additional model families (Mistral, CodeLlama)
- [ ] Advanced hyperparameter optimization
- [ ] Distributed training support
- [ ] Enhanced evaluation metrics

### Medium-term (v1.2)
- [ ] Multi-modal financial analysis (text + charts)
- [ ] RLHF integration for human feedback
- [ ] Production inference server
- [ ] Model versioning and A/B testing

### Long-term (v2.0)
- [ ] AutoML for financial model selection
- [ ] Cross-lingual financial understanding
- [ ] Real-time market data integration
- [ ] Federated learning capabilities

---

**Built with ❤️ for the Financial AI Community**

*This framework represents the culmination of extensive research in financial NLP, domain adaptation, and efficient LLM training. We hope it accelerates your financial AI projects and contributes to the advancement of the field.*