# Troubleshooting Guide

## Common Issues and Solutions

### 1. Framework Doesn't Start / Silent Failure

**Symptoms:**
- Script runs but stops without output
- No error messages
- Execution halts after initialization

**Solutions:**

#### Check Dependencies
```bash
# Run the test script first
python test_setup.py

# Or check manually
python -c "import torch, transformers, datasets, pandas; print('Dependencies OK')"
```

#### Check File Structure
Ensure all required files exist:
```
financial-llm-framework/
├── src/
│   ├── core/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── config/
│   ├── config.yaml
│   └── environment/
├── main.py
└── requirements.txt
```

#### Run Diagnostic Test
```bash
# For Colab specifically
python colab_test.py

# For general debugging
python test_setup.py
```

### 2. Google Colab Issues

**Issue: Silent failure in Colab**

**Solutions:**

1. **Check runtime type:**
   - Use GPU runtime for best performance
   - Runtime → Change runtime type → GPU

2. **Install dependencies explicitly:**
   ```python
   !pip install -r requirements.txt
   !pip install --upgrade torch transformers
   ```

3. **Run diagnostic:**
   ```python
   !python colab_test.py
   ```

4. **Check paths:**
   ```python
   import os
   print("Current directory:", os.getcwd())
   print("Files:", os.listdir("."))
   ```

### 3. Mac M1 Pro Issues

**Issue: Model loading fails on Mac M1**

**Solutions:**

1. **Use MPS environment:**
   ```bash
   python main.py --environment mac_m1
   ```

2. **Install MPS-compatible PyTorch:**
   ```bash
   pip install torch torchvision torchaudio
   ```

3. **Check MPS availability:**
   ```python
   import torch
   print("MPS available:", torch.backends.mps.is_available())
   ```

4. **Reduce memory usage:**
   - The Mac M1 config automatically uses smaller batch sizes
   - Model sequence length is reduced to 1024
   - LoRA rank is reduced to 8

### 4. Memory Issues

**Issue: Out of Memory (OOM) errors**

**Solutions:**

1. **Reduce batch size:**
   ```yaml
   # In config file
   training:
     per_device_train_batch_size: 1
     gradient_accumulation_steps: 16
   ```

2. **Enable gradient checkpointing:**
   ```yaml
   training:
     gradient_checkpointing: true
   ```

3. **Reduce sequence length:**
   ```yaml
   model:
     max_sequence_length: 512
   ```

4. **Use CPU offloading:**
   ```bash
   python main.py --environment mac_m1  # Has CPU optimizations
   ```

### 5. Model Loading Issues

**Issue: Model fails to load**

**Solutions:**

1. **Check internet connection** (models download from HuggingFace)

2. **Try different model:**
   ```yaml
   model:
     base_model: "microsoft/DialoGPT-medium"  # Smaller alternative
   ```

3. **Disable quantization:**
   ```yaml
   model:
     quantization:
       enabled: false
   ```

4. **Use CPU-only mode:**
   ```yaml
   hardware:
     device: "cpu"
   ```

### 6. Data Loading Issues

**Issue: Cannot load dataset**

**Solutions:**

1. **Check file format:**
   ```python
   import pandas as pd
   
   # Try different separators
   df = pd.read_csv("Train dataset.csv", sep='|')  # or sep=','
   print(df.head())
   print(df.columns.tolist())
   ```

2. **Verify data structure:**
   ```python
   # Required columns
   required_cols = ['transcript']
   missing = [col for col in required_cols if col not in df.columns]
   if missing:
       print(f"Missing columns: {missing}")
   ```

3. **Update data paths:**
   ```yaml
   # In config/environment/[env].yaml
   data:
     train_path: "/path/to/your/train.csv"
     val_path: "/path/to/your/val.csv"
   ```

### 7. Training Issues

**Issue: Training doesn't start or fails**

**Solutions:**

1. **Check GPU memory:**
   ```python
   import torch
   if torch.cuda.is_available():
       print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
   ```

2. **Start with smaller dataset:**
   ```yaml
   data:
     max_samples: 1000  # Limit for testing
   ```

3. **Reduce model complexity:**
   ```yaml
   model:
     lora:
       rank: 4      # Smaller rank
       alpha: 8     # Smaller alpha
   ```

### 8. Evaluation Issues

**Issue: Evaluation fails or takes too long**

**Solutions:**

1. **Use quick evaluation:**
   ```bash
   python main.py --quick-eval --no-train
   ```

2. **Reduce evaluation samples:**
   ```yaml
   evaluation:
     quick_eval_samples: 50
     full_eval_samples: 200
   ```

3. **Skip baseline comparison:**
   ```python
   # In code, set baseline_results = None to skip comparison
   ```

### 9. Configuration Issues

**Issue: Configuration errors**

**Solutions:**

1. **Validate YAML syntax:**
   ```python
   import yaml
   with open('config/config.yaml') as f:
       config = yaml.safe_load(f)
   print("Config valid")
   ```

2. **Check environment files:**
   ```bash
   ls config/environment/
   # Should show: local.yaml, kaggle.yaml, colab.yaml, mac_m1.yaml
   ```

3. **Use default config:**
   ```bash
   python main.py  # Uses default local environment
   ```

### 10. Import/Module Issues

**Issue: Module not found errors**

**Solutions:**

1. **Check Python path:**
   ```python
   import sys
   print(sys.path)
   # Should include the src/ directory
   ```

2. **Reinstall requirements:**
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

3. **Check virtual environment:**
   ```bash
   which python
   pip list | grep torch
   ```

## Environment-Specific Guides

### Google Colab Setup
```python
# 1. Clone repository
!git clone https://github.com/Hirata-Origami/financial-llm-framework.git
%cd financial-llm-framework

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Test setup
!python colab_test.py

# 4. Run framework
!python main.py --environment colab
```

### Mac M1 Pro Setup
```bash
# 1. Create conda environment
conda create -n financial-llm python=3.10
conda activate financial-llm

# 2. Install PyTorch for M1
conda install pytorch torchvision torchaudio -c pytorch

# 3. Install other dependencies
pip install -r requirements.txt

# 4. Run with M1 optimizations
python main.py --environment mac_m1
```

### Kaggle Setup
```python
# In Kaggle notebook
import os
os.chdir('/kaggle/working')

# Clone repository
!git clone https://github.com/Hirata-Origami/financial-llm-framework.git
%cd financial-llm-framework

# Install dependencies
!pip install -r requirements.txt

# Run with Kaggle optimizations
!python main.py --environment kaggle
```

## Debug Commands

### Check System Info
```python
import torch
import transformers
import platform

print(f"Platform: {platform.platform()}")
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if hasattr(torch.backends, 'mps'):
    print(f"MPS: {torch.backends.mps.is_available()}")
```

### Test Model Loading
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "microsoft/DialoGPT-small"  # Small test model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Model loaded successfully")
```

### Test Data Loading
```python
import pandas as pd

# Test with small sample data
data = {
    'transcript': [
        "Revenue increased by 15% this quarter.",
        "Market conditions remain favorable.",
        "Investment in new technologies shows promise."
    ]
}
df = pd.DataFrame(data)
print("Data loading test passed")
```

## Getting Help

If you're still experiencing issues:

1. **Run the diagnostic tests:**
   ```bash
   python test_setup.py
   python colab_test.py  # For Colab
   ```

2. **Check the logs:**
   ```bash
   ls outputs/logs/
   cat outputs/logs/financial_llm_*.log
   ```

3. **Create a minimal reproduction:**
   ```python
   # Minimal test case
   from src.core.config import ConfigManager
   config = ConfigManager("config/config.yaml", "local")
   print("Basic setup works")
   ```

4. **Check GitHub issues:** Look for similar problems in the repository issues

5. **Create an issue:** Include the output of diagnostic tests and error messages
