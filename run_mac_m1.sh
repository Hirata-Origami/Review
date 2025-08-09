#!/bin/bash

# Financial LLM Framework - Mac M1 Pro Startup Script
# Author: Bharath Pranav S

echo "🍎 Financial LLM Framework for Mac M1 Pro"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if environment exists
if conda info --envs | grep -q "financial-llm"; then
    echo "✓ Environment 'financial-llm' found"
else
    echo "📦 Creating conda environment..."
    conda create -n financial-llm python=3.10 -y
    echo "✓ Environment created"
fi

# Activate environment
echo "🔄 Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate financial-llm

# Check if PyTorch is installed
if python -c "import torch" 2>/dev/null; then
    echo "✓ PyTorch already installed"
    python -c "import torch; print(f'  PyTorch version: {torch.__version__}')"
else
    echo "📦 Installing PyTorch for Mac M1..."
    conda install pytorch torchvision torchaudio -c pytorch -y
fi

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check MPS availability
echo "🔍 Checking MPS availability..."
python -c "
import torch
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✓ MPS (Metal Performance Shaders) is available')
    print('  Your Mac M1 Pro can use GPU acceleration!')
else:
    print('❌ MPS not available - will use CPU only')
"

# Run setup test
echo "🧪 Running setup test..."
python test_setup.py

# Check if test passed
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup complete! You can now run:"
    echo "   python main.py --environment mac_m1"
    echo ""
    echo "📚 Documentation: docs/README.md"
    echo "🔧 Troubleshooting: docs/TROUBLESHOOTING.md"
else
    echo ""
    echo "⚠️  Setup test failed. Please check the errors above."
    echo "   See docs/TROUBLESHOOTING.md for help."
fi
