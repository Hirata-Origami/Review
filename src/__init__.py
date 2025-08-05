"""
Financial Domain LLM Fine-tuning Framework

A comprehensive framework for fine-tuning large language models on financial domain data,
specifically designed for the SPGISpeech dataset containing business call transcripts.

This framework provides:
- Enterprise-grade architecture with proper separation of concerns
- Advanced training strategies using QLoRA/Unsloth for memory efficiency
- Comprehensive evaluation metrics and comparative analysis
- Production-ready deployment utilities
- Extensive monitoring and logging capabilities

Author: Bharath Pranav S
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Bharath Pranav S"

from .core import *
from .data import *
from .models import *
from .training import *
from .evaluation import *
from .utils import *