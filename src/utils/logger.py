"""
Advanced Logging Module

Comprehensive logging system for financial LLM training framework
with structured logging, performance monitoring, and integration capabilities.
"""

import logging
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime
import traceback
from contextlib import contextmanager

from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install as install_rich_traceback

# Install rich traceback handler for better error formatting
install_rich_traceback(show_locals=True)

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with JSON output."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Create base log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry['extra'] = record.extra_fields
        
        return json.dumps(log_entry, default=str)

class PerformanceLogger:
    """Logger for performance metrics and monitoring."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timers = {}
    
    @contextmanager
    def timer(self, operation_name: str):
        """Context manager for timing operations."""
        start_time = time.time()
        self.logger.info(f"Starting operation: {operation_name}")
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.logger.info(f"Completed operation: {operation_name} in {duration:.3f}s")
            self.timers[operation_name] = duration
    
    def log_memory_usage(self, stage: str, memory_stats: Dict[str, Any]):
        """Log memory usage statistics."""
        self.logger.info(
            f"Memory usage at {stage}",
            extra={'extra_fields': {'stage': stage, 'memory_stats': memory_stats}}
        )
    
    def log_training_metrics(self, epoch: int, step: int, metrics: Dict[str, float]):
        """Log training metrics."""
        self.logger.info(
            f"Training metrics - Epoch {epoch}, Step {step}",
            extra={'extra_fields': {'epoch': epoch, 'step': step, 'metrics': metrics}}
        )
    
    def log_evaluation_results(self, model_name: str, results: Dict[str, Any]):
        """Log evaluation results."""
        self.logger.info(
            f"Evaluation results for {model_name}",
            extra={'extra_fields': {'model_name': model_name, 'results': results}}
        )

class LoggerManager:
    """Centralized logger management for the framework."""
    
    def __init__(self, 
                 log_dir: Optional[Union[str, Path]] = None,
                 console_level: int = logging.INFO,
                 file_level: int = logging.DEBUG,
                 structured_logging: bool = False):
        """
        Initialize logger manager.
        
        Args:
            log_dir: Directory for log files
            console_level: Console logging level
            file_level: File logging level
            structured_logging: Enable structured JSON logging
        """
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.console_level = console_level
        self.file_level = file_level
        self.structured_logging = structured_logging
        
        self.console = Console()
        self.loggers = {}
        
        # Setup root logger
        self._setup_root_logger()
    
    def _setup_root_logger(self):
        """Setup root logger configuration."""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler with Rich formatting
        console_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True
        )
        console_handler.setLevel(self.console_level)
        
        if not self.structured_logging:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
        
        root_logger.addHandler(console_handler)
        
        # File handlers
        self._setup_file_handlers(root_logger)
    
    def _setup_file_handlers(self, logger: logging.Logger):
        """Setup file handlers for different log levels."""
        # General log file
        general_handler = logging.FileHandler(
            self.log_dir / f"financial_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        general_handler.setLevel(self.file_level)
        
        # Error log file
        error_handler = logging.FileHandler(self.log_dir / "errors.log")
        error_handler.setLevel(logging.ERROR)
        
        # Choose formatter
        if self.structured_logging:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        
        general_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)
        
        logger.addHandler(general_handler)
        logger.addHandler(error_handler)
        
        # Performance log file (structured)
        perf_handler = logging.FileHandler(self.log_dir / "performance.jsonl")
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(StructuredFormatter())
        
        # Create performance logger
        perf_logger = logging.getLogger('performance')
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)
        perf_logger.propagate = False
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the given name."""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def get_performance_logger(self, name: str) -> PerformanceLogger:
        """Get performance logger for the given name."""
        base_logger = self.get_logger(f"performance.{name}")
        return PerformanceLogger(base_logger)
    
    def set_level(self, level: Union[int, str]):
        """Set logging level for all loggers."""
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        for logger in self.loggers.values():
            logger.setLevel(level)
    
    def log_system_info(self):
        """Log system information at startup."""
        try:
            import torch
            import platform
            import psutil
        except ImportError as e:
            logger = self.get_logger('system')
            logger.warning(f"Cannot import required modules for system info: {e}")
            return
        
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                system_info[f'gpu_{i}'] = {
                    'name': gpu_props.name,
                    'memory_total_gb': gpu_props.total_memory / (1024**3),
                    'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
                }
        
        logger = self.get_logger('system')
        logger.info(
            "System information logged",
            extra={'extra_fields': {'system_info': system_info}}
        )
    
    def create_run_directory(self, run_name: Optional[str] = None) -> Path:
        """Create a directory for this training run."""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        run_dir = self.log_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        logger = self.get_logger('run_manager')
        logger.info(f"Created run directory: {run_dir}")
        
        return run_dir
    
    def setup_experiment_logging(self, experiment_name: str, config: Dict[str, Any]):
        """Setup logging for a specific experiment."""
        exp_logger = self.get_logger(f"experiment.{experiment_name}")
        
        exp_logger.info(
            f"Starting experiment: {experiment_name}",
            extra={'extra_fields': {'experiment_config': config}}
        )
        
        return exp_logger
    
    def cleanup(self):
        """Cleanup logging resources."""
        # Close all file handlers
        for logger in self.loggers.values():
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
        
        # Close root logger handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()

# Global logger manager instance
_logger_manager: Optional[LoggerManager] = None

def initialize_logging(log_dir: Optional[Union[str, Path]] = None,
                      console_level: int = logging.INFO,
                      file_level: int = logging.DEBUG,
                      structured_logging: bool = False) -> LoggerManager:
    """Initialize global logging system."""
    global _logger_manager
    
    _logger_manager = LoggerManager(
        log_dir=log_dir,
        console_level=console_level,
        file_level=file_level,
        structured_logging=structured_logging
    )
    
    # Log system info
    _logger_manager.log_system_info()
    
    return _logger_manager

def get_logger(name: str) -> logging.Logger:
    """Get logger instance (initializes logging if not already done)."""
    global _logger_manager
    
    if _logger_manager is None:
        _logger_manager = initialize_logging()
    
    return _logger_manager.get_logger(name)

def get_performance_logger(name: str) -> PerformanceLogger:
    """Get performance logger instance."""
    global _logger_manager
    
    if _logger_manager is None:
        _logger_manager = initialize_logging()
    
    return _logger_manager.get_performance_logger(name)

def get_logger_manager() -> LoggerManager:
    """Get the global logger manager instance."""
    global _logger_manager
    
    if _logger_manager is None:
        _logger_manager = initialize_logging()
    
    return _logger_manager

# Context manager for logging errors
@contextmanager
def log_errors(logger: Optional[logging.Logger] = None, 
               operation_name: str = "operation",
               reraise: bool = True):
    """Context manager for automatic error logging."""
    if logger is None:
        logger = get_logger("error_handler")
    
    try:
        yield
    except Exception as e:
        logger.error(
            f"Error in {operation_name}: {str(e)}",
            exc_info=True,
            extra={'extra_fields': {'operation': operation_name}}
        )
        
        if reraise:
            raise

# Decorators for automatic logging
def log_function_call(logger: Optional[logging.Logger] = None):
    """Decorator to log function calls and execution time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_logger = logger or get_logger(func.__module__)
            
            with get_performance_logger(func.__module__).timer(func.__name__):
                try:
                    result = func(*args, **kwargs)
                    func_logger.debug(f"Function {func.__name__} completed successfully")
                    return result
                except Exception as e:
                    func_logger.error(f"Function {func.__name__} failed: {str(e)}", exc_info=True)
                    raise
        
        return wrapper
    return decorator