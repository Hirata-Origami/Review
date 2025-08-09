"""
Training Callbacks Module

Advanced callbacks for training monitoring and optimization.
"""

from transformers.trainer_callback import TrainerCallback
from .memory import MemoryManager
from .logger import get_logger

logger = get_logger(__name__)

class MetricsCallback(TrainerCallback):
    """Callback for enhanced metrics logging."""
    
    def __init__(self):
        self.step_count = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs."""
        if logs:
            self.step_count += 1
            logger.info(f"Training metrics at step {self.step_count}: {logs}")

class MemoryCallback(TrainerCallback):
    """Callback for memory monitoring and cleanup."""
    
    def __init__(self, memory_manager: MemoryManager, cleanup_frequency: int = 100):
        self.memory_manager = memory_manager
        self.cleanup_frequency = cleanup_frequency
        self.step_count = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        self.step_count += 1
        
        if self.step_count % self.cleanup_frequency == 0:
            stats = self.memory_manager.cleanup_memory()
            logger.debug(f"Memory cleanup at step {self.step_count}: {stats}")

class TimeCallback(TrainerCallback):
    """Callback for timing monitoring."""
    
    def __init__(self):
        self.start_time = None
        self.step_times = []
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        import time
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each step."""
        import time
        if self.start_time:
            step_time = time.time() - self.start_time
            self.step_times.append(step_time)
            self.start_time = time.time()
