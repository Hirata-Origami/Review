"""
Memory Management Module

Advanced memory monitoring and optimization utilities for efficient
GPU and CPU memory usage during LLM training and inference.
"""

import gc
import threading
import time
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from contextlib import contextmanager
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from .logger import get_logger

logger = get_logger(__name__)

@dataclass
class MemoryStats:
    """Container for memory statistics"""
    # CPU Memory
    cpu_used_gb: float
    cpu_available_gb: float
    cpu_percent: float
    
    # GPU Memory (if available)
    gpu_allocated_gb: Optional[float] = None
    gpu_reserved_gb: Optional[float] = None
    gpu_total_gb: Optional[float] = None
    gpu_free_gb: Optional[float] = None
    gpu_percent: Optional[float] = None
    
    # Model-specific
    model_memory_gb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cpu_used_gb': self.cpu_used_gb,
            'cpu_available_gb': self.cpu_available_gb,
            'cpu_percent': self.cpu_percent,
            'gpu_allocated_gb': self.gpu_allocated_gb,
            'gpu_reserved_gb': self.gpu_reserved_gb,
            'gpu_total_gb': self.gpu_total_gb,
            'gpu_free_gb': self.gpu_free_gb,
            'gpu_percent': self.gpu_percent,
            'model_memory_gb': self.model_memory_gb
        }

class MemoryMonitor:
    """Real-time memory monitoring with alerting capabilities."""
    
    def __init__(self, 
                 check_interval: float = 10.0,
                 memory_threshold: float = 0.9,
                 gpu_threshold: float = 0.9):
        """
        Initialize memory monitor.
        
        Args:
            check_interval: Monitoring interval in seconds
            memory_threshold: Alert threshold for CPU memory (0-1)
            gpu_threshold: Alert threshold for GPU memory (0-1)
        """
        self.check_interval = check_interval
        self.memory_threshold = memory_threshold
        self.gpu_threshold = gpu_threshold
        
        self.monitoring = False
        self.monitor_thread = None
        self.stats_history: List[MemoryStats] = []
        self.max_history = 1000
        
    def start_monitoring(self):
        """Start background memory monitoring."""
        if self.monitoring:
            logger.warning("Memory monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                stats = self.get_memory_stats()
                self.stats_history.append(stats)
                
                # Trim history
                if len(self.stats_history) > self.max_history:
                    self.stats_history = self.stats_history[-self.max_history:]
                
                # Check thresholds
                self._check_thresholds(stats)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(self.check_interval)
    
    def _check_thresholds(self, stats: MemoryStats):
        """Check memory thresholds and log warnings."""
        # CPU memory check
        if stats.cpu_percent > self.memory_threshold:
            logger.warning(
                f"High CPU memory usage: {stats.cpu_percent:.1%} "
                f"({stats.cpu_used_gb:.1f}GB / {stats.cpu_used_gb + stats.cpu_available_gb:.1f}GB)"
            )
        
        # GPU memory check
        if stats.gpu_percent and stats.gpu_percent > self.gpu_threshold:
            logger.warning(
                f"High GPU memory usage: {stats.gpu_percent:.1%} "
                f"({stats.gpu_allocated_gb:.1f}GB / {stats.gpu_total_gb:.1f}GB)"
            )
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        # CPU memory
        cpu_memory = psutil.virtual_memory()
        cpu_used_gb = cpu_memory.used / (1024**3)
        cpu_available_gb = cpu_memory.available / (1024**3)
        cpu_percent = cpu_memory.percent / 100.0
        
        # GPU memory
        gpu_allocated_gb = None
        gpu_reserved_gb = None
        gpu_total_gb = None
        gpu_free_gb = None
        gpu_percent = None
        
        if torch.cuda.is_available():
            try:
                gpu_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
                gpu_reserved_gb = torch.cuda.memory_reserved() / (1024**3)
                
                # Get total GPU memory for current device
                device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
                gpu_total_gb = device_props.total_memory / (1024**3)
                gpu_free_gb = gpu_total_gb - gpu_allocated_gb
                gpu_percent = gpu_allocated_gb / gpu_total_gb
                
            except Exception as e:
                logger.debug(f"Failed to get GPU memory stats: {e}")
        
        return MemoryStats(
            cpu_used_gb=cpu_used_gb,
            cpu_available_gb=cpu_available_gb,
            cpu_percent=cpu_percent,
            gpu_allocated_gb=gpu_allocated_gb,
            gpu_reserved_gb=gpu_reserved_gb,
            gpu_total_gb=gpu_total_gb,
            gpu_free_gb=gpu_free_gb,
            gpu_percent=gpu_percent
        )
    
    def get_peak_memory(self) -> Optional[MemoryStats]:
        """Get peak memory usage from history."""
        if not self.stats_history:
            return None
        
        # Find peak GPU memory
        peak_gpu = max(
            (s for s in self.stats_history if s.gpu_allocated_gb is not None),
            key=lambda s: s.gpu_allocated_gb,
            default=None
        )
        
        # Find peak CPU memory
        peak_cpu = max(self.stats_history, key=lambda s: s.cpu_used_gb)
        
        # Combine peak stats
        if peak_gpu:
            return MemoryStats(
                cpu_used_gb=peak_cpu.cpu_used_gb,
                cpu_available_gb=peak_cpu.cpu_available_gb,
                cpu_percent=peak_cpu.cpu_percent,
                gpu_allocated_gb=peak_gpu.gpu_allocated_gb,
                gpu_reserved_gb=peak_gpu.gpu_reserved_gb,
                gpu_total_gb=peak_gpu.gpu_total_gb,
                gpu_free_gb=peak_gpu.gpu_free_gb,
                gpu_percent=peak_gpu.gpu_percent
            )
        else:
            return peak_cpu

class MemoryManager:
    """Advanced memory management for LLM training and inference."""
    
    def __init__(self, 
                 auto_cleanup: bool = True,
                 cleanup_threshold: float = 0.8,
                 enable_monitoring: bool = False):
        """
        Initialize memory manager.
        
        Args:
            auto_cleanup: Enable automatic memory cleanup
            cleanup_threshold: Memory threshold for automatic cleanup
            enable_monitoring: Enable background monitoring
        """
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, memory monitoring will be limited")
            enable_monitoring = False
            
        self.auto_cleanup = auto_cleanup
        self.cleanup_threshold = cleanup_threshold
        
        self.monitor = MemoryMonitor() if enable_monitoring and PSUTIL_AVAILABLE else None
        self.cleanup_count = 0
        
        if enable_monitoring and self.monitor:
            try:
                self.monitor.start_monitoring()
            except Exception as e:
                logger.warning(f"Failed to start memory monitoring: {e}")
                self.monitor = None
        
        logger.info(f"Memory manager initialized (auto_cleanup={auto_cleanup})")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics as dict."""
        if self.monitor:
            stats = self.monitor.get_memory_stats()
        else:
            stats = self._get_current_stats()
        
        return {
            'used_gb': stats.cpu_used_gb,
            'used_mb': stats.cpu_used_gb * 1024,
            'available_gb': stats.cpu_available_gb,
            'percent': stats.cpu_percent,
            'gpu_allocated_gb': stats.gpu_allocated_gb or 0.0,
            'gpu_percent': stats.gpu_percent or 0.0
        }
    
    def _get_current_stats(self) -> MemoryStats:
        """Get current memory stats without monitor."""
        # CPU memory
        cpu_memory = psutil.virtual_memory()
        cpu_used_gb = cpu_memory.used / (1024**3)
        cpu_available_gb = cpu_memory.available / (1024**3)
        cpu_percent = cpu_memory.percent / 100.0
        
        # GPU memory
        gpu_allocated_gb = None
        gpu_reserved_gb = None
        gpu_total_gb = None
        gpu_free_gb = None
        gpu_percent = None
        
        if torch.cuda.is_available():
            try:
                gpu_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
                gpu_reserved_gb = torch.cuda.memory_reserved() / (1024**3)
                
                device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
                gpu_total_gb = device_props.total_memory / (1024**3)
                gpu_free_gb = gpu_total_gb - gpu_allocated_gb
                gpu_percent = gpu_allocated_gb / gpu_total_gb
                
            except Exception:
                pass
        
        return MemoryStats(
            cpu_used_gb=cpu_used_gb,
            cpu_available_gb=cpu_available_gb,
            cpu_percent=cpu_percent,
            gpu_allocated_gb=gpu_allocated_gb,
            gpu_reserved_gb=gpu_reserved_gb,
            gpu_total_gb=gpu_total_gb,
            gpu_free_gb=gpu_free_gb,
            gpu_percent=gpu_percent
        )
    
    def cleanup_memory(self, force: bool = False) -> Dict[str, float]:
        """
        Perform memory cleanup.
        
        Args:
            force: Force cleanup regardless of threshold
            
        Returns:
            Memory stats before and after cleanup
        """
        stats_before = self.get_memory_stats()
        
        # Check if cleanup is needed
        if not force and self.auto_cleanup:
            gpu_needs_cleanup = (stats_before.get('gpu_percent', 0) > self.cleanup_threshold)
            cpu_needs_cleanup = (stats_before.get('percent', 0) > self.cleanup_threshold)
            
            if not (gpu_needs_cleanup or cpu_needs_cleanup):
                return {'cleanup_performed': False, **stats_before}
        
        # Perform cleanup
        logger.debug("Performing memory cleanup...")
        
        # Python garbage collection
        collected = gc.collect()
        
        # PyTorch GPU cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Get stats after cleanup
        stats_after = self.get_memory_stats()
        
        self.cleanup_count += 1
        
        freed_mb = (stats_before.get('used_mb', 0) - stats_after.get('used_mb', 0))
        gpu_freed_gb = (stats_before.get('gpu_allocated_gb', 0) - stats_after.get('gpu_allocated_gb', 0))
        
        logger.debug(
            f"Memory cleanup #{self.cleanup_count} completed. "
            f"Freed: {freed_mb:.1f}MB CPU, {gpu_freed_gb:.3f}GB GPU. "
            f"Collected: {collected} objects"
        )
        
        return {
            'cleanup_performed': True,
            'objects_collected': collected,
            'cpu_freed_mb': freed_mb,
            'gpu_freed_gb': gpu_freed_gb,
            **stats_after
        }
    
    def get_model_memory_usage(self, model: torch.nn.Module) -> float:
        """
        Estimate model memory usage in MB.
        
        Args:
            model: PyTorch model
            
        Returns:
            Estimated memory usage in MB
        """
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size_mb = (param_size + buffer_size) / (1024 ** 2)
        return total_size_mb
    
    def optimize_memory_for_training(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Apply memory optimizations for training.
        
        Args:
            model: Model to optimize
            
        Returns:
            Dictionary of applied optimizations
        """
        optimizations = {}
        
        # Enable gradient checkpointing if available
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            optimizations['gradient_checkpointing'] = True
            logger.info("Enabled gradient checkpointing")
        
        # Set model to train mode
        model.train()
        
        # Apply memory efficient attention if available
        if torch.cuda.is_available() and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            optimizations['efficient_attention'] = True
            logger.info("Using memory efficient attention")
        
        # Cleanup before training
        cleanup_stats = self.cleanup_memory(force=True)
        optimizations['initial_cleanup'] = cleanup_stats
        
        return optimizations
    
    def optimize_memory_for_inference(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Apply memory optimizations for inference.
        
        Args:
            model: Model to optimize
            
        Returns:
            Dictionary of applied optimizations
        """
        optimizations = {}
        
        # Set model to eval mode
        model.eval()
        
        # Disable gradients
        torch.set_grad_enabled(False)
        optimizations['gradients_disabled'] = True
        
        # Move to half precision if on GPU
        if torch.cuda.is_available() and next(model.parameters()).device.type == 'cuda':
            if next(model.parameters()).dtype == torch.float32:
                model.half()
                optimizations['half_precision'] = True
                logger.info("Converted model to half precision for inference")
        
        # Cleanup
        cleanup_stats = self.cleanup_memory(force=True)
        optimizations['cleanup'] = cleanup_stats
        
        return optimizations
    
    @contextmanager
    def memory_tracking(self, operation_name: str = "operation"):
        """
        Context manager for tracking memory usage during an operation.
        
        Args:
            operation_name: Name of the operation being tracked
        """
        logger.debug(f"Starting memory tracking for: {operation_name}")
        
        # Get initial stats
        stats_before = self.get_memory_stats()
        
        try:
            yield stats_before
        finally:
            # Get final stats
            stats_after = self.get_memory_stats()
            
            # Calculate differences
            cpu_diff = stats_after['used_gb'] - stats_before['used_gb']
            gpu_diff = stats_after.get('gpu_allocated_gb', 0) - stats_before.get('gpu_allocated_gb', 0)
            
            logger.debug(
                f"Memory tracking for {operation_name} completed. "
                f"CPU change: {cpu_diff:+.3f}GB, GPU change: {gpu_diff:+.3f}GB"
            )
    
    def set_memory_fraction(self, fraction: float):
        """
        Set GPU memory fraction (for PyTorch).
        
        Args:
            fraction: Fraction of GPU memory to use (0.0 to 1.0)
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, cannot set memory fraction")
            return
        
        if not (0.0 < fraction <= 1.0):
            raise ValueError("Memory fraction must be between 0.0 and 1.0")
        
        try:
            torch.cuda.set_per_process_memory_fraction(fraction)
            logger.info(f"Set GPU memory fraction to {fraction:.2%}")
        except Exception as e:
            logger.error(f"Failed to set memory fraction: {e}")
    
    def get_memory_summary(self) -> str:
        """Get a formatted memory summary."""
        stats = self.get_memory_stats()
        
        summary = f"""
Memory Summary:
  CPU Usage: {stats['used_gb']:.2f}GB ({stats['percent']:.1%})
  CPU Available: {stats['available_gb']:.2f}GB
        """
        
        if torch.cuda.is_available():
            summary += f"""
  GPU Allocated: {stats['gpu_allocated_gb']:.2f}GB ({stats['gpu_percent']:.1%})
  Cleanup Count: {self.cleanup_count}
            """
        
        return summary.strip()
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.monitor:
            self.monitor.stop_monitoring()
        
        # Final cleanup
        try:
            self.cleanup_memory(force=True)
        except Exception:
            pass