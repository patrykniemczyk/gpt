"""Metrics tracking and evaluation utilities."""

import time
import math
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import torch
import torch.nn.functional as F
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MetricsTracker:
    """Tracks and computes training and evaluation metrics.
    
    This class provides utilities for tracking various metrics during
    training including loss, perplexity, learning rate, and timing
    information. It supports smoothed metrics and efficient computation.
    
    Args:
        window_size: Size of rolling window for smoothed metrics
        device: Device for tensor computations
    """
    
    def __init__(self, window_size: int = 100, device: Optional[torch.device] = None) -> None:
        self.window_size = window_size
        self.device = device or torch.device('cpu')
        
        # Rolling windows for smoothed metrics
        self.losses = deque(maxlen=window_size)
        self.perplexities = deque(maxlen=window_size)
        
        # Cumulative metrics
        self.total_loss = 0.0
        self.total_samples = 0
        self.total_tokens = 0
        
        # Timing
        self.start_time = time.time()
        self.step_times = deque(maxlen=window_size)
        
        # Learning rate tracking
        self.learning_rates = deque(maxlen=window_size)
        
        # Step counting
        self.step_count = 0
        
        logger.debug(f"Initialized MetricsTracker with window_size={window_size}")

    def update(
        self,
        loss: float,
        batch_size: int,
        num_tokens: int,
        learning_rate: Optional[float] = None,
        step_time: Optional[float] = None
    ) -> None:
        """Update metrics with new batch results.
        
        Args:
            loss: Loss value for this batch
            batch_size: Number of samples in batch
            num_tokens: Number of tokens processed
            learning_rate: Current learning rate
            step_time: Time taken for this step
        """
        self.step_count += 1
        
        # Update loss tracking
        self.losses.append(loss)
        self.total_loss += loss * batch_size
        self.total_samples += batch_size
        self.total_tokens += num_tokens
        
        # Update perplexity
        perplexity = math.exp(loss) if loss < 10 else float('inf')
        self.perplexities.append(perplexity)
        
        # Update learning rate
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        
        # Update timing
        if step_time is not None:
            self.step_times.append(step_time)

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current smoothed metrics.
        
        Returns:
            Dictionary of current metric values
        """
        metrics = {}
        
        # Loss metrics
        if self.losses:
            metrics['loss'] = sum(self.losses) / len(self.losses)
            metrics['loss_latest'] = self.losses[-1]
        
        # Perplexity metrics
        if self.perplexities:
            valid_perplexities = [p for p in self.perplexities if math.isfinite(p)]
            if valid_perplexities:
                metrics['perplexity'] = sum(valid_perplexities) / len(valid_perplexities)
                metrics['perplexity_latest'] = self.perplexities[-1]
        
        # Learning rate
        if self.learning_rates:
            metrics['learning_rate'] = self.learning_rates[-1]
        
        # Timing metrics
        if self.step_times:
            metrics['step_time'] = sum(self.step_times) / len(self.step_times)
            metrics['steps_per_second'] = 1.0 / metrics['step_time']
        
        # Cumulative metrics
        if self.total_samples > 0:
            metrics['avg_loss'] = self.total_loss / self.total_samples
        
        # Tokens per second
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            metrics['tokens_per_second'] = self.total_tokens / elapsed_time
        
        metrics['step'] = self.step_count
        metrics['total_tokens'] = self.total_tokens
        
        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all metrics.
        
        Returns:
            Dictionary with detailed metrics summary
        """
        current = self.get_current_metrics()
        
        summary = {
            'current': current,
            'totals': {
                'steps': self.step_count,
                'samples': self.total_samples,
                'tokens': self.total_tokens,
                'elapsed_time': time.time() - self.start_time
            }
        }
        
        # Add distribution info for losses
        if len(self.losses) > 1:
            losses_list = list(self.losses)
            summary['loss_distribution'] = {
                'min': min(losses_list),
                'max': max(losses_list),
                'std': self._calculate_std(losses_list)
            }
        
        return summary

    def reset(self) -> None:
        """Reset all metrics."""
        self.losses.clear()
        self.perplexities.clear()
        self.learning_rates.clear()
        self.step_times.clear()
        
        self.total_loss = 0.0
        self.total_samples = 0
        self.total_tokens = 0
        self.step_count = 0
        self.start_time = time.time()
        
        logger.debug("Metrics tracker reset")

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    pad_token_id: int = 0
) -> Dict[str, float]:
    """Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on
        max_batches: Maximum number of batches to evaluate (None = all)
        pad_token_id: Padding token ID to ignore in loss computation
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0
    
    start_time = time.time()
    
    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break
        
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Forward pass
        logits = model(input_ids)
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=pad_token_id,
            reduction='sum'
        )
        
        # Count non-padding tokens
        non_pad_tokens = (target_ids != pad_token_id).sum().item()
        
        total_loss += loss.item()
        total_tokens += non_pad_tokens
        total_batches += 1
    
    # Compute metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
    eval_time = time.time() - start_time
    
    metrics = {
        'eval_loss': avg_loss,
        'eval_perplexity': perplexity,
        'eval_tokens': total_tokens,
        'eval_batches': total_batches,
        'eval_time': eval_time,
        'eval_tokens_per_second': total_tokens / eval_time if eval_time > 0 else 0
    }
    
    logger.info(f"Evaluation complete: loss={avg_loss:.4f}, perplexity={perplexity:.2f}, "
               f"tokens={total_tokens}, time={eval_time:.2f}s")
    
    return metrics


@torch.no_grad()
def compute_token_accuracy(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    pad_token_id: int = 0,
    max_batches: Optional[int] = None
) -> float:
    """Compute token-level accuracy.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on
        pad_token_id: Padding token ID to ignore
        max_batches: Maximum number of batches to evaluate
        
    Returns:
        Token-level accuracy as a float
    """
    model.eval()
    
    correct_tokens = 0
    total_tokens = 0
    
    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break
        
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Forward pass
        logits = model(input_ids)
        predictions = torch.argmax(logits, dim=-1)
        
        # Create mask for non-padding tokens
        mask = target_ids != pad_token_id
        
        # Count correct predictions
        correct = (predictions == target_ids) & mask
        correct_tokens += correct.sum().item()
        total_tokens += mask.sum().item()
    
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    logger.info(f"Token accuracy: {accuracy:.4f} ({correct_tokens}/{total_tokens})")
    
    return accuracy