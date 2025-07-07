"""Main training orchestration and utilities."""

import time
import math
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..model import GPT
from ..tokenizer import BPETokenizer
from ..config import GPTConfig
from .metrics import MetricsTracker, evaluate_model
from ..utils.checkpoint import CheckpointManager
from ..utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """Main trainer class for GPT model training.
    
    This class orchestrates the entire training process including model
    optimization, metrics tracking, checkpointing, evaluation, and
    early stopping. It supports modern training techniques like mixed
    precision, gradient accumulation, and learning rate scheduling.
    
    Args:
        model: GPT model to train
        tokenizer: Tokenizer instance
        config: Configuration object
        device: Device to run training on
    """
    
    def __init__(
        self,
        model: GPT,
        tokenizer: BPETokenizer,
        config: GPTConfig,
        device: torch.device
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.scheduler_T_max,
            eta_min=config.training.scheduler_eta_min
        )
        
        # Initialize mixed precision scaler if enabled
        self.scaler = torch.cuda.amp.GradScaler() if config.training.mixed_precision else None
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=Path(config.files.output_dir) / "checkpoints",
            max_checkpoints=5
        )
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(device=device)
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_eval_loss = float('inf')
        self.early_stopping_counter = 0
        
        # Warmup scheduler
        self.warmup_steps = config.training.warmup_steps
        self.initial_lr = config.training.learning_rate
        
        logger.info("Trainer initialized successfully")
        logger.info(f"Model has {model.get_num_params():,} parameters")
        logger.info(f"Training configuration: {config.training}")

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        resume_from_checkpoint: bool = True
    ) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader
            resume_from_checkpoint: Whether to resume from existing checkpoint
            
        Returns:
            Dictionary with training results and metrics
        """
        # Resume from checkpoint if available
        if resume_from_checkpoint and self.checkpoint_manager.has_checkpoint():
            self._load_checkpoint()
        
        logger.info("Starting training...")
        logger.info(f"Training for {self.config.training.epochs} epochs")
        logger.info(f"Starting from epoch {self.current_epoch + 1}")
        
        training_start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, self.config.training.epochs):
                self.current_epoch = epoch
                
                # Train for one epoch
                train_metrics = self._train_epoch(train_dataloader)
                
                # Evaluate if validation data is available
                eval_metrics = {}
                if eval_dataloader is not None:
                    eval_metrics = self._evaluate_epoch(eval_dataloader)
                
                # Log epoch results
                self._log_epoch_results(epoch, train_metrics, eval_metrics)
                
                # Save checkpoint
                is_best = False
                if eval_metrics and eval_metrics.get('eval_loss', float('inf')) < self.best_eval_loss:
                    self.best_eval_loss = eval_metrics['eval_loss']
                    is_best = True
                    self.early_stopping_counter = 0
                elif eval_dataloader is not None:
                    self.early_stopping_counter += 1
                
                self._save_checkpoint(train_metrics, eval_metrics, is_best)
                
                # Check early stopping
                if self._should_early_stop():
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        
        total_training_time = time.time() - training_start_time
        
        # Final results
        results = {
            'total_epochs': self.current_epoch + 1,
            'total_steps': self.current_step,
            'best_eval_loss': self.best_eval_loss,
            'total_training_time': total_training_time,
            'final_metrics': self.metrics_tracker.get_summary()
        }
        
        logger.info("Training completed!")
        logger.info(f"Total training time: {total_training_time:.2f} seconds")
        logger.info(f"Best evaluation loss: {self.best_eval_loss:.4f}")
        
        return results

    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Dictionary with epoch training metrics
        """
        self.model.train()
        epoch_start_time = time.time()
        
        # Reset metrics for this epoch
        epoch_metrics = MetricsTracker(device=self.device)
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (input_ids, target_ids) in enumerate(pbar):
            step_start_time = time.time()
            
            # Move to device
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass with optional mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    loss = self._compute_loss(input_ids, target_ids)
            else:
                loss = self._compute_loss(input_ids, target_ids)
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update learning rate
                self._update_learning_rate()
                
                self.current_step += 1
            
            # Update metrics
            step_time = time.time() - step_start_time
            num_tokens = (target_ids != self.tokenizer.pad_token_id).sum().item()
            
            epoch_metrics.update(
                loss=loss.item(),
                batch_size=input_ids.size(0),
                num_tokens=num_tokens,
                learning_rate=self.optimizer.param_groups[0]['lr'],
                step_time=step_time
            )
            
            self.metrics_tracker.update(
                loss=loss.item(),
                batch_size=input_ids.size(0),
                num_tokens=num_tokens,
                learning_rate=self.optimizer.param_groups[0]['lr'],
                step_time=step_time
            )
            
            # Update progress bar
            current_metrics = epoch_metrics.get_current_metrics()
            pbar.set_postfix({
                'loss': f"{current_metrics.get('loss', 0):.4f}",
                'ppl': f"{current_metrics.get('perplexity', 0):.2f}",
                'lr': f"{current_metrics.get('learning_rate', 0):.2e}"
            })
            
            # Periodic evaluation during training
            if (self.current_step % self.config.training.eval_interval == 0 and 
                self.current_step > 0):
                logger.info(f"Step {self.current_step} metrics: {current_metrics}")
        
        epoch_time = time.time() - epoch_start_time
        epoch_metrics_summary = epoch_metrics.get_current_metrics()
        epoch_metrics_summary['epoch_time'] = epoch_time
        
        return epoch_metrics_summary

    def _evaluate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model for one epoch.
        
        Args:
            dataloader: Evaluation data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Running evaluation...")
        
        eval_metrics = evaluate_model(
            model=self.model,
            dataloader=dataloader,
            device=self.device,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        return eval_metrics

    def _compute_loss(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """Compute loss for a batch.
        
        Args:
            input_ids: Input token IDs
            target_ids: Target token IDs
            
        Returns:
            Loss tensor
        """
        logits = self.model(input_ids)
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=self.tokenizer.pad_token_id
        )
        
        return loss

    def _update_learning_rate(self) -> None:
        """Update learning rate with warmup and scheduling."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr_scale = self.current_step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.initial_lr * lr_scale
        else:
            # Use scheduler after warmup
            self.scheduler.step()

    def _save_checkpoint(
        self, 
        train_metrics: Dict[str, float], 
        eval_metrics: Dict[str, float],
        is_best: bool
    ) -> None:
        """Save training checkpoint.
        
        Args:
            train_metrics: Training metrics for this epoch
            eval_metrics: Evaluation metrics for this epoch
            is_best: Whether this is the best checkpoint so far
        """
        metrics = {**train_metrics, **eval_metrics}
        
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            step=self.current_step,
            metrics=metrics,
            is_best=is_best,
            extra_data={
                'best_eval_loss': self.best_eval_loss,
                'early_stopping_counter': self.early_stopping_counter
            }
        )

    def _load_checkpoint(self) -> None:
        """Load training checkpoint."""
        try:
            checkpoint_data = self.checkpoint_manager.load_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=self.device
            )
            
            self.current_epoch = checkpoint_data.get('epoch', 0)
            self.current_step = checkpoint_data.get('step', 0)
            
            extra_data = checkpoint_data.get('extra_data', {})
            self.best_eval_loss = extra_data.get('best_eval_loss', float('inf'))
            self.early_stopping_counter = extra_data.get('early_stopping_counter', 0)
            
            logger.info(f"Resumed training from epoch {self.current_epoch + 1}, "
                       f"step {self.current_step}")
            
        except FileNotFoundError:
            logger.info("No checkpoint found, starting from scratch")

    def _should_early_stop(self) -> bool:
        """Check if early stopping should be triggered.
        
        Returns:
            True if training should stop early
        """
        patience = self.config.training.early_stopping_patience
        return patience > 0 and self.early_stopping_counter >= patience

    def _log_epoch_results(
        self, 
        epoch: int, 
        train_metrics: Dict[str, float], 
        eval_metrics: Dict[str, float]
    ) -> None:
        """Log results for an epoch.
        
        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            eval_metrics: Evaluation metrics
        """
        logger.info(f"Epoch {epoch + 1} completed:")
        logger.info(f"  Train - Loss: {train_metrics.get('loss', 0):.4f}, "
                   f"Perplexity: {train_metrics.get('perplexity', 0):.2f}")
        
        if eval_metrics:
            logger.info(f"  Eval  - Loss: {eval_metrics.get('eval_loss', 0):.4f}, "
                       f"Perplexity: {eval_metrics.get('eval_perplexity', 0):.2f}")
        
        logger.info(f"  Learning Rate: {train_metrics.get('learning_rate', 0):.2e}")
        logger.info(f"  Epoch Time: {train_metrics.get('epoch_time', 0):.2f}s")

    def generate_sample(
        self, 
        prompt: str = "", 
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> str:
        """Generate a sample from the model.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated text
        """
        self.model.eval()
        
        # Encode prompt
        if prompt:
            input_ids = self.tokenizer.encode(prompt)
        else:
            input_ids = [self.tokenizer.bos_token_id]
        
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        
        return generated_text