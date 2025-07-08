"""Checkpoint management utilities."""

from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import torch

from ..utils.logging import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Manages model checkpoints with automatic cleanup and resumption.

    This class handles saving and loading model checkpoints, including
    model state, optimizer state, scheduler state, and training metrics.
    It supports automatic cleanup of old checkpoints and easy resumption
    of training from the latest checkpoint.

    Args:
        checkpoint_dir: Directory to store checkpoints
        max_checkpoints: Maximum number of checkpoints to keep (0 = keep all)
        prefix: Prefix for checkpoint filenames
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        prefix: str = "checkpoint",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.prefix = prefix

        logger.debug(
            f"Initialized CheckpointManager: dir={checkpoint_dir}, "
            f"max_checkpoints={max_checkpoints}"
        )

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Optional learning rate scheduler
            epoch: Current epoch number
            step: Current step number
            metrics: Training metrics to save
            is_best: Whether this is the best checkpoint so far
            extra_data: Additional data to save

        Returns:
            Path to saved checkpoint
        """
        checkpoint_data = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics or {},
            "extra_data": extra_data or {},
        }

        if scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()

        # Create checkpoint filename
        checkpoint_name = f"{self.prefix}_epoch_{epoch}_step_{step}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Save as best checkpoint if specified
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint_data, best_path)
            logger.info(f"Best checkpoint saved: {best_path}")

        # Save as latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint_data, latest_path)

        # Cleanup old checkpoints
        if self.max_checkpoints > 0:
            self._cleanup_old_checkpoints()

        return checkpoint_path

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Load a checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            checkpoint_path: Specific checkpoint to load (if None, loads latest)
            device: Device to load checkpoint on

        Returns:
            Dictionary containing loaded checkpoint data

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "latest.pth"
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")

        # Load checkpoint data
        checkpoint_data = torch.load(checkpoint_path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint_data["model_state_dict"])

        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

        # Load scheduler state if provided
        if scheduler is not None and "scheduler_state_dict" in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

        logger.info(
            f"Checkpoint loaded successfully from epoch {checkpoint_data.get('epoch', 0)}, "
            f"step {checkpoint_data.get('step', 0)}"
        )

        return checkpoint_data

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to the latest checkpoint.

        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        latest_path = self.checkpoint_dir / "latest.pth"
        return latest_path if latest_path.exists() else None

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to the best checkpoint.

        Returns:
            Path to best checkpoint or None if no best checkpoint exists
        """
        best_path = self.checkpoint_dir / "best_model.pth"
        return best_path if best_path.exists() else None

    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoints.

        Returns:
            List of checkpoint paths sorted by modification time
        """
        pattern = f"{self.prefix}_epoch_*.pth"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        return sorted(checkpoints, key=lambda p: p.stat().st_mtime)

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoints = self.list_checkpoints()

        if len(checkpoints) > self.max_checkpoints:
            # Keep the most recent max_checkpoints, remove the rest
            to_remove = checkpoints[: -self.max_checkpoints]
            for checkpoint_path in to_remove:
                try:
                    checkpoint_path.unlink()
                    logger.debug(f"Removed old checkpoint: {checkpoint_path}")
                except OSError as e:
                    logger.warning(
                        f"Failed to remove checkpoint {checkpoint_path}: {e}"
                    )

    def has_checkpoint(self) -> bool:
        """Check if any checkpoints exist.

        Returns:
            True if checkpoints exist, False otherwise
        """
        return self.get_latest_checkpoint() is not None
