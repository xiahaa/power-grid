"""
Utility functions for training and evaluation

Author: Your Name
Date: 2026-01-18
"""

import torch
import numpy as np
from typing import Dict, Tuple
import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str
):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: str = 'cuda'
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from epoch {epoch}")
    return model, optimizer, epoch, loss


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience: int = 30, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


class MetricsCalculator:
    """Calculate evaluation metrics"""

    @staticmethod
    def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Root Mean Squared Error"""
        return torch.sqrt(((pred - target) ** 2).mean()).item()

    @staticmethod
    def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Mean Absolute Error"""
        return torch.abs(pred - target).mean().item()

    @staticmethod
    def mape(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> float:
        """Mean Absolute Percentage Error"""
        return (torch.abs((pred - target) / (target + epsilon)) * 100).mean().item()

    @staticmethod
    def compute_all_metrics(
        pred_states: Dict[str, torch.Tensor],
        true_states: Dict[str, torch.Tensor],
        pred_params: Dict[str, torch.Tensor],
        true_params: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute all metrics"""
        metrics = {}

        # State metrics
        for key in ['v_mag', 'v_ang']:
            if key in pred_states and key in true_states:
                metrics[f'{key}_rmse'] = MetricsCalculator.rmse(
                    pred_states[key], true_states[key]
                )
                metrics[f'{key}_mae'] = MetricsCalculator.mae(
                    pred_states[key], true_states[key]
                )
                metrics[f'{key}_mape'] = MetricsCalculator.mape(
                    pred_states[key], true_states[key]
                )

        # Parameter metrics
        for key in ['r_line', 'x_line']:
            if key in pred_params and key in true_params:
                metrics[f'{key}_rmse'] = MetricsCalculator.rmse(
                    pred_params[key], true_params[key]
                )
                metrics[f'{key}_mae'] = MetricsCalculator.mae(
                    pred_params[key], true_params[key]
                )

        return metrics


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict
):
    """Create learning rate scheduler"""
    scheduler_config = config['training']['lr_scheduler']
    scheduler_type = scheduler_config['type']

    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config['T_max'],
            eta_min=scheduler_config.get('eta_min', 0)
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 50),
            gamma=scheduler_config.get('gamma', 0.5)
        )
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 10)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Test utilities
    config = load_config("configs/ieee33_config.yaml")
    print(f"Loaded config for {config['system']['name']}")

    # Test metrics
    pred = torch.randn(10, 33)
    target = pred + torch.randn(10, 33) * 0.1

    print(f"RMSE: {MetricsCalculator.rmse(pred, target):.6f}")
    print(f"MAE: {MetricsCalculator.mae(pred, target):.6f}")
    print(f"MAPE: {MetricsCalculator.mape(pred, target):.2f}%")
