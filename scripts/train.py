"""
Main training script for Physics-Informed Graph Mamba

Usage:
    python scripts/train.py --config configs/ieee33_config.yaml

Author: Your Name
Date: 2026-01-18
"""

import sys
import os
# Add src directory to path (insert at beginning to avoid conflicts)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
from tqdm import tqdm
import time

# SwanLab integration
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("Warning: SwanLab not installed. Install with: pip install swanlab")

from data.dataloader import get_dataloader
from models.graph_mamba import GraphMamba
from physics.constraints import PhysicsInformedLayer, PhysicsInformedGraphMamba
from train.loss import JointEstimationLoss
from utils.utils import (
    load_config, save_checkpoint, load_checkpoint,
    EarlyStopping, MetricsCalculator, get_lr_scheduler,
    AverageMeter, count_parameters, set_seed
)


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int
) -> dict:
    """Train for one epoch"""
    model.train()

    losses = {
        'total': AverageMeter(),
        'state': AverageMeter(),
        'parameter': AverageMeter(),
        'physics': AverageMeter()
    }

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        measurements = {k: v.to(device) for k, v in batch['measurements'].items()}
        obs_mask = batch['obs_mask'].to(device)
        true_states = {k: v.to(device) for k, v in batch['true_states'].items()}
        true_params = {k: v.to(device) for k, v in batch['parameters'].items()}
        edge_index = batch['topology']['edge_index'].to(device)
        edge_attr = batch['topology']['edge_attr'].to(device)

        # Forward pass
        pred_states, pred_params, physics_loss = model(
            measurements, edge_index, edge_attr, obs_mask
        )

        # Compute loss
        loss, loss_dict = criterion(
            pred_states, true_states,
            pred_params, true_params,
            physics_loss,
            obs_mask
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update meters
        batch_size = measurements['v_mag'].shape[0]
        losses['total'].update(loss_dict['total'], batch_size)
        losses['state'].update(loss_dict['state'], batch_size)
        losses['parameter'].update(loss_dict['parameter'], batch_size)
        losses['physics'].update(loss_dict['physics'], batch_size)

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['total'].avg:.4f}",
            'state': f"{losses['state'].avg:.4f}",
            'param': f"{losses['parameter'].avg:.4f}"
        })

    return {k: v.avg for k, v in losses.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str
) -> dict:
    """Validate model"""
    model.eval()

    losses = {
        'total': AverageMeter(),
        'state': AverageMeter(),
        'parameter': AverageMeter(),
        'physics': AverageMeter()
    }

    all_pred_states = []
    all_true_states = []
    all_pred_params = []
    all_true_params = []

    for batch in tqdm(dataloader, desc="Validating"):
        # Move to device
        measurements = {k: v.to(device) for k, v in batch['measurements'].items()}
        obs_mask = batch['obs_mask'].to(device)
        true_states = {k: v.to(device) for k, v in batch['true_states'].items()}
        true_params = {k: v.to(device) for k, v in batch['parameters'].items()}
        edge_index = batch['topology']['edge_index'].to(device)
        edge_attr = batch['topology']['edge_attr'].to(device)

        # Forward pass
        pred_states, pred_params, physics_loss = model(
            measurements, edge_index, edge_attr, obs_mask
        )

        # Compute loss
        loss, loss_dict = criterion(
            pred_states, true_states,
            pred_params, true_params,
            physics_loss,
            obs_mask
        )

        # Update meters
        batch_size = measurements['v_mag'].shape[0]
        losses['total'].update(loss_dict['total'], batch_size)
        losses['state'].update(loss_dict['state'], batch_size)
        losses['parameter'].update(loss_dict['parameter'], batch_size)
        losses['physics'].update(loss_dict['physics'], batch_size)

        # Store for metrics
        all_pred_states.append(pred_states)
        all_true_states.append(true_states)
        all_pred_params.append(pred_params)
        all_true_params.append(true_params)

    # Compute metrics
    pred_states_all = {
        k: torch.cat([d[k] for d in all_pred_states])
        for k in all_pred_states[0].keys()
    }
    true_states_all = {
        k: torch.cat([d[k] for d in all_true_states])
        for k in all_true_states[0].keys()
    }
    pred_params_all = {
        k: torch.cat([d[k] for d in all_pred_params])
        for k in all_pred_params[0].keys()
    }
    true_params_all = {
        k: torch.cat([d[k] for d in all_true_params])
        for k in all_true_params[0].keys()
    }

    metrics = MetricsCalculator.compute_all_metrics(
        pred_states_all, true_states_all,
        pred_params_all, true_params_all
    )

    results = {**{k: v.avg for k, v in losses.items()}, **metrics}
    return results


def main(args):
    # Load configuration
    config = load_config(args.config)
    print(f"\n{'='*60}")
    print(f"Training {config['system']['name']} system")
    print(f"{'='*60}\n")

    # Set seed
    set_seed(42)

    # Device
    device = config['hardware']['device'] if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create save directory
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize loggers
    writer = None
    swanlab_run = None

    # TensorBoard
    if config['logging']['use_tensorboard']:
        log_dir = Path(config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print("✓ TensorBoard logging enabled")

    # SwanLab
    if config['logging'].get('use_swanlab', False):
        if SWANLAB_AVAILABLE:
            swanlab_run = swanlab.init(
                project=config['logging'].get('swanlab_project', 'power-grid-estimation'),
                experiment_name=config['logging'].get('swanlab_experiment', config['system']['name']),
                config={
                    'system': config['system']['name'],
                    'num_buses': config['system']['num_buses'],
                    'pmu_coverage': config['system']['pmu_coverage'],
                    'batch_size': config['training']['batch_size'],
                    'learning_rate': config['training']['learning_rate'],
                    'num_epochs': config['training']['num_epochs'],
                    'spatial_encoder': config['model']['spatial_encoder'],
                    'temporal_encoder': config['model']['temporal_encoder'],
                    'physics_enabled': config['physics']['enabled'],
                    'constraint_type': config['physics']['constraint_type'],
                },
                description=f"Physics-Informed Graph Mamba for {config['system']['name']} power grid estimation"
            )
            print("✓ SwanLab logging enabled")
            print(f"  Project: {config['logging'].get('swanlab_project', 'power-grid-estimation')}")
            print(f"  Experiment: {config['logging'].get('swanlab_experiment', config['system']['name'])}")
        else:
            print("⚠ SwanLab requested but not installed. Skipping.")
    else:
        print("✓ SwanLab logging disabled")

    # Data loaders
    print("\nLoading data...")
    data_path = f"data/raw/{config['system']['name']}_dataset.pkl"

    train_loader = get_dataloader(
        data_path=data_path,
        batch_size=config['training']['batch_size'],
        split='train',
        num_workers=config['hardware']['num_workers'],
        sequence_length=10
    )

    val_loader = get_dataloader(
        data_path=data_path,
        batch_size=config['training']['batch_size'],
        split='val',
        num_workers=config['hardware']['num_workers'],
        sequence_length=10
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Model
    print("\nBuilding model...")
    graph_mamba = GraphMamba(
        num_nodes=config['system']['num_buses'],
        num_edges=len(train_loader.dataset.data[0]['topology']['edge_index'][0]),
        input_dim=3,
        spatial_config=config['model']['spatial_encoder'],
        temporal_config=config['model']['temporal_encoder'],
        state_head_config=config['model']['state_head'],
        parameter_head_config=config['model']['parameter_head']
    )

    physics_layer = PhysicsInformedLayer(
        constraint_type=config['physics']['constraint_type'],
        projection_method=config['physics']['projection_method'],
        max_iterations=config['physics']['max_iterations'],
        tolerance=config['physics']['tolerance'],
        voltage_limits=config['physics']['power_flow']['voltage_limits'],
        power_balance_weight=config['physics']['power_flow']['power_balance_weight']
    )

    model = PhysicsInformedGraphMamba(graph_mamba, physics_layer).to(device)

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # Loss and optimizer
    criterion = JointEstimationLoss(
        state_weight=config['loss']['state_weight'],
        parameter_weight=config['loss']['parameter_weight'],
        physics_weight=config['loss']['physics_weight'],
        temporal_smoothness=config['loss']['temporal_smoothness']
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = get_lr_scheduler(optimizer, config)

    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping']['patience'],
        min_delta=config['training']['early_stopping']['min_delta']
    )

    # Training loop
    print("\nStarting training...\n")
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(1, config['training']['num_epochs'] + 1):
        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_results = validate(model, val_loader, criterion, device)

        # Scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_results['total'])
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Logging
        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}:")
        print(f"  Train Loss: {train_losses['total']:.6f}")
        print(f"  Val Loss: {val_results['total']:.6f}")
        print(f"  Val V_mag RMSE: {val_results.get('v_mag_rmse', 0):.6f}")
        print(f"  Val R_line MAE: {val_results.get('r_line_mae', 0):.6f}")
        print(f"  LR: {current_lr:.2e}")

        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/train', train_losses['total'], epoch)
            writer.add_scalar('Loss/val', val_results['total'], epoch)
            writer.add_scalar('Loss/train_state', train_losses['state'], epoch)
            writer.add_scalar('Loss/train_parameter', train_losses['parameter'], epoch)
            writer.add_scalar('Loss/train_physics', train_losses['physics'], epoch)
            writer.add_scalar('LR', current_lr, epoch)
            for key, val in val_results.items():
                if 'rmse' in key or 'mae' in key or 'mape' in key:
                    writer.add_scalar(f'Metrics/{key}', val, epoch)

        # SwanLab logging
        if swanlab_run:
            swanlab.log({
                # Losses
                'train/loss_total': train_losses['total'],
                'train/loss_state': train_losses['state'],
                'train/loss_parameter': train_losses['parameter'],
                'train/loss_physics': train_losses['physics'],
                'val/loss_total': val_results['total'],
                'val/loss_state': val_results.get('state', 0),
                'val/loss_parameter': val_results.get('parameter', 0),
                'val/loss_physics': val_results.get('physics', 0),

                # State estimation metrics
                'val/v_mag_rmse': val_results.get('v_mag_rmse', 0),
                'val/v_mag_mae': val_results.get('v_mag_mae', 0),
                'val/v_mag_mape': val_results.get('v_mag_mape', 0),
                'val/v_ang_rmse': val_results.get('v_ang_rmse', 0),
                'val/v_ang_mae': val_results.get('v_ang_mae', 0),

                # Parameter estimation metrics
                'val/r_line_rmse': val_results.get('r_line_rmse', 0),
                'val/r_line_mae': val_results.get('r_line_mae', 0),
                'val/x_line_rmse': val_results.get('x_line_rmse', 0),
                'val/x_line_mae': val_results.get('x_line_mae', 0),

                # Training dynamics
                'train/learning_rate': current_lr,
                'train/epoch': epoch,
            }, step=epoch)

        # Save best model
        if val_results['total'] < best_val_loss:
            best_val_loss = val_results['total']
            save_checkpoint(
                model, optimizer, epoch, best_val_loss,
                save_dir / 'best_model.pt'
            )

        # Periodic save
        if epoch % config['training']['save_freq'] == 0:
            save_checkpoint(
                model, optimizer, epoch, val_results['total'],
                save_dir / f'checkpoint_epoch_{epoch}.pt'
            )

        # Early stopping
        if early_stopping(val_results['total']):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    # Training complete
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed in {elapsed/3600:.2f} hours")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"{'='*60}\n")

    # Close loggers
    if writer:
        writer.close()

    if swanlab_run:
        # Log final summary
        swanlab.log({
            'summary/best_val_loss': best_val_loss,
            'summary/total_epochs': epoch,
            'summary/training_time_hours': elapsed / 3600,
            'summary/model_parameters': n_params,
        })
        swanlab.finish()
        print("✓ SwanLab experiment finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    args = parser.parse_args()

    main(args)
