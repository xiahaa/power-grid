"""
PyTorch Dataset and DataLoader for Power Grid Data

Author: Your Name
Date: 2026-01-18
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from torch_geometric.data import Data


class PowerGridDataset(Dataset):
    """PyTorch Dataset for power grid state estimation"""

    def __init__(
        self,
        data_path: str,
        sequence_length: int = 10,
        prediction_horizon: int = 1,
        split: str = 'train',
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    ):
        """
        Args:
            data_path: Path to pickled dataset
            sequence_length: Number of historical time steps
            prediction_horizon: Number of future steps to predict
            split: 'train', 'val', or 'test'
            split_ratios: Train/val/test split ratios
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.split = split

        # Load data
        with open(data_path, 'rb') as f:
            self.raw_data = pickle.load(f)

        # Split dataset
        self.data = self._split_data(split_ratios)

        # Build sequence indices
        self.sequences = self._build_sequences()

        print(f"{split.upper()} set: {len(self.sequences)} sequences")

    def _split_data(self, split_ratios: Tuple[float, float, float]) -> List[Dict]:
        """Split data into train/val/test"""
        n_total = len(self.raw_data)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])

        if self.split == 'train':
            return self.raw_data[:n_train]
        elif self.split == 'val':
            return self.raw_data[n_train:n_train+n_val]
        else:  # test
            return self.raw_data[n_train+n_val:]

    def _build_sequences(self) -> List[Tuple[int, int]]:
        """Build list of (scenario_idx, start_time) tuples"""
        sequences = []
        for scenario_idx, scenario in enumerate(self.data):
            n_times = scenario['measurements']['v_mag'].shape[0]
            max_start = n_times - self.sequence_length - self.prediction_horizon
            for start_t in range(max_start):
                sequences.append((scenario_idx, start_t))
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get one sequence sample"""
        scenario_idx, start_t = self.sequences[idx]
        scenario = self.data[scenario_idx]

        # Time indices
        hist_slice = slice(start_t, start_t + self.sequence_length)
        pred_slice = slice(
            start_t + self.sequence_length,
            start_t + self.sequence_length + self.prediction_horizon
        )

        # Extract measurements (inputs)
        measurements = {
            'v_mag': torch.tensor(
                scenario['measurements']['v_mag'][hist_slice],
                dtype=torch.float32
            ),
            'p_bus': torch.tensor(
                scenario['measurements']['p_bus'][hist_slice],
                dtype=torch.float32
            ),
            'q_bus': torch.tensor(
                scenario['measurements']['q_bus'][hist_slice],
                dtype=torch.float32
            ),
        }

        # Observation mask
        obs_mask = torch.tensor(
            scenario['obs_mask'][hist_slice],
            dtype=torch.bool
        )

        # True states (targets)
        true_states = {
            'v_mag': torch.tensor(
                scenario['true_states']['v_mag'][pred_slice],
                dtype=torch.float32
            ),
            'v_ang': torch.tensor(
                scenario['true_states']['v_ang'][pred_slice],
                dtype=torch.float32
            ),
        }

        # Parameters (targets for parameter estimation)
        parameters = {
            'r_line': torch.tensor(
                scenario['parameters']['r_line'],
                dtype=torch.float32
            ),
            'x_line': torch.tensor(
                scenario['parameters']['x_line'],
                dtype=torch.float32
            ),
        }

        # Topology (graph structure)
        topology = {
            'edge_index': torch.tensor(
                scenario['topology']['edge_index'],
                dtype=torch.long
            ),
            'edge_attr': torch.tensor(
                scenario['topology']['edge_attr'],
                dtype=torch.float32
            ),
        }

        return {
            'measurements': measurements,
            'obs_mask': obs_mask,
            'true_states': true_states,
            'parameters': parameters,
            'topology': topology,
            'pmu_buses': scenario['pmu_buses'],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching graph data"""
    # Stack measurements and masks
    measurements = {
        key: torch.stack([b['measurements'][key] for b in batch])
        for key in batch[0]['measurements'].keys()
    }

    obs_mask = torch.stack([b['obs_mask'] for b in batch])

    true_states = {
        key: torch.stack([b['true_states'][key] for b in batch])
        for key in batch[0]['true_states'].keys()
    }

    parameters = {
        key: torch.stack([b['parameters'][key] for b in batch])
        for key in batch[0]['parameters'].keys()
    }

    # Graphs (same topology for all samples in practice, but keep flexible)
    topology = {
        'edge_index': batch[0]['topology']['edge_index'],  # Same for all
        'edge_attr': batch[0]['topology']['edge_attr'],
    }

    return {
        'measurements': measurements,
        'obs_mask': obs_mask,
        'true_states': true_states,
        'parameters': parameters,
        'topology': topology,
    }


def get_dataloader(
    data_path: str,
    batch_size: int,
    split: str = 'train',
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """Create DataLoader for power grid data"""
    dataset = PowerGridDataset(data_path=data_path, split=split, **kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


if __name__ == "__main__":
    # Test dataset
    loader = get_dataloader(
        data_path="data/raw/ieee33_dataset.pkl",
        batch_size=4,
        split='train'
    )

    batch = next(iter(loader))
    print("\nBatch shapes:")
    print(f"  V measurements: {batch['measurements']['v_mag'].shape}")
    print(f"  Obs mask: {batch['obs_mask'].shape}")
    print(f"  True V: {batch['true_states']['v_mag'].shape}")
    print(f"  Parameters: {batch['parameters']['r_line'].shape}")
    print(f"  Edge index: {batch['topology']['edge_index'].shape}")
