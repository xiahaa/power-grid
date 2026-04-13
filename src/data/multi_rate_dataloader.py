"""
Multi-Rate DataLoader for DSSE with Heterogeneous Measurements

Handles loading of dual-rate (SCADA + PMU) measurement data with proper
temporal alignment for the MultiRateMambaFusion model.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class MultiRateDataset(Dataset):
    """
    Dataset for multi-rate (SCADA + PMU) power grid data.

    Returns separate SCADA and PMU measurement sequences along with
    their respective observation masks.
    """

    def __init__(
        self,
        data_path: str,
        sequence_length: int = 10,
        prediction_horizon: int = 1,
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        max_sequences: Optional[int] = None,
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.split = split
        self.max_sequences = max_sequences

        with open(data_path, "rb") as f:
            self.raw_data = pickle.load(f)

        self.data = self._split_data(split_ratios)
        self.sequences = self._build_sequences()
        if self.max_sequences is not None:
            self.sequences = self.sequences[: self.max_sequences]

        print(f"MultiRate {split.upper()} set: {len(self.sequences)} sequences")

    def _split_data(self, split_ratios):
        n_total = len(self.raw_data)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])

        if self.split == "train":
            return self.raw_data[:n_train]
        elif self.split == "val":
            return self.raw_data[n_train : n_train + n_val]
        else:
            return self.raw_data[n_train + n_val :]

    def _build_sequences(self):
        sequences = []
        for scenario_idx, scenario in enumerate(self.data):
            n_times = scenario["true_states"]["v_mag"].shape[0]
            max_start = n_times - self.sequence_length - self.prediction_horizon + 1

            if max_start <= 0:
                continue

            for start_t in range(max_start):
                target_t = start_t + self.sequence_length
                v_target = scenario["true_states"]["v_mag"][
                    target_t : target_t + self.prediction_horizon
                ]

                if np.std(v_target) < 1e-8:
                    continue

                if np.isnan(v_target).mean() > 0.1:
                    continue

                has_topo = False
                topo_mask = scenario.get(
                    "topology_change_mask", np.zeros(n_times, dtype=bool)
                )
                if (
                    topo_mask is not None
                    and topo_mask[start_t : start_t + self.sequence_length].any()
                ):
                    has_topo = True

                sequences.append((scenario_idx, start_t, has_topo))

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        scenario_idx, start_t, has_topo = self.sequences[idx]
        scenario = self.data[scenario_idx]

        hist_slice = slice(start_t, start_t + self.sequence_length)
        pred_slice = slice(
            start_t + self.sequence_length,
            start_t + self.sequence_length + self.prediction_horizon,
        )

        mr = scenario["multi_rate_measurements"]

        # SCADA measurements (slow rate)
        scada_meas = {}
        for key in ["v_mag", "p_bus", "q_bus"]:
            if key in mr["scada"]:
                scada_meas[key] = torch.tensor(
                    mr["scada"][key][hist_slice], dtype=torch.float32
                )
            else:
                scada_meas[key] = torch.zeros(
                    self.sequence_length, scenario["num_buses"], dtype=torch.float32
                )

        scada_mask = torch.tensor(mr["scada_obs_mask"][hist_slice], dtype=torch.bool)

        # PMU measurements (fast rate)
        pmu_meas = {}
        for key in ["v_mag", "v_ang"]:
            if key in mr["pmu"]:
                pmu_meas[key] = torch.tensor(
                    mr["pmu"][key][hist_slice], dtype=torch.float32
                )
            else:
                pmu_meas[key] = torch.zeros(
                    self.sequence_length, scenario["num_buses"], dtype=torch.float32
                )

        pmu_mask = torch.tensor(mr["pmu_obs_mask"][hist_slice], dtype=torch.bool)

        # True states (targets)
        true_state_raw = {
            "v_mag": torch.tensor(
                scenario["true_states"]["v_mag"][pred_slice], dtype=torch.float32
            ).squeeze(0),
            "v_ang": torch.tensor(
                scenario["true_states"]["v_ang"][pred_slice], dtype=torch.float32
            ).squeeze(0),
        }
        true_state_masks = {
            key: torch.isfinite(value) for key, value in true_state_raw.items()
        }
        true_states = {
            key: torch.nan_to_num(value, nan=0.0)
            for key, value in true_state_raw.items()
        }

        # Parameters
        num_lines = scenario["num_lines"]
        parameters = {
            "r_line": torch.tensor(
                scenario["parameters"]["r_line"][:num_lines], dtype=torch.float32
            ),
            "x_line": torch.tensor(
                scenario["parameters"]["x_line"][:num_lines], dtype=torch.float32
            ),
        }

        # Topology
        edge_index_np = scenario["topology"]["edge_index"]
        edge_attr_np = scenario["topology"]["edge_attr"]

        # If edge_attr has more rows than 2*num_lines (due to switches), truncate
        num_edges = edge_index_np.shape[1]
        if num_edges > 2 * num_lines:
            edge_index_np = edge_index_np[:, : 2 * num_lines]
            edge_attr_np = edge_attr_np[: 2 * num_lines]
            num_edges = 2 * num_lines

        topology = {
            "edge_index": torch.tensor(edge_index_np, dtype=torch.long),
            "edge_attr": torch.tensor(edge_attr_np, dtype=torch.float32),
        }

        # Expand parameters to match edges
        parameters_expanded = {}
        for key in ["r_line", "x_line"]:
            line_params = parameters[key]
            if num_edges == 2 * len(line_params):
                parameters_expanded[key] = line_params.repeat_interleave(2)
            else:
                parameters_expanded[key] = line_params[:num_edges]

        return {
            "scada_meas": scada_meas,
            "scada_mask": scada_mask,
            "pmu_meas": pmu_meas,
            "pmu_mask": pmu_mask,
            "true_states": true_states,
            "true_state_masks": true_state_masks,
            "parameters": parameters_expanded,
            "topology": topology,
            "has_topology_change": torch.tensor(has_topo, dtype=torch.bool),
        }


def multi_rate_collate_fn(batch):
    """Custom collate for multi-rate data."""
    scada_meas = {
        key: torch.stack([b["scada_meas"][key] for b in batch])
        for key in batch[0]["scada_meas"].keys()
    }
    scada_mask = torch.stack([b["scada_mask"] for b in batch])

    pmu_meas = {
        key: torch.stack([b["pmu_meas"][key] for b in batch])
        for key in batch[0]["pmu_meas"].keys()
    }
    pmu_mask = torch.stack([b["pmu_mask"] for b in batch])

    true_states = {
        key: torch.stack([b["true_states"][key] for b in batch])
        for key in batch[0]["true_states"].keys()
    }
    true_state_masks = {
        key: torch.stack([b["true_state_masks"][key] for b in batch])
        for key in batch[0]["true_state_masks"].keys()
    }

    parameters = {
        key: torch.stack([b["parameters"][key] for b in batch])
        for key in batch[0]["parameters"].keys()
    }

    topology = {
        "edge_index": batch[0]["topology"]["edge_index"],
        "edge_attr": batch[0]["topology"]["edge_attr"],
    }

    has_topo_change = torch.stack([b["has_topology_change"] for b in batch])

    return {
        "scada_meas": scada_meas,
        "scada_mask": scada_mask,
        "pmu_meas": pmu_meas,
        "pmu_mask": pmu_mask,
        "true_states": true_states,
        "true_state_masks": true_state_masks,
        "parameters": parameters,
        "topology": topology,
        "has_topology_change": has_topo_change,
    }


def get_multi_rate_dataloader(
    data_path: str,
    batch_size: int,
    split: str = "train",
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Create DataLoader for multi-rate power grid data."""
    dataset = MultiRateDataset(data_path=data_path, split=split, **kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=multi_rate_collate_fn,
        pin_memory=True,
    )


if __name__ == "__main__":
    print("MultiRate DataLoader module loaded.")
    print("Usage: get_multi_rate_dataloader(data_path, batch_size, split)")
