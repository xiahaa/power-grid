"""
Data Generation Module for Power Grid State & Parameter Estimation

Generates realistic power grid data using Pandapower with:
- Dynamic load/PV profiles
- Parameter drift (line aging, temperature effects)
- Sparse PMU measurements
- Topology changes

Author: Your Name
Date: 2026-01-18
"""

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class PowerGridDataGenerator:
    """Generate synthetic power grid data with realistic dynamics"""

    def __init__(
        self,
        system_name: str = "ieee33",
        num_scenarios: int = 1000,
        time_steps: int = 288,
        pmu_coverage: float = 0.3,
        noise_std: float = 0.02,
        parameter_drift_enabled: bool = True,
        pv_penetration: float = 0.4,
        seed: int = 42
    ):
        """
        Args:
            system_name: "ieee33" or "ieee118"
            num_scenarios: Number of independent scenarios
            time_steps: Number of time steps per scenario
            pmu_coverage: Fraction of buses with PMUs
            noise_std: Measurement noise standard deviation
            parameter_drift_enabled: Enable line parameter drift
            pv_penetration: Fraction of buses with PV
            seed: Random seed
        """
        self.system_name = system_name
        self.num_scenarios = num_scenarios
        self.time_steps = time_steps
        self.pmu_coverage = pmu_coverage
        self.noise_std = noise_std
        self.parameter_drift_enabled = parameter_drift_enabled
        self.pv_penetration = pv_penetration
        self.seed = seed

        np.random.seed(seed)

        # Load network
        self.net = self._load_network()
        self.num_buses = len(self.net.bus)
        self.num_lines = len(self.net.line)

        # Store original parameters
        self.original_r = self.net.line['r_ohm_per_km'].values.copy()
        self.original_x = self.net.line['x_ohm_per_km'].values.copy()

        # Setup PMU locations
        self.pmu_buses = self._select_pmu_locations()

        print(f"Initialized {system_name} system:")
        print(f"  Buses: {self.num_buses}, Lines: {self.num_lines}")
        print(f"  PMU coverage: {len(self.pmu_buses)}/{self.num_buses} buses")

    def _load_network(self) -> pp.pandapowerNet:
        """Load standard IEEE test system"""
        if self.system_name == "ieee33":
            # IEEE 33-bus radial distribution system
            net = pn.case33bw()
        elif self.system_name == "ieee118":
            # IEEE 118-bus transmission system
            net = pn.case118()
        else:
            raise ValueError(f"Unknown system: {self.system_name}")
        return net

    def _select_pmu_locations(self) -> List[int]:
        """Select PMU locations for maximum observability"""
        num_pmus = int(self.num_buses * self.pmu_coverage)

        # Strategy: prioritize buses with high degree (many connections)
        bus_degrees = np.zeros(self.num_buses)
        for _, line in self.net.line.iterrows():
            bus_degrees[int(line['from_bus'])] += 1
            bus_degrees[int(line['to_bus'])] += 1

        # Always include slack bus
        pmu_buses = [self.net.ext_grid.at[0, 'bus']]

        # Select remaining based on degree
        remaining = num_pmus - 1
        candidates = np.argsort(bus_degrees)[::-1]
        for bus in candidates:
            if bus not in pmu_buses:
                pmu_buses.append(int(bus))
                if len(pmu_buses) >= num_pmus:
                    break

        return sorted(pmu_buses)

    def _generate_load_profile(self, base_load: float, hour: float) -> float:
        """Generate realistic load profile (residential/commercial)"""
        # Daily pattern with morning/evening peaks
        daily_pattern = (
            0.6 +
            0.3 * np.sin(2 * np.pi * (hour - 6) / 24) +  # Sinusoidal
            0.1 * np.sin(4 * np.pi * (hour - 9) / 24)    # Harmonics
        )
        # Add randomness
        noise = np.random.normal(0, 0.05)
        return base_load * np.clip(daily_pattern + noise, 0.3, 1.2)

    def _generate_pv_profile(self, base_pv: float, hour: float) -> float:
        """Generate PV generation profile (solar irradiance)"""
        # Solar generation: peak at noon, zero at night
        if 6 <= hour <= 18:
            solar_pattern = np.sin(np.pi * (hour - 6) / 12) ** 2
            # Cloud variability
            cloud_factor = np.random.uniform(0.7, 1.0)
            return base_pv * solar_pattern * cloud_factor
        else:
            return 0.0

    def _apply_parameter_drift(
        self,
        scenario_idx: int,
        drift_lines: List[int] = None,
        drift_range: Tuple[float, float] = (0.05, 0.15)
    ):
        """Apply parameter drift to simulate line aging/temperature effects"""
        if not self.parameter_drift_enabled:
            return

        if drift_lines is None:
            # Randomly select 5-10% of lines
            num_drift_lines = max(1, int(0.1 * self.num_lines))
            drift_lines = np.random.choice(
                self.num_lines,
                size=num_drift_lines,
                replace=False
            )

        for line_idx in drift_lines:
            # Resistance increases with aging/temperature (5-15%)
            drift_factor = np.random.uniform(*drift_range)
            # Gradual drift over scenarios
            progress = scenario_idx / self.num_scenarios
            current_drift = 1.0 + drift_factor * progress

            self.net.line.at[line_idx, 'r_ohm_per_km'] = (
                self.original_r[line_idx] * current_drift
            )
            # Reactance changes less (1-3%)
            self.net.line.at[line_idx, 'x_ohm_per_km'] = (
                self.original_x[line_idx] * (1.0 + 0.02 * progress)
            )

    def _add_pv_systems(self):
        """Add PV systems to random buses"""
        num_pv_buses = int(self.num_buses * self.pv_penetration)
        # Avoid slack bus
        available_buses = [b for b in range(self.num_buses) if b != self.net.ext_grid.at[0, 'bus']]
        pv_buses = np.random.choice(available_buses, size=num_pv_buses, replace=False)

        self.pv_buses = []
        for bus in pv_buses:
            # PV capacity: 50-200% of load
            bus_load = self.net.load[self.net.load.bus == bus]['p_mw'].sum()
            if bus_load > 0:
                pv_capacity = bus_load * np.random.uniform(0.5, 2.0)
                pp.create_sgen(
                    self.net,
                    bus=int(bus),
                    p_mw=0,  # Will be updated dynamically
                    q_mvar=0,
                    name=f"PV_{bus}"
                )
                self.pv_buses.append(int(bus))

    def generate_scenario(
        self,
        scenario_idx: int
    ) -> Dict[str, np.ndarray]:
        """Generate one complete scenario with time series data"""

        # Reset network
        self.net = self._load_network()
        self._add_pv_systems()
        self._apply_parameter_drift(scenario_idx)

        # Storage for time series
        measurements = {
            'v_mag': np.zeros((self.time_steps, self.num_buses)),
            'v_ang': np.zeros((self.time_steps, self.num_buses)),
            'p_bus': np.zeros((self.time_steps, self.num_buses)),
            'q_bus': np.zeros((self.time_steps, self.num_buses)),
            'p_line': np.zeros((self.time_steps, self.num_lines)),
            'q_line': np.zeros((self.time_steps, self.num_lines)),
        }

        true_states = {
            'v_mag': np.zeros((self.time_steps, self.num_buses)),
            'v_ang': np.zeros((self.time_steps, self.num_buses)),
        }

        parameters = {
            'r_line': self.net.line['r_ohm_per_km'].values.copy(),
            'x_line': self.net.line['x_ohm_per_km'].values.copy(),
        }

        # Time series simulation
        for t in range(self.time_steps):
            hour = (t * 5 / 60) % 24  # 5-minute intervals

            # Update loads
            for idx, load in self.net.load.iterrows():
                base_p = self.net.load.at[idx, 'p_mw']
                base_q = self.net.load.at[idx, 'q_mvar']
                self.net.load.at[idx, 'p_mw'] = self._generate_load_profile(base_p, hour)
                self.net.load.at[idx, 'q_mvar'] = self._generate_load_profile(base_q, hour)

            # Update PV generation
            for idx, sgen in self.net.sgen.iterrows():
                bus = self.net.sgen.at[idx, 'bus']
                bus_load = self.net.load[self.net.load.bus == bus]['p_mw'].sum()
                base_pv = bus_load * np.random.uniform(0.5, 2.0)
                self.net.sgen.at[idx, 'p_mw'] = self._generate_pv_profile(base_pv, hour)

            # Run power flow
            try:
                pp.runpp(self.net, algorithm='nr', max_iteration=50)

                # Store true states
                true_states['v_mag'][t] = self.net.res_bus['vm_pu'].values
                true_states['v_ang'][t] = np.deg2rad(self.net.res_bus['va_degree'].values)

                # Store measurements (with noise for non-PMU buses)
                measurements['v_mag'][t] = true_states['v_mag'][t].copy()
                measurements['v_ang'][t] = true_states['v_ang'][t].copy()
                measurements['p_bus'][t] = self.net.res_bus['p_mw'].values
                measurements['q_bus'][t] = self.net.res_bus['q_mvar'].values
                measurements['p_line'][t] = self.net.res_line['p_from_mw'].values
                measurements['q_line'][t] = self.net.res_line['q_from_mvar'].values

            except Exception as e:
                print(f"Power flow failed at t={t}: {e}")
                # Use previous values
                if t > 0:
                    true_states['v_mag'][t] = true_states['v_mag'][t-1]
                    true_states['v_ang'][t] = true_states['v_ang'][t-1]
                    measurements['v_mag'][t] = measurements['v_mag'][t-1]
                    measurements['v_ang'][t] = measurements['v_ang'][t-1]

        # Add measurement noise
        measurements = self._add_measurement_noise(measurements)

        # Create sparse observation mask (only PMU buses fully observed)
        obs_mask = self._create_observation_mask()

        return {
            'measurements': measurements,
            'true_states': true_states,
            'parameters': parameters,
            'obs_mask': obs_mask,
            'pmu_buses': self.pmu_buses,
            'topology': self._get_topology()
        }

    def _add_measurement_noise(
        self,
        measurements: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Add Gaussian noise to measurements"""
        noisy = {}
        for key, val in measurements.items():
            noise = np.random.normal(0, self.noise_std, val.shape)
            noisy[key] = val + noise * np.abs(val)
        return noisy

    def _create_observation_mask(self) -> np.ndarray:
        """Create observation mask (1 = observed, 0 = missing)"""
        mask = np.zeros((self.time_steps, self.num_buses), dtype=bool)
        # PMU buses always observed
        mask[:, self.pmu_buses] = True
        # Random additional observations (e.g., SCADA)
        for t in range(self.time_steps):
            num_extra = np.random.randint(0, int(0.1 * self.num_buses))
            extra_buses = np.random.choice(
                [b for b in range(self.num_buses) if b not in self.pmu_buses],
                size=num_extra,
                replace=False
            )
            mask[t, extra_buses] = True
        return mask

    def _get_topology(self) -> Dict[str, np.ndarray]:
        """Extract network topology as adjacency information"""
        edge_index = []
        edge_attr = []

        for _, line in self.net.line.iterrows():
            from_bus = int(line['from_bus'])
            to_bus = int(line['to_bus'])
            # Undirected graph
            edge_index.append([from_bus, to_bus])
            edge_index.append([to_bus, from_bus])

            # Edge attributes: impedance
            r = line['r_ohm_per_km']
            x = line['x_ohm_per_km']
            edge_attr.extend([[r, x], [r, x]])

        return {
            'edge_index': np.array(edge_index, dtype=np.int64).T,
            'edge_attr': np.array(edge_attr, dtype=np.float32),
            'num_nodes': self.num_buses
        }

    def generate_dataset(
        self,
        save_path: Optional[str] = None
    ) -> List[Dict]:
        """Generate complete dataset with all scenarios"""
        print(f"\nGenerating {self.num_scenarios} scenarios...")

        dataset = []
        for scenario_idx in tqdm(range(self.num_scenarios)):
            scenario_data = self.generate_scenario(scenario_idx)
            dataset.append(scenario_data)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'wb') as f:
                pickle.dump(dataset, f)

            print(f"\nDataset saved to: {save_path}")
            self._print_statistics(dataset)

        return dataset

    def _print_statistics(self, dataset: List[Dict]):
        """Print dataset statistics"""
        all_vmag = np.concatenate([d['true_states']['v_mag'] for d in dataset])
        all_params_r = np.array([d['parameters']['r_line'] for d in dataset])

        print("\nDataset Statistics:")
        print(f"  Voltage magnitude: {all_vmag.mean():.4f} ± {all_vmag.std():.4f} p.u.")
        print(f"  Voltage range: [{all_vmag.min():.4f}, {all_vmag.max():.4f}] p.u.")
        print(f"  Line resistance: {all_params_r.mean():.4f} ± {all_params_r.std():.4f} Ω/km")
        if self.parameter_drift_enabled:
            r_drift = (all_params_r[-1].mean() / all_params_r[0].mean() - 1) * 100
            print(f"  Parameter drift: +{r_drift:.2f}% over scenarios")


if __name__ == "__main__":
    # Example usage
    generator = PowerGridDataGenerator(
        system_name="ieee33",
        num_scenarios=100,
        time_steps=288,
        pmu_coverage=0.3,
        parameter_drift_enabled=True
    )

    dataset = generator.generate_dataset(
        save_path="data/raw/ieee33_dataset.pkl"
    )
