"""
Extended Data Generator V2 for DSSE Research

Generates realistic power grid data with:
- Multi-rate heterogeneous measurements (SCADA + PMU)
- Dynamic topology changes (switch operations)
- Parameter drift (line aging)
- Multiple DER scenarios (PV, storage)

Supports IEEE 33, 69, 118, and 123-bus systems.
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
import copy

warnings.filterwarnings("ignore")

from .heterogeneous_measurements import HeterogeneousMeasurementSimulator
from .topology_manager import TopologyManager, TopologyEventType


class PowerGridDataGeneratorV2:
    """
    V2 data generator supporting heterogeneous measurements
    and dynamic topology changes.
    """

    SUPPORTED_SYSTEMS = {
        "ieee33": {"loader": "case33bw", "num_buses": 33},
        "ieee69": {"loader": "case69", "num_buses": 69},
        "ieee118": {"loader": "case118", "num_buses": 118},
        "ieee123": {"loader": "custom", "num_buses": 123},
    }

    def __init__(
        self,
        system_name: str = "ieee123",
        num_scenarios: int = 1000,
        time_steps: int = 288,
        pmu_coverage: float = 0.25,
        scada_coverage: float = 0.6,
        noise_std_pmu: float = 0.001,
        noise_std_scada: float = 0.01,
        parameter_drift_enabled: bool = True,
        pv_penetration: float = 0.35,
        topology_change_enabled: bool = False,
        topology_scenario_type: str = "single_switch",
        seed: int = 42,
    ):
        self.system_name = system_name
        self.num_scenarios = num_scenarios
        self.time_steps = time_steps
        self.pmu_coverage = pmu_coverage
        self.scada_coverage = scada_coverage
        self.noise_std_pmu = noise_std_pmu
        self.noise_std_scada = noise_std_scada
        self.parameter_drift_enabled = parameter_drift_enabled
        self.pv_penetration = pv_penetration
        self.topology_change_enabled = topology_change_enabled
        self.topology_scenario_type = topology_scenario_type
        self.seed = seed

        np.random.seed(seed)

        self.net = self._load_network()
        self.num_buses = len(self.net.bus)
        self.num_lines = len(self.net.line)

        self.original_r = self.net.line["r_ohm_per_km"].values.copy()
        self.original_x = self.net.line["x_ohm_per_km"].values.copy()

        self.measurement_sim = HeterogeneousMeasurementSimulator(
            num_buses=self.num_buses,
            scada_coverage=scada_coverage,
            pmu_coverage=pmu_coverage,
            seed=seed,
        )

        self._identify_switch_lines()

        meas_summary = self.measurement_sim.get_measurement_summary()
        print("Initialized %s V2 generator:" % system_name)
        print("  Buses: %d, Lines: %d" % (self.num_buses, self.num_lines))
        print("  PMU buses: %d" % meas_summary["num_pmu_buses"])
        print("  SCADA buses: %d" % meas_summary["num_scada_buses"])
        print(
            "  Total observability: %.1f%%"
            % (meas_summary["total_observability"] * 100)
        )
        print("  Switch lines: %d" % len(self.switch_line_indices))

    def _load_network(self) -> pp.pandapowerNet:
        """Load IEEE test system."""
        if self.system_name == "ieee33":
            return pn.case33bw()
        elif self.system_name == "ieee69":
            return pn.case69()
        elif self.system_name == "ieee118":
            return pn.case118()
        elif self.system_name == "ieee123":
            from .ieee123_network import build_ieee123_network

            return build_ieee123_network()
        else:
            raise ValueError("Unknown system: %s" % self.system_name)

    def _identify_switch_lines(self):
        """Identify switchable tie lines in the network."""
        if self.system_name == "ieee123":
            from .ieee123_network import get_switch_line_indices

            self.switch_line_indices = get_switch_line_indices(self.net)
        else:
            num_switches = max(3, len(self.net.line) // 10)
            self.switch_line_indices = list(
                range(len(self.net.line) - num_switches, len(self.net.line))
            )

    def _generate_load_profile(self, base_load: float, hour: float) -> float:
        """Generate realistic load profile (residential/commercial)."""
        daily_pattern = (
            0.6
            + 0.3 * np.sin(2 * np.pi * (hour - 6) / 24)
            + 0.1 * np.sin(4 * np.pi * (hour - 9) / 24)
        )
        noise = np.random.normal(0, 0.05)
        return base_load * np.clip(daily_pattern + noise, 0.3, 1.2)

    def _generate_pv_profile(self, base_pv: float, hour: float) -> float:
        """Generate PV generation profile with cloud variability."""
        if 6 <= hour <= 18:
            solar_pattern = np.sin(np.pi * (hour - 6) / 12) ** 2
            cloud_factor = np.random.uniform(0.7, 1.0)
            return base_pv * solar_pattern * cloud_factor
        else:
            return 0.0

    def _add_pv_systems(self):
        """Add PV systems to random buses."""
        num_pv_buses = int(self.num_buses * self.pv_penetration)
        slack_bus = self.net.ext_grid.at[0, "bus"]
        available = [b for b in range(self.num_buses) if b != slack_bus]
        pv_buses = np.random.choice(available, size=num_pv_buses, replace=False)

        self.pv_buses = []
        for bus in pv_buses:
            bus_load = self.net.load[self.net.load.bus == bus]["p_mw"].sum()
            if bus_load > 0:
                pv_capacity = bus_load * np.random.uniform(0.5, 2.0)
                pp.create_sgen(
                    self.net,
                    bus=int(bus),
                    p_mw=pv_capacity,
                    q_mvar=0,
                    name="PV_%d" % bus,
                )
                self.pv_buses.append(int(bus))

    def _apply_parameter_drift(self, scenario_idx: int):
        """Apply parameter drift to simulate line aging."""
        if not self.parameter_drift_enabled:
            return
        num_drift = max(1, int(0.1 * self.num_lines))
        drift_lines = np.random.choice(self.num_lines, size=num_drift, replace=False)
        progress = scenario_idx / max(1, self.num_scenarios)
        for line_idx in drift_lines:
            drift_factor = np.random.uniform(0.05, 0.15)
            current_drift = 1.0 + drift_factor * progress
            self.net.line.at[line_idx, "r_ohm_per_km"] = (
                self.original_r[line_idx] * current_drift
            )
            self.net.line.at[line_idx, "x_ohm_per_km"] = self.original_x[line_idx] * (
                1.0 + 0.02 * progress
            )

    def _get_topology(self, topo_manager=None) -> Dict[str, np.ndarray]:
        """Extract network topology as adjacency information."""
        if topo_manager is not None:
            edge_index = topo_manager.get_edge_index()
            edge_attr = topo_manager.get_edge_attr()
            if edge_index.shape[1] > 0:
                return {
                    "edge_index": edge_index,
                    "edge_attr": edge_attr,
                    "num_nodes": self.num_buses,
                }

        edge_list = []
        attr_list = []
        for idx in self.net.line.index:
            fb = int(self.net.line.at[idx, "from_bus"])
            tb = int(self.net.line.at[idx, "to_bus"])
            r = self.net.line.at[idx, "r_ohm_per_km"]
            x = self.net.line.at[idx, "x_ohm_per_km"]
            edge_list.append([fb, tb])
            edge_list.append([tb, fb])
            attr_list.append([r, x])
            attr_list.append([r, x])

        return {
            "edge_index": np.array(edge_list, dtype=np.int64).T,
            "edge_attr": np.array(attr_list, dtype=np.float32),
            "num_nodes": self.num_buses,
        }

    def generate_scenario(self, scenario_idx: int) -> Dict:
        """
        Generate one complete scenario with time series data,
        heterogeneous measurements, and optional topology changes.
        """
        if not hasattr(self, "_base_net"):
            self._base_net = copy.deepcopy(self.net)
        self.net = copy.deepcopy(self._base_net)
        base_loads_p = self.net.load["p_mw"].values.copy()
        base_loads_q = self.net.load["q_mvar"].values.copy()
        self._add_pv_systems()
        pv_capacities = (
            self.net.sgen["p_mw"].values.copy()
            if len(self.net.sgen) > 0
            else np.array([])
        )
        self._apply_parameter_drift(scenario_idx)

        topo_manager = TopologyManager(
            self.net,
            switch_line_indices=self.switch_line_indices,
            seed=self.seed + scenario_idx,
        )

        topology_events = []
        topo_change_mask = np.zeros(self.time_steps, dtype=bool)

        if self.topology_change_enabled:
            topology_events = topo_manager.generate_topology_change_scenario(
                time_steps=self.time_steps,
                scenario_type=self.topology_scenario_type,
                num_events=min(3, len(self.switch_line_indices)),
            )
            for ev in topology_events:
                if 0 <= ev.time_step < self.time_steps:
                    topo_change_mask[ev.time_step] = True

        true_states = {
            "v_mag": np.zeros((self.time_steps, self.num_buses)),
            "v_ang": np.zeros((self.time_steps, self.num_buses)),
            "p_bus": np.zeros((self.time_steps, self.num_buses)),
            "q_bus": np.zeros((self.time_steps, self.num_buses)),
        }

        parameters = {
            "r_line": self.net.line["r_ohm_per_km"].values.copy(),
            "x_line": self.net.line["x_ohm_per_km"].values.copy(),
        }

        for t in range(self.time_steps):
            hour = (t * 5 / 60) % 24

            if self.topology_change_enabled:
                for ev in topology_events:
                    if ev.time_step == t:
                        if ev.event_type == TopologyEventType.SWITCH_OPEN:
                            topo_manager.open_line(ev.line_idx, t)
                        elif ev.event_type == TopologyEventType.SWITCH_CLOSE:
                            topo_manager.close_line(ev.line_idx, t)

            for idx in range(len(self.net.load)):
                self.net.load.at[idx, "p_mw"] = self._generate_load_profile(
                    base_loads_p[idx], hour
                )
                self.net.load.at[idx, "q_mvar"] = self._generate_load_profile(
                    base_loads_q[idx], hour
                )

            for idx in range(len(self.net.sgen)):
                self.net.sgen.at[idx, "p_mw"] = self._generate_pv_profile(
                    pv_capacities[idx], hour
                )

            try:
                pp.runpp(self.net, algorithm="nr", max_iteration=50)
                if self.net.converged:
                    true_states["v_mag"][t] = self.net.res_bus["vm_pu"].values
                    true_states["v_ang"][t] = np.deg2rad(
                        self.net.res_bus["va_degree"].values
                    )
                    true_states["p_bus"][t] = self.net.res_bus["p_mw"].values
                    true_states["q_bus"][t] = self.net.res_bus["q_mvar"].values
                elif t > 0:
                    for k in true_states:
                        true_states[k][t] = true_states[k][t - 1]
            except Exception:
                if t > 0:
                    for k in true_states:
                        true_states[k][t] = true_states[k][t - 1]

        multi_rate_data = self.measurement_sim.generate_multi_rate_sequence(true_states)

        final_topology = self._get_topology(topo_manager)

        return {
            "true_states": true_states,
            "parameters": parameters,
            "multi_rate_measurements": multi_rate_data,
            "topology": final_topology,
            "topology_events": topology_events,
            "topology_change_mask": topo_change_mask,
            "scenario_idx": scenario_idx,
            "system_name": self.system_name,
            "num_buses": self.num_buses,
            "num_lines": self.num_lines,
        }

    def generate_dataset(
        self, save_path: Optional[str] = None, topology_mix: bool = True
    ) -> List[Dict]:
        """Generate complete dataset with optional topology scenario mixing."""
        print(
            "\nGenerating %d scenarios for %s..."
            % (self.num_scenarios, self.system_name)
        )

        dataset = []
        topo_types = ["single_switch", "multi_switch", "cascading", "reconfiguration"]

        for idx in tqdm(range(self.num_scenarios)):
            if topology_mix and self.topology_change_enabled:
                self.topology_scenario_type = topo_types[idx % len(topo_types)]
            scenario = self.generate_scenario(idx)
            dataset.append(scenario)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(dataset, f)
            print("\nDataset saved to: %s" % save_path)

        self._print_statistics(dataset)
        return dataset

    def _print_statistics(self, dataset: List[Dict]):
        """Print dataset statistics."""
        all_vmag = np.concatenate([d["true_states"]["v_mag"] for d in dataset])
        print("\nDataset Statistics (%s):" % self.system_name)
        print("  Scenarios: %d" % len(dataset))
        print("  V_mag: %.4f +/- %.4f p.u." % (all_vmag.mean(), all_vmag.std()))
        print("  V_mag range: [%.4f, %.4f] p.u." % (all_vmag.min(), all_vmag.max()))
        n_topo = sum(1 for d in dataset if len(d.get("topology_events", [])) > 0)
        print("  Topology changes: %d/%d scenarios" % (n_topo, len(dataset)))


if __name__ == "__main__":
    gen = PowerGridDataGeneratorV2(
        system_name="ieee123",
        num_scenarios=10,
        time_steps=288,
        pmu_coverage=0.25,
        scada_coverage=0.6,
        topology_change_enabled=True,
        topology_scenario_type="single_switch",
        seed=42,
    )
    dataset = gen.generate_dataset(save_path="data/raw/ieee123_dataset.pkl")
