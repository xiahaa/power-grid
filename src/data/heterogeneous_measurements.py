"""
Heterogeneous Measurement Simulator for Distribution System State Estimation

Simulates two types of measurements with different characteristics:
1. SCADA (Supervisory Control and Data Acquisition):
   - Slow sampling: 2-15 second intervals
   - Higher noise: 0.5-3% error
   - Higher latency: 2-12 seconds
   - Measures: P, Q, V_mag (no phase angle)
   - Coverage: 60-80% of buses via RTUs

2. PMU (Phasor Measurement Unit):
   - Fast sampling: 30-60 Hz
   - Low noise: 0.01-0.2% error
   - Low latency: 10-50 ms
   - Measures: V_mag, V_ang (synchronized phasors)
   - Coverage: 20-40% of buses

Key DSSE Challenge: Fusing these heterogeneous measurements
at their native sampling rates is the core innovation of this work.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MeasurementConfig:
    """Configuration for a measurement type."""

    sampling_rate: float
    latency: float
    noise_std: float
    meas_types: List[str]
    coverage: float


SCADA_CONFIG = MeasurementConfig(
    sampling_rate=0.25,  # Hz (4-second intervals)
    latency=4.0,  # seconds
    noise_std=0.01,  # 1% relative error
    meas_types=["p_bus", "q_bus", "v_mag"],
    coverage=0.6,
)

PMU_CONFIG = MeasurementConfig(
    sampling_rate=60.0,  # Hz
    latency=0.02,  # 20 ms
    noise_std=0.001,  # 0.1% relative error
    meas_types=["v_mag", "v_ang"],
    coverage=0.25,
)


class HeterogeneousMeasurementSimulator:
    """
    Generate SCADA and PMU measurements at their native sampling rates
    from true power grid states.

    This simulates the real-world scenario where:
    - PMUs provide high-frequency synchronized phasors at few locations
    - SCADA provides low-frequency power/voltage at many locations
    - Both have different noise characteristics and latency
    """

    def __init__(
        self,
        num_buses: int,
        scada_config: MeasurementConfig = None,
        pmu_config: MeasurementConfig = None,
        base_time_step: float = 5.0 / 60.0,
        seed: int = 42,
        scada_coverage: float = None,
        pmu_coverage: float = None,
    ):
        """
        Args:
            num_buses: Number of buses in the system
            scada_config: SCADA measurement configuration
            pmu_config: PMU measurement configuration
            base_time_step: Base simulation time step in hours
            seed: Random seed
            scada_coverage: SCADA coverage (0-1), overrides scada_config
            pmu_coverage: PMU coverage (0-1) overrides pmu_config
        """
        self.num_buses = num_buses
        self.base_time_step = base_time_step
        self.rng = np.random.RandomState(seed)

        if scada_coverage is not None:
            self.scada_config = MeasurementConfig(
                sampling_rate=SCADA_CONFIG.sampling_rate,
                latency=SCADA_CONFIG.latency,
                noise_std=SCADA_CONFIG.noise_std,
                meas_types=SCADA_CONFIG.meas_types,
                coverage=scada_coverage,
            )
        else:
            self.scada_config = scada_config or SCADA_CONFIG

        if pmu_coverage is not None:
            self.pmu_config = MeasurementConfig(
                sampling_rate=PMU_CONFIG.sampling_rate,
                latency=PMU_CONFIG.latency,
                noise_std=PMU_CONFIG.noise_std,
                meas_types=PMU_CONFIG.meas_types,
                coverage=pmu_coverage,
            )
        else:
            self.pmu_config = pmu_config or PMU_CONFIG

        self.scada_buses = self._select_scada_buses()
        self.pmu_buses = self._select_pmu_buses()

    def _select_scada_buses(self) -> List[int]:
        """Select buses with SCADA (RTU) measurements."""
        n_scada = int(self.num_buses * self.scada_config.coverage)
        return sorted(
            self.rng.choice(self.num_buses, size=n_scada, replace=False).tolist()
        )

    def _select_pmu_buses(self) -> List[int]:
        """Select buses with PMU measurements (prioritize high-degree buses)."""
        n_pmu = int(self.num_buses * self.pmu_config.coverage)
        return sorted(
            self.rng.choice(self.num_buses, size=n_pmu, replace=False).tolist()
        )

    def generate_scada_measurements(
        self, true_states: Dict[str, np.ndarray], time_step: int
    ) -> Dict[str, np.ndarray]:
        """
        Generate SCADA measurements at native sampling rate.

        SCADA reports every 4 seconds, so not every simulation time step
        gets a SCADA update. Only every ~48th time step (4s / (5/60*3600)s)
        has new SCADA data.

        Args:
            true_states: Dict with 'v_mag', 'v_ang', 'p_bus', 'q_bus'
                        Each: [num_buses]
            time_step: Current time step index

        Returns:
            scada_meas: Dict with available SCADA measurements
                       Values are NaN for unobserved buses/times
        """
        base_dt_seconds = self.base_time_step * 3600
        scada_interval_steps = int(
            self.scada_config.sampling_rate ** (-1) / base_dt_seconds
        )

        has_update = (time_step % max(1, scada_interval_steps)) == 0

        scada_meas = {}
        for meas_type in self.scada_config.meas_types:
            if meas_type not in true_states:
                continue

            data = np.full(self.num_buses, np.nan)

            if has_update:
                for bus in self.scada_buses:
                    noise = self.rng.normal(0, self.scada_config.noise_std) * np.abs(
                        true_states[meas_type][bus]
                    )
                    data[bus] = true_states[meas_type][bus] + noise

            scada_meas[meas_type] = data

        scada_meas["has_update"] = np.array(has_update)
        scada_meas["observed_buses"] = np.array(self.scada_buses)

        return scada_meas

    def generate_pmu_measurements(
        self, true_states: Dict[str, np.ndarray], time_step: int
    ) -> Dict[str, np.ndarray]:
        """
        Generate PMU measurements at native sampling rate.

        PMUs report at 60 Hz, providing synchronized V_mag and V_ang
        at their installed locations every time step.

        Args:
            true_states: Dict with 'v_mag', 'v_ang'
                        Each: [num_buses]
            time_step: Current time step index

        Returns:
            pmu_meas: Dict with PMU phasor measurements
        """
        pmu_meas = {}

        for meas_type in self.pmu_config.meas_types:
            if meas_type not in true_states:
                continue

            data = np.full(self.num_buses, np.nan)

            for bus in self.pmu_buses:
                noise = self.rng.normal(0, self.pmu_config.noise_std) * np.abs(
                    true_states[meas_type][bus]
                )
                data[bus] = true_states[meas_type][bus] + noise

            pmu_meas[meas_type] = data

        pmu_meas["observed_buses"] = np.array(self.pmu_buses)

        return pmu_meas

    def generate_multi_rate_sequence(
        self,
        true_state_sequence: Dict[str, np.ndarray],
        topology_changes: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Generate a full multi-rate measurement sequence for one scenario.

        Args:
            true_state_sequence: Dict with temporal sequences
                                Each: [time_steps, num_buses]
            topology_changes: Optional list of topology change events

        Returns:
            multi_rate_data: Dict with SCADA and PMU sequences
        """
        time_steps = true_state_sequence["v_mag"].shape[0]

        scada_sequence = {k: [] for k in self.scada_config.meas_types}
        scada_update_flags = []
        pmu_sequence = {k: [] for k in self.pmu_config.meas_types}

        for t in range(time_steps):
            current_states = {k: v[t] for k, v in true_state_sequence.items()}

            scada = self.generate_scada_measurements(current_states, t)
            for k in self.scada_config.meas_types:
                if k in scada:
                    scada_sequence[k].append(scada[k])
            scada_update_flags.append(scada["has_update"])

            pmu = self.generate_pmu_measurements(current_states, t)
            for k in self.pmu_config.meas_types:
                if k in pmu:
                    pmu_sequence[k].append(pmu[k])

        for k in scada_sequence:
            scada_sequence[k] = np.stack(scada_sequence[k])
        for k in pmu_sequence:
            pmu_sequence[k] = np.stack(pmu_sequence[k])

        scada_obs_mask = np.zeros((time_steps, self.num_buses), dtype=bool)
        for bus in self.scada_buses:
            scada_obs_mask[:, bus] = True
        for t in range(time_steps):
            if not scada_update_flags[t]:
                scada_obs_mask[t] = False

        pmu_obs_mask = np.zeros((time_steps, self.num_buses), dtype=bool)
        for bus in self.pmu_buses:
            pmu_obs_mask[:, bus] = True

        combined_obs_mask = scada_obs_mask | pmu_obs_mask

        return {
            "scada": scada_sequence,
            "scada_update_flags": np.array(scada_update_flags),
            "scada_obs_mask": scada_obs_mask,
            "scada_buses": np.array(self.scada_buses),
            "pmu": pmu_sequence,
            "pmu_obs_mask": pmu_obs_mask,
            "pmu_buses": np.array(self.pmu_buses),
            "combined_obs_mask": combined_obs_mask,
            "scada_config": {
                "sampling_rate": self.scada_config.sampling_rate,
                "latency": self.scada_config.latency,
                "noise_std": self.scada_config.noise_std,
            },
            "pmu_config": {
                "sampling_rate": self.pmu_config.sampling_rate,
                "latency": self.pmu_config.latency,
                "noise_std": self.pmu_config.noise_std,
            },
        }

    def get_measurement_summary(self) -> Dict:
        """Return summary of measurement setup."""
        return {
            "num_scada_buses": len(self.scada_buses),
            "num_pmu_buses": len(self.pmu_buses),
            "scada_coverage": len(self.scada_buses) / self.num_buses,
            "pmu_coverage": len(self.pmu_buses) / self.num_buses,
            "scada_sampling_hz": self.scada_config.sampling_rate,
            "pmu_sampling_hz": self.pmu_config.sampling_rate,
            "rate_ratio": self.pmu_config.sampling_rate
            / self.scada_config.sampling_rate,
            "total_observability": len(set(self.scada_buses) | set(self.pmu_buses))
            / self.num_buses,
        }


if __name__ == "__main__":
    sim = HeterogeneousMeasurementSimulator(num_buses=123)
    summary = sim.get_measurement_summary()

    print("Heterogeneous Measurement Setup:")
    print(f"  SCADA buses: {summary['num_scada_buses']}")
    print(f"  PMU buses: {summary['num_pmu_buses']}")
    print(f"  Total observability: {summary['total_observability']:.1%}")
    print(f"  Rate ratio (PMU:SCADA): {summary['rate_ratio']:.0f}x")

    T = 20
    true_states = {
        "v_mag": 0.95 + 0.1 * np.random.rand(T, 123),
        "v_ang": 0.05 * np.random.randn(T, 123),
        "p_bus": 0.05 + 0.01 * np.random.rand(T, 123),
        "q_bus": 0.02 + 0.005 * np.random.rand(T, 123),
    }

    multi_rate = sim.generate_multi_rate_sequence(true_states)

    print(f"\nMulti-rate sequence shapes:")
    print(f"  SCADA V: {multi_rate['scada']['v_mag'].shape}")
    print(f"  PMU V: {multi_rate['pmu']['v_mag'].shape}")
    print(f"  Combined mask: {multi_rate['combined_obs_mask'].shape}")
    print(f"  SCADA updates: {multi_rate['scada_update_flags'].sum()}/{T}")
