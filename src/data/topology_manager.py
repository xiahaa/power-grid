"""
Topology Manager for Distribution System State Estimation

Handles dynamic topology changes including:
1. Switch operations (open/close)
2. Line outages (faults)
3. Network reconfiguration (load balancing)
4. Cascading failures

This is a key DSSE challenge: the network topology changes during
operation due to switching events, faults, and maintenance, but
the state estimator must continue providing accurate estimates.
"""

import numpy as np
import pandapower as pp
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import copy


class TopologyEventType(Enum):
    SWITCH_OPEN = "switch_open"
    SWITCH_CLOSE = "switch_close"
    LINE_OUTAGE = "line_outage"
    LINE_RESTORE = "line_restore"
    RECONFIGURATION = "reconfiguration"


@dataclass
class TopologyEvent:
    """A single topology change event."""
    event_type: TopologyEventType
    line_idx: int
    time_step: int
    from_bus: int
    to_bus: int
    description: str = ""


class TopologyManager:
    """
    Manages dynamic topology changes for data generation and inference.

    Supports:
    - Single switch operations
    - Multiple simultaneous switches
    - Cascading events (sequential within time window)
    - Planned reconfiguration scenarios
    """

    IMPEDANCE_DISCONNECT_FACTOR = 1e6

    def __init__(
        self,
        net: pp.pandapowerNet,
        switch_line_indices: Optional[List[int]] = None,
        seed: int = 42
    ):
        """
        Args:
            net: pandapower network
            switch_line_indices: Indices of switchable lines
            seed: Random seed
        """
        self.original_net = copy.deepcopy(net)
        self.net = net
        self.rng = np.random.RandomState(seed)

        self.switch_line_indices = switch_line_indices or []
        self.event_history: List[TopologyEvent] = []

        self.original_r = net.line['r_ohm_per_km'].values.copy()
        self.original_x = net.line['x_ohm_per_km'].values.copy()
        self.is_line_connected = np.ones(len(net.line), dtype=bool)

        self._build_bus_degree_map()

    def _build_bus_degree_map(self):
        """Build mapping of bus -> connected lines for subgraph extraction."""
        self.bus_to_lines: Dict[int, List[int]] = {}
        for idx in self.net.line.index:
            fb = int(self.net.line.at[idx, 'from_bus'])
            tb = int(self.net.line.at[idx, 'to_bus'])
            self.bus_to_lines.setdefault(fb, []).append(idx)
            self.bus_to_lines.setdefault(tb, []).append(idx)

    def get_edge_index(self) -> np.ndarray:
        """Extract current topology as edge index (updated for disconnected lines)."""
        edge_list = []

        for idx in self.net.line.index:
            if not self.is_line_connected[idx]:
                continue

            fb = int(self.net.line.at[idx, 'from_bus'])
            tb = int(self.net.line.at[idx, 'to_bus'])
            r = self.net.line.at[idx, 'r_ohm_per_km']
            x = self.net.line.at[idx, 'x_ohm_per_km']

            edge_list.append([fb, tb])
            edge_list.append([tb, fb])

        if len(edge_list) == 0:
            return np.zeros((2, 0), dtype=np.int64)

        return np.array(edge_list, dtype=np.int64).T

    def get_edge_attr(self) -> np.ndarray:
        """Extract edge attributes for connected lines."""
        attr_list = []

        for idx in self.net.line.index:
            if not self.is_line_connected[idx]:
                continue

            r = self.original_r[idx]
            x = self.original_x[idx]

            attr_list.append([r, x])
            attr_list.append([r, x])

        if len(attr_list) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        return np.array(attr_list, dtype=np.float32)

    def open_line(self, line_idx: int, time_step: int = 0) -> TopologyEvent:
        """
        Open a line (disconnect) by setting impedance very high.

        Args:
            line_idx: Index of line to open
            time_step: When the event occurs

        Returns:
            TopologyEvent describing what happened
        """
        fb = int(self.net.line.at[line_idx, 'from_bus'])
        tb = int(self.net.line.at[line_idx, 'to_bus'])

        self.net.line.at[line_idx, 'r_ohm_per_km'] = (
            self.original_r[line_idx] * self.IMPEDANCE_DISCONNECT_FACTOR
        )
        self.net.line.at[line_idx, 'x_ohm_per_km'] = (
            self.original_x[line_idx] * self.IMPEDANCE_DISCONNECT_FACTOR
        )
        self.is_line_connected[line_idx] = False

        event = TopologyEvent(
            event_type=TopologyEventType.SWITCH_OPEN,
            line_idx=line_idx,
            time_step=time_step,
            from_bus=fb,
            to_bus=tb,
            description=f"Line {line_idx} ({fb}->{tb}) opened"
        )
        self.event_history.append(event)
        return event

    def close_line(self, line_idx: int, time_step: int = 0) -> TopologyEvent:
        """
        Close a line (reconnect) by restoring original impedance.

        Args:
            line_idx: Index of line to close
            time_step: When the event occurs

        Returns:
            TopologyEvent describing what happened
        """
        fb = int(self.net.line.at[line_idx, 'from_bus'])
        tb = int(self.net.line.at[line_idx, 'to_bus'])

        self.net.line.at[line_idx, 'r_ohm_per_km'] = self.original_r[line_idx]
        self.net.line.at[line_idx, 'x_ohm_per_km'] = self.original_x[line_idx]
        self.is_line_connected[line_idx] = True

        event = TopologyEvent(
            event_type=TopologyEventType.SWITCH_CLOSE,
            line_idx=line_idx,
            time_step=time_step,
            from_bus=fb,
            to_bus=tb,
            description=f"Line {line_idx} ({fb}->{tb}) closed"
        )
        self.event_history.append(event)
        return event

    def apply_random_switch_operation(
        self,
        time_step: int
    ) -> Optional[TopologyEvent]:
        """
        Apply a random switch operation if switch lines are available.

        Args:
            time_step: Current time step

        Returns:
            TopologyEvent or None if no switches available
        """
        if len(self.switch_line_indices) == 0:
            return None

        line_idx = self.rng.choice(self.switch_line_indices)

        if self.is_line_connected[line_idx]:
            return self.open_line(line_idx, time_step)
        else:
            return self.close_line(line_idx, time_step)

    def generate_topology_change_scenario(
        self,
        time_steps: int,
        scenario_type: str = "single_switch",
        num_events: int = 1,
        time_window: int = 10
    ) -> List[TopologyEvent]:
        """
        Generate a sequence of topology change events for one scenario.

        Args:
            time_steps: Total number of time steps
            scenario_type: Type of topology change
                "single_switch": One switch operation
                "multi_switch": Multiple simultaneous switches
                "cascading": Sequential switches within time window
                "reconfiguration": Load balancing reconfiguration
            num_events: Number of events for multi/cascading
            time_window: Time window for cascading events

        Returns:
            List of TopologyEvent objects
        """
        self.reset()

        events = []

        if scenario_type == "single_switch":
            t = self.rng.randint(time_steps // 4, 3 * time_steps // 4)
            event = self.apply_random_switch_operation(t)
            if event:
                events.append(event)

        elif scenario_type == "multi_switch":
            n = min(num_events, len(self.switch_line_indices))
            t = self.rng.randint(time_steps // 4, 3 * time_steps // 4)
            chosen = self.rng.choice(
                len(self.switch_line_indices), size=n, replace=False
            )
            for idx in chosen:
                line_idx = self.switch_line_indices[idx]
                if self.is_line_connected[line_idx]:
                    event = self.open_line(line_idx, t)
                    events.append(event)

        elif scenario_type == "cascading":
            n = min(num_events, len(self.switch_line_indices))
            start_t = self.rng.randint(time_steps // 4, time_steps // 2)
            chosen = self.rng.choice(
                len(self.switch_line_indices), size=n, replace=False
            )
            for i, idx in enumerate(chosen):
                t = start_t + i * self.rng.randint(1, max(2, time_window // n))
                if t < time_steps:
                    line_idx = self.switch_line_indices[idx]
                    if self.is_line_connected[line_idx]:
                        event = self.open_line(line_idx, t)
                        events.append(event)

        elif scenario_type == "reconfiguration":
            t = self.rng.randint(time_steps // 3, 2 * time_steps // 3)
            n_switch = min(3, len(self.switch_line_indices))
            chosen = self.rng.choice(
                len(self.switch_line_indices), size=n_switch, replace=False
            )
            for idx in chosen:
                line_idx = self.switch_line_indices[idx]
                if self.is_line_connected[line_idx]:
                    event = self.open_line(line_idx, t)
                else:
                    event = self.close_line(line_idx, t)
                events.append(event)

        return events

    def get_affected_nodes(self, line_idx: int, k_hop: int = 2) -> List[int]:
        """
        Get nodes within k hops of a disconnected line's endpoints.

        Used for incremental model update: only update the subgraph
        around the topology change instead of the entire network.

        Args:
            line_idx: Index of the affected line
            k_hop: Number of hops to expand

        Returns:
            List of affected node indices
        """
        fb = int(self.net.line.at[line_idx, 'from_bus'])
        tb = int(self.net.line.at[line_idx, 'to_bus'])

        affected = set([fb, tb])
        frontier = set([fb, tb])

        for _ in range(k_hop):
            new_frontier = set()
            for bus in frontier:
                if bus in self.bus_to_lines:
                    for line_id in self.bus_to_lines[bus]:
                        if self.is_line_connected[line_id] or line_id == line_idx:
                            other_fb = int(self.net.line.at[line_id, 'from_bus'])
                            other_tb = int(self.net.line.at[line_id, 'to_bus'])
                            neighbor = other_tb if bus == other_fb else other_fb
                            if neighbor not in affected:
                                new_frontier.add(neighbor)
                                affected.add(neighbor)
            frontier = new_frontier

        return sorted(list(affected))

    def reset(self):
        """Reset network to original topology."""
        self.net.line['r_ohm_per_km'] = self.original_r.copy()
        self.net.line['x_ohm_per_km'] = self.original_x.copy()
        self.is_line_connected[:] = True
        self.event_history.clear()

    def get_topology_change_mask(
        self,
        time_steps: int,
        events: List[TopologyEvent]
    ) -> np.ndarray:
        """
        Create a boolean mask indicating which time steps have topology changes.

        Args:
            time_steps: Total time steps
            events: List of topology events

        Returns:
            mask: [time_steps] boolean array
        """
        mask = np.zeros(time_steps, dtype=bool)
        for event in events:
            if 0 <= event.time_step < time_steps:
                mask[event.time_step] = True
        return mask

    def get_edge_change_log(
        self,
        time_steps: int,
        events: List[TopologyEvent]
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Get the edge_index and edge_attr at each topology change point.

        Returns:
            Dict mapping time_step -> (edge_index, edge_attr)
        """
        self.reset()
        change_log = {}

        for t in range(time_steps):
            for event in events:
                if event.time_step == t:
                    if event.event_type == TopologyEventType.SWITCH_OPEN:
                        self.open_line(event.line_idx, t)
                    elif event.event_type == TopologyEventType.SWITCH_CLOSE:
                        self.close_line(event.line_idx, t)

            if any(e.time_step == t for e in events):
                change_log[t] = (
                    self.get_edge_index(),
                    self.get_edge_attr()
                )

        return change_log


if __name__ == "__main__":
    from ieee123_network import build_ieee123_network, get_switch_line_indices

    net = build_ieee123_network()
    switch_indices = get_switch_line_indices(net)

    tm = TopologyManager(net, switch_line_indices=switch_indices, seed=42)

    print("Testing topology scenarios:")

    for scenario in ["single_switch", "multi_switch", "cascading", "reconfiguration"]:
        events = tm.generate_topology_change_scenario(
            time_steps=288,
            scenario_type=scenario,
            num_events=3
        )
        tm.reset()

        print(f"\n  {scenario}: {len(events)} events")
        for e in events:
            print(f"    t={e.time_step}: {e.description}")
            affected = tm.get_affected_nodes(e.line_idx, k_hop=2)
            print(f"    Affected nodes (2-hop): {len(affected)} buses")
        tm.reset()
