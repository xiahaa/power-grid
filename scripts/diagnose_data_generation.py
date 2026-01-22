"""
Diagnostic script for data generation issues

Author: Assistant
Date: 2026-01-22
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandapower as pp
import pandapower.networks as pn

def test_single_scenario():
    """Test data generation for one scenario to identify issues"""

    # Load network
    net = pn.case33bw()

    print("="*60)
    print("IEEE 33-Bus Network Analysis")
    print("="*60)

    print(f"\nNetwork Overview:")
    print(f"  Number of buses: {len(net.bus)}")
    print(f"  Number of lines: {len(net.line)}")
    print(f"  Number of loads: {len(net.load)}")

    # Check initial loads
    print(f"\nInitial Load Statistics:")
    print(f"  Total P load: {net.load['p_mw'].sum():.4f} MW")
    print(f"  Total Q load: {net.load['q_mvar'].sum():.4f} MVAr")
    print(f"  Min P load: {net.load['p_mw'].min():.4f} MW")
    print(f"  Max P load: {net.load['p_mw'].max():.4f} MW")
    print(f"  Load per bus (sample):")
    for idx in range(min(5, len(net.load))):
        print(f"    Bus {net.load.iloc[idx]['bus']}: P={net.load.iloc[idx]['p_mw']:.4f} MW")

    # Run initial power flow
    print(f"\n" + "="*60)
    print("Initial Power Flow (no modifications)")
    print("="*60)
    try:
        pp.runpp(net, algorithm='nr', max_iteration=50)
        print("✓ Power flow converged")
        print(f"  V_mag range: [{net.res_bus['vm_pu'].min():.6f}, {net.res_bus['vm_pu'].max():.6f}]")
        print(f"  V_mag std: {net.res_bus['vm_pu'].std():.6f}")
        print(f"  Slack bus power: P={net.res_ext_grid['p_mw'].values[0]:.4f} MW")
    except Exception as e:
        print(f"✗ Power flow FAILED: {e}")

    # Add PV systems (simulating current implementation)
    print(f"\n" + "="*60)
    print("Adding PV Systems (40% penetration)")
    print("="*60)

    pv_penetration = 0.4
    num_buses = len(net.bus)
    num_pv_buses = int(num_buses * pv_penetration)
    available_buses = [b for b in range(num_buses) if b != net.ext_grid.at[0, 'bus']]
    pv_buses = np.random.choice(available_buses, size=num_pv_buses, replace=False)

    for bus in pv_buses:
        bus_load = net.load[net.load.bus == bus]['p_mw'].sum()
        if bus_load > 0:
            # This is the BUG - base_pv changes every timestep!
            pv_capacity = bus_load * np.random.uniform(0.5, 2.0)
            pp.create_sgen(net, bus=int(bus), p_mw=0, q_mvar=0, name=f"PV_{bus}")
            print(f"  Bus {bus}: Load={bus_load:.4f} MW, PV capacity={pv_capacity:.4f} MW")

    print(f"\nCreated {len(net.sgen)} PV systems")

    # Simulate time series
    print(f"\n" + "="*60)
    print("Simulating Time Series")
    print("="*60)

    failures = []
    v_mag_stds = []

    for t in [0, 1, 5, 10, 50, 100, 200]:
        hour = (t * 5 / 60) % 24

        # Update loads (from data generator)
        for idx, load in net.load.iterrows():
            base_p = net.load.at[idx, 'p_mw']
            base_q = net.load.at[idx, 'q_mvar']

            daily_pattern = (
                0.6 +
                0.3 * np.sin(2 * np.pi * (hour - 6) / 24) +
                0.1 * np.sin(4 * np.pi * (hour - 9) / 24)
            )
            noise = np.random.normal(0, 0.05)
            load_factor = np.clip(daily_pattern + noise, 0.3, 1.2)

            net.load.at[idx, 'p_mw'] = base_p * load_factor
            net.load.at[idx, 'q_mvar'] = base_q * load_factor

        # Update PV (BUG: regenerates base_pv every time!)
        for idx, sgen in net.sgen.iterrows():
            bus = net.sgen.at[idx, 'bus']
            bus_load = net.load[net.load.bus == bus]['p_mw'].sum()
            base_pv = bus_load * np.random.uniform(0.5, 2.0)  # BUG!

            if 6 <= hour <= 18:
                solar_pattern = np.sin(np.pi * (hour - 6) / 12) ** 2
                cloud_factor = np.random.uniform(0.7, 1.0)
                pv_output = base_pv * solar_pattern * cloud_factor
            else:
                pv_output = 0.0

            net.sgen.at[idx, 'p_mw'] = pv_output

        total_load = net.load['p_mw'].sum()
        total_pv = net.sgen['p_mw'].sum()
        net_load = total_load - total_pv

        # Try power flow
        try:
            pp.runpp(net, algorithm='nr', max_iteration=50)
            v_std = net.res_bus['vm_pu'].std()
            v_mag_stds.append(v_std)
            status = "✓ CONVERGED"
        except Exception as e:
            failures.append(t)
            v_std = 0.0
            v_mag_stds.append(0.0)
            status = f"✗ FAILED: {str(e)[:50]}"

        print(f"\nt={t:3d}, hour={hour:5.2f}: {status}")
        print(f"  Load: {total_load:6.2f} MW, PV: {total_pv:6.2f} MW, Net: {net_load:6.2f} MW")
        print(f"  V_mag std: {v_std:.6f}")
        if v_std < 1e-6:
            print(f"  ⚠️  WARNING: Flat voltage profile!")

    print(f"\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Failed timesteps: {len(failures)}/{7} = {len(failures)/7*100:.1f}%")
    print(f"Flat profiles (std < 1e-6): {sum(1 for std in v_mag_stds if std < 1e-6)}/{7}")
    print(f"Average V_mag std: {np.mean([s for s in v_mag_stds if s > 0]):.6f}")

if __name__ == "__main__":
    np.random.seed(42)
    test_single_scenario()
