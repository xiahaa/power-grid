"""
Test the fixed data generator

Author: Assistant
Date: 2026-01-22
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from data.data_generator import PowerGridDataGenerator

def test_fixed_generator():
    """Test that the fixed generator produces good data"""

    print("="*70)
    print("Testing FIXED Data Generator")
    print("="*70)

    # Create generator
    generator = PowerGridDataGenerator(
        system_name="ieee33",
        num_scenarios=2,
        time_steps=50,  # Shorter for testing
        pmu_coverage=0.3,
        parameter_drift_enabled=True,
        pv_penetration=0.4,
        seed=42
    )

    # Generate one scenario
    print("\nGenerating test scenario...")
    scenario = generator.generate_scenario(0)

    # Analyze voltage profiles over time
    v_mag = scenario['true_states']['v_mag']

    print(f"\n{'='*70}")
    print("Voltage Profile Analysis")
    print("="*70)

    timesteps_to_check = [0, 5, 10, 20, 40, 49]
    good_timesteps = 0

    for t in timesteps_to_check:
        v_std = v_mag[t].std()
        v_min = v_mag[t].min()
        v_max = v_mag[t].max()
        v_mean = v_mag[t].mean()

        status = "✓ GOOD" if v_std > 0.001 else "✗ BAD (flat)"
        if v_std > 0.001:
            good_timesteps += 1

        print(f"t={t:3d}: mean={v_mean:.6f}, std={v_std:.6f}, range=[{v_min:.6f}, {v_max:.6f}] {status}")

    print(f"\n{'='*70}")
    print(f"Summary: {good_timesteps}/{len(timesteps_to_check)} timesteps have good variation")
    print("="*70)

    # Check loads over time
    print(f"\n{'='*70}")
    print("Load Profile Analysis")
    print("="*70)

    measurements = scenario['measurements']
    p_bus = measurements['p_bus']

    for t in timesteps_to_check:
        total_p = np.abs(p_bus[t]).sum()
        print(f"t={t:3d}: Total load = {total_p:.4f} MW")

    # Overall statistics
    print(f"\n{'='*70}")
    print("Overall Statistics")
    print("="*70)

    all_v_stds = [v_mag[t].std() for t in range(len(v_mag))]
    flat_count = sum(1 for std in all_v_stds if std < 1e-6)
    low_var_count = sum(1 for std in all_v_stds if std < 0.001)
    good_count = sum(1 for std in all_v_stds if std >= 0.001)

    print(f"Total timesteps: {len(v_mag)}")
    print(f"Completely flat (std < 1e-6): {flat_count} ({flat_count/len(v_mag)*100:.1f}%)")
    print(f"Low variation (std < 0.001): {low_var_count} ({low_var_count/len(v_mag)*100:.1f}%)")
    print(f"Good variation (std >= 0.001): {good_count} ({good_count/len(v_mag)*100:.1f}%)")
    print(f"\nAverage V_mag std: {np.mean(all_v_stds):.6f}")
    print(f"Min V_mag std: {np.min(all_v_stds):.6f}")
    print(f"Max V_mag std: {np.max(all_v_stds):.6f}")

    print(f"\nVoltage magnitude range: [{v_mag.min():.6f}, {v_mag.max():.6f}] p.u.")
    print(f"Voltage magnitude std (all): {v_mag.std():.6f} p.u.")

    # Success criteria
    print(f"\n{'='*70}")
    print("Success Criteria")
    print("="*70)

    criteria = {
        "< 10% flat profiles": flat_count < len(v_mag) * 0.1,
        "> 80% good variation": good_count > len(v_mag) * 0.8,
        "Avg std > 0.001": np.mean(all_v_stds) > 0.001,
        "V_mag range > 0.05": (v_mag.max() - v_mag.min()) > 0.05,
    }

    all_passed = True
    for criterion, passed in criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{criterion:30s}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED! Data generator is working correctly.")
    else:
        print(f"\n⚠️  SOME TESTS FAILED. Data quality may still be insufficient.")

    return all_passed

if __name__ == "__main__":
    np.random.seed(42)
    success = test_fixed_generator()
    sys.exit(0 if success else 1)
