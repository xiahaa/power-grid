"""
Test SwanLab integration

This script verifies that SwanLab is correctly installed and configured.

Usage:
    python scripts/test_swanlab.py
"""

import sys
sys.path.append('src')

def test_swanlab_installation():
    """Test if SwanLab is installed"""
    print("Testing SwanLab installation...")

    try:
        import swanlab
        print(f"✓ SwanLab installed (version {swanlab.__version__})")
        return True
    except ImportError:
        print("✗ SwanLab not installed")
        print("  Install with: pip install swanlab")
        return False


def test_swanlab_init():
    """Test SwanLab initialization"""
    print("\nTesting SwanLab initialization...")

    try:
        import swanlab

        # Initialize test run
        run = swanlab.init(
            project="swanlab-test",
            experiment_name="integration-test",
            config={
                'test_param': 42,
                'model': 'graph-mamba'
            },
            mode="disabled"  # Don't actually upload
        )

        print("✓ SwanLab initialization successful")

        # Test logging
        swanlab.log({
            'test_metric': 1.23,
            'accuracy': 0.95
        }, step=1)

        print("✓ SwanLab logging successful")

        # Finish
        swanlab.finish()
        print("✓ SwanLab finish successful")

        return True

    except Exception as e:
        print(f"✗ SwanLab test failed: {e}")
        return False


def test_config_integration():
    """Test SwanLab config file integration"""
    print("\nTesting config file integration...")

    try:
        from utils.utils import load_config

        config = load_config("configs/ieee33_config.yaml")

        # Check SwanLab settings
        if 'use_swanlab' in config['logging']:
            use_swanlab = config['logging']['use_swanlab']
            project = config['logging'].get('swanlab_project', 'N/A')
            experiment = config['logging'].get('swanlab_experiment', 'N/A')

            print(f"✓ Config loaded successfully")
            print(f"  use_swanlab: {use_swanlab}")
            print(f"  swanlab_project: {project}")
            print(f"  swanlab_experiment: {experiment}")

            return True
        else:
            print("⚠ SwanLab settings not found in config")
            print("  Add the following to your config file:")
            print("  logging:")
            print("    use_swanlab: true")
            print("    swanlab_project: 'power-grid-estimation'")
            print("    swanlab_experiment: 'ieee33-graph-mamba'")
            return False

    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("SwanLab Integration Test")
    print("="*60)

    results = []

    # Test 1: Installation
    results.append(test_swanlab_installation())

    if results[0]:
        # Test 2: Initialization
        results.append(test_swanlab_init())

        # Test 3: Config integration
        results.append(test_config_integration())

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    test_names = [
        "SwanLab Installation",
        "SwanLab Initialization",
        "Config Integration"
    ]

    for i, (name, result) in enumerate(zip(test_names[:len(results)], results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {name}: {status}")

    all_passed = all(results)

    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed! SwanLab is ready to use.")
        print("\nNext steps:")
        print("1. Train a model: python scripts/train.py --config configs/ieee33_config.yaml")
        print("2. View dashboard: https://swanlab.cn")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("1. Install SwanLab: pip install swanlab")
        print("2. Login to SwanLab: swanlab login")
        print("3. Check config file: configs/ieee33_config.yaml")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
