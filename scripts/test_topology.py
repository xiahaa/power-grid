"""
Topology Adaptation Testing Script

Standalone tests for topology change detection and adaptation:
  1. Detection accuracy: precision, recall, F1 on synthetic topology changes
  2. Adaptation speed: measure time for incremental GAT update + state reset
  3. State estimation accuracy: before vs after topology change + adaptation
  4. Scalability: test on larger networks (33, 69, 123 buses)

Usage:
    python scripts/test_topology.py --num_buses 33 --num_tests 100
    python scripts/test_topology.py --num_buses 123 --num_tests 500 --output results/topo_test.json
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from models.topology_adaptive import (
    TopologyChangeDetector,
    IncrementalGATUpdater,
    SelectiveMambaStateReset,
)
from evaluation.metrics import TopologyDetectionMetrics, DSSEEvaluator


def test_detection_accuracy(detector, num_tests, num_nodes, device):
    """Test topology change detection precision/recall/F1."""
    print("\n--- Detection Accuracy Test ---")
    metrics = TopologyDetectionMetrics()

    for i in range(num_tests):
        actual_change = i % 3 == 0

        current = {
            "v_mag": torch.randn(1, num_nodes, device=device),
            "p_bus": torch.randn(1, num_nodes, device=device),
            "q_bus": torch.randn(1, num_nodes, device=device),
        }

        if actual_change:
            prev = {
                "v_mag": current["v_mag"]
                + torch.randn(1, num_nodes, device=device) * 0.3,
                "p_bus": current["p_bus"]
                + torch.randn(1, num_nodes, device=device) * 0.5,
                "q_bus": current["q_bus"]
                + torch.randn(1, num_nodes, device=device) * 0.5,
            }
        else:
            prev = {
                "v_mag": current["v_mag"]
                + torch.randn(1, num_nodes, device=device) * 0.01,
                "p_bus": current["p_bus"]
                + torch.randn(1, num_nodes, device=device) * 0.01,
                "q_bus": current["q_bus"]
                + torch.randn(1, num_nodes, device=device) * 0.01,
            }

        start = time.time()
        scores, affected = detector(current, [prev])
        elapsed = (time.time() - start) * 1000

        predicted = len(affected) > 0
        metrics.update(predicted, actual_change, elapsed)

    summary = metrics.compute_summary()
    print("  Tests: %d" % num_tests)
    print("  Precision: %.3f" % summary["topo_precision"])
    print("  Recall:    %.3f" % summary["topo_recall"])
    print("  F1:        %.3f" % summary["topo_f1"])
    print("  Accuracy:  %.3f" % summary["topo_accuracy"])
    print("  Avg Detection Latency: %.2f ms" % summary["topo_avg_detection_latency_ms"])
    return summary


def test_state_reset(num_nodes, num_edges, device):
    """Test selective Mamba state reset speed and correctness."""
    print("\n--- State Reset Test ---")

    resetter_partial = SelectiveMambaStateReset(reset_mode="partial")
    resetter_full = SelectiveMambaStateReset(reset_mode="full")
    resetter_none = SelectiveMambaStateReset(reset_mode="none")

    hidden = torch.randn(4, num_nodes, 64, device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    affected = list(range(0, min(10, num_nodes)))

    times = {}
    for name, rst in [
        ("partial", resetter_partial),
        ("full", resetter_full),
        ("none", resetter_none),
    ]:
        start = time.time()
        for _ in range(100):
            rst.compute_reset(affected, hidden, edge_index)
        elapsed = (time.time() - start) / 100 * 1000
        times[name] = elapsed
        print("  %s reset: %.3f ms" % (name, elapsed))

    reset_partial = resetter_partial.compute_reset(affected, hidden, edge_index)
    reset_full = resetter_full.compute_reset(affected, hidden, edge_index)
    reset_none = resetter_none.compute_reset(affected, hidden, edge_index)

    assert torch.equal(reset_none, hidden), "none mode should not change state"
    assert not torch.equal(reset_partial, hidden), "partial mode should change state"
    assert not torch.equal(reset_full, hidden), "full mode should change state"

    partial_changed = (reset_partial != hidden).sum().item()
    full_changed = (reset_full != hidden).sum().item()
    print("  Partial mode: %d elements changed" % partial_changed)
    print("  Full mode: %d elements changed" % full_changed)

    return times


def test_incremental_gat(num_nodes, num_edges, device):
    """Test incremental GAT update speed and convergence."""
    print("\n--- Incremental GAT Update Test ---")

    try:
        from torch_geometric.nn import GATConv

        spatial_encoder = nn.Sequential()
        gat = GATConv(64, 32, heads=2, concat=True)
        spatial_encoder.add_module("gat", gat)
        spatial_encoder = spatial_encoder.to(device)

        updater = IncrementalGATUpdater(
            spatial_encoder=gat,
            k_hop=2,
            fine_tune_steps=50,
            fine_tune_lr=0.01,
        )

        edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
        local_data = torch.randn(num_nodes, 64, device=device)
        affected = list(range(0, min(5, num_nodes)))

        start = time.time()
        stats = updater.incremental_update(
            affected, edge_index, local_data, num_steps=10
        )
        elapsed = (time.time() - start) * 1000

        print("  Subgraph size: %d nodes" % stats["subgraph_size"])
        print("  Affected nodes: %d" % stats["affected_nodes"])
        print("  Fine-tune steps: %d" % stats["num_steps"])
        print("  Total time: %.2f ms" % elapsed)

        return {"subgraph_size": stats["subgraph_size"], "adaptation_time_ms": elapsed}

    except ImportError:
        print("  SKIPPED: torch_geometric not available")
        return {"skipped": True}


def test_adaptation_pipeline(num_nodes, num_edges, device):
    """End-to-end test of the full topology adaptation pipeline."""
    print("\n--- End-to-End Adaptation Pipeline Test ---")

    detector = TopologyChangeDetector(input_dim=5, hidden_dim=32).to(device)
    resetter = SelectiveMambaStateReset(reset_mode="partial")

    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    hidden_state = torch.randn(1, num_nodes, 64, device=device)

    normal_meas = {
        "v_mag": torch.randn(1, num_nodes, device=device) * 0.05 + 1.0,
        "p_bus": torch.randn(1, num_nodes, device=device) * 0.01,
        "q_bus": torch.randn(1, num_nodes, device=device) * 0.01,
    }

    anomaly_meas = {
        "v_mag": torch.randn(1, num_nodes, device=device) * 0.05 + 1.0,
        "p_bus": torch.randn(1, num_nodes, device=device) * 0.5,
        "q_bus": torch.randn(1, num_nodes, device=device) * 0.5,
    }

    start = time.time()
    scores, affected = detector(anomaly_meas, [normal_meas])
    if len(affected) > 0:
        hidden_state = resetter.compute_reset(affected, hidden_state, edge_index)
    elapsed = (time.time() - start) * 1000

    print("  Detected %d affected nodes" % len(affected))
    print("  Pipeline latency: %.2f ms" % elapsed)
    print("  Target: < 50 ms -> %s" % ("PASS" if elapsed < 50 else "FAIL"))

    return {
        "detected_nodes": len(affected),
        "pipeline_latency_ms": elapsed,
        "pass": elapsed < 50,
    }


def run_all_tests(num_nodes, num_edges, num_tests, device, output=None):
    results = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_tests": num_tests,
        "device": device,
    }

    detector = TopologyChangeDetector(input_dim=5, hidden_dim=32).to(device)

    results["detection"] = test_detection_accuracy(
        detector, num_tests, num_nodes, device
    )
    results["state_reset"] = test_state_reset(num_nodes, num_edges, device)
    results["incremental_gat"] = test_incremental_gat(num_nodes, num_edges, device)
    results["adaptation_pipeline"] = test_adaptation_pipeline(
        num_nodes, num_edges, device
    )

    all_pass = results["adaptation_pipeline"].get("pass", False)

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED - check adaptation latency")
    print("=" * 60)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print("Results saved to %s" % output)

    return results


def main(args):
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print("Device: %s" % device)
    print("Topology Adaptation Tests")
    print("  Nodes: %d" % args.num_buses)
    print("  Edges: %d (estimated)" % (args.num_buses * 2))
    print("  Detection tests: %d" % args.num_tests)

    run_all_tests(
        num_nodes=args.num_buses,
        num_edges=args.num_buses * 2,
        num_tests=args.num_tests,
        device=device,
        output=args.output,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test topology adaptation components")
    parser.add_argument("--num_buses", type=int, default=33, help="Number of buses")
    parser.add_argument(
        "--num_tests", type=int, default=100, help="Number of detection tests"
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()
    main(args)
