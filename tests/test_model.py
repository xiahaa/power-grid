"""
Unit tests for Graph Mamba model

Run with: pytest tests/test_model.py -v
"""

import sys
sys.path.append('src')

import torch
import pytest
from models.graph_mamba import GraphMamba, SpatialEncoder, MambaBlock
from physics.constraints import PhysicsInformedLayer, PowerFlowConstraints


class TestSpatialEncoder:
    """Test GAT spatial encoder"""

    def test_forward_pass(self):
        """Test forward pass with dummy data"""
        encoder = SpatialEncoder(
            in_channels=64,
            hidden_dim=32,
            num_heads=4,
            num_layers=2
        )

        batch_size, num_nodes = 4, 33
        x = torch.randn(batch_size, num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, 64))

        output = encoder(x, edge_index)

        assert output.shape == (batch_size, num_nodes, 32 * 4)

    def test_output_range(self):
        """Test output is finite"""
        encoder = SpatialEncoder(in_channels=64, hidden_dim=32)

        x = torch.randn(2, 10, 64)
        edge_index = torch.randint(0, 10, (2, 20))

        output = encoder(x, edge_index)

        assert torch.isfinite(output).all()


class TestMambaBlock:
    """Test Mamba temporal encoder"""

    def test_forward_pass(self):
        """Test forward pass"""
        mamba = MambaBlock(
            d_model=64,
            d_state=16,
            num_layers=2
        )

        batch_size, seq_len = 4, 10
        x = torch.randn(batch_size, seq_len, 64)

        output = mamba(x)

        assert output.shape == (batch_size, seq_len, 64)

    def test_long_sequence(self):
        """Test with long sequence"""
        mamba = MambaBlock(d_model=64)

        x = torch.randn(2, 100, 64)
        output = mamba(x)

        assert output.shape == (2, 100, 64)
        assert torch.isfinite(output).all()


class TestGraphMamba:
    """Test full Graph Mamba model"""

    def test_forward_pass(self):
        """Test complete forward pass"""
        model = GraphMamba(
            num_nodes=33,
            num_edges=64,
            input_dim=3
        )

        batch_size, seq_len = 4, 10
        measurements = {
            'v_mag': torch.randn(batch_size, seq_len, 33),
            'p_bus': torch.randn(batch_size, seq_len, 33),
            'q_bus': torch.randn(batch_size, seq_len, 33),
        }
        edge_index = torch.randint(0, 33, (2, 64))

        states, parameters = model(measurements, edge_index)

        assert states['v_mag'].shape == (batch_size, 33)
        assert states['v_ang'].shape == (batch_size, 33)
        assert parameters['r_line'].shape == (batch_size, 64)
        assert parameters['x_line'].shape == (batch_size, 64)

    def test_output_constraints(self):
        """Test output satisfies physical constraints"""
        model = GraphMamba(num_nodes=33, num_edges=64)

        measurements = {
            'v_mag': torch.randn(2, 10, 33),
            'p_bus': torch.randn(2, 10, 33),
            'q_bus': torch.randn(2, 10, 33),
        }
        edge_index = torch.randint(0, 33, (2, 64))

        states, parameters = model(measurements, edge_index)

        # Voltage magnitude should be in reasonable range
        assert (states['v_mag'] >= 0.85).all()
        assert (states['v_mag'] <= 1.15).all()

        # Parameters should be positive
        assert (parameters['r_line'] >= 0).all()
        assert (parameters['x_line'] >= 0).all()


class TestPhysicsConstraints:
    """Test physics-informed layer"""

    def test_power_mismatch(self):
        """Test power flow mismatch calculation"""
        batch_size, num_nodes, num_edges = 2, 33, 64

        v_mag = torch.rand(batch_size, num_nodes) * 0.2 + 0.9
        v_ang = torch.randn(batch_size, num_nodes) * 0.1
        r_line = torch.rand(batch_size, num_edges) * 0.5
        x_line = torch.rand(batch_size, num_edges) * 0.5
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        p_mismatch, q_mismatch = PowerFlowConstraints.compute_power_mismatch(
            v_mag, v_ang, edge_index, r_line, x_line
        )

        assert p_mismatch.shape == (batch_size, num_nodes)
        assert q_mismatch.shape == (batch_size, num_nodes)
        assert torch.isfinite(p_mismatch).all()

    def test_soft_constraints(self):
        """Test soft constraint mode"""
        physics_layer = PhysicsInformedLayer(constraint_type="soft")

        states = {
            'v_mag': torch.rand(2, 33),
            'v_ang': torch.randn(2, 33) * 0.1
        }
        parameters = {
            'r_line': torch.rand(2, 64),
            'x_line': torch.rand(2, 64)
        }
        edge_index = torch.randint(0, 33, (2, 64))

        corrected_states, loss = physics_layer(states, parameters, edge_index)

        assert loss >= 0
        assert torch.isfinite(loss)


class TestIntegration:
    """Integration tests"""

    def test_end_to_end(self):
        """Test complete pipeline"""
        from models.graph_mamba import GraphMamba
        from physics.constraints import PhysicsInformedLayer, PhysicsInformedGraphMamba

        graph_mamba = GraphMamba(num_nodes=33, num_edges=64)
        physics_layer = PhysicsInformedLayer(constraint_type="soft")
        model = PhysicsInformedGraphMamba(graph_mamba, physics_layer)

        measurements = {
            'v_mag': torch.randn(2, 10, 33),
            'p_bus': torch.randn(2, 10, 33),
            'q_bus': torch.randn(2, 10, 33),
        }
        edge_index = torch.randint(0, 33, (2, 64))
        obs_mask = torch.ones(2, 10, 33, dtype=torch.bool)

        states, parameters, physics_loss = model(
            measurements, edge_index, obs_mask=obs_mask
        )

        assert 'v_mag' in states
        assert 'r_line' in parameters
        assert physics_loss >= 0

    def test_backward_pass(self):
        """Test gradient flow"""
        model = GraphMamba(num_nodes=33, num_edges=64)

        measurements = {
            'v_mag': torch.randn(2, 5, 33, requires_grad=True),
            'p_bus': torch.randn(2, 5, 33),
            'q_bus': torch.randn(2, 5, 33),
        }
        edge_index = torch.randint(0, 33, (2, 64))

        states, parameters = model(measurements, edge_index)
        loss = states['v_mag'].sum()
        loss.backward()

        # Check gradients exist
        assert measurements['v_mag'].grad is not None
        assert torch.isfinite(measurements['v_mag'].grad).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
