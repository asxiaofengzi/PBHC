#!/usr/bin/env python3
"""
Test suite for multimodal motion fusion learning system.
Validates implementation correctness and integration quality.
"""

import sys
import torch
import numpy as np
import unittest
from pathlib import Path
import tempfile
import yaml
from unittest.mock import Mock, patch

# Add PBHC to path
PBHC_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PBHC_ROOT))

try:
    from innovation_multimodal.motion_encoder import MotionEncoder, MotionCompatibilityMatrix
    from innovation_multimodal.fusion_controller import FusionController, FusionMetrics
    from innovation_multimodal.multimodal_env import MultimodalMotionTrackingEnv
    from innovation_multimodal.multimodal_ppo import MultimodalPPO, MultiExpertActor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure the multimodal modules are properly installed.")
    sys.exit(1)

class TestMotionEncoder(unittest.TestCase):
    """Test cases for motion encoder module."""
    
    def setUp(self):
        self.input_dim = 100
        self.latent_dim = 128
        self.num_motion_types = 6
        self.encoder = MotionEncoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            num_motion_types=self.num_motion_types
        )
    
    def test_encoder_initialization(self):
        """Test motion encoder initializes correctly."""
        self.assertEqual(self.encoder.latent_dim, self.latent_dim)
        self.assertEqual(self.encoder.num_motion_types, self.num_motion_types)
    
    def test_encoding_shape(self):
        """Test encoding produces correct output shape."""
        batch_size = 32
        input_data = torch.randn(batch_size, self.input_dim)
        
        encoded = self.encoder.encode(input_data)
        self.assertEqual(encoded.shape, (batch_size, self.latent_dim))
    
    def test_reconstruction(self):
        """Test reconstruction from latent space."""
        batch_size = 16
        input_data = torch.randn(batch_size, self.input_dim)
        
        # Encode and reconstruct
        encoded = self.encoder.encode(input_data)
        reconstructed = self.encoder.decode(encoded)
        
        self.assertEqual(reconstructed.shape, input_data.shape)
    
    def test_motion_classification(self):
        """Test motion type classification."""
        batch_size = 8
        input_data = torch.randn(batch_size, self.input_dim)
        
        classification = self.encoder.classify_motion_type(input_data)
        self.assertEqual(classification.shape, (batch_size, self.num_motion_types))
        
        # Check probabilities sum to 1
        prob_sums = torch.sum(classification, dim=1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6))

class TestFusionController(unittest.TestCase):
    """Test cases for fusion controller module."""
    
    def setUp(self):
        self.obs_dim = 200
        self.latent_dim = 128
        self.num_experts = 6
        self.controller = FusionController(
            obs_dim=self.obs_dim,
            latent_dim=self.latent_dim,
            num_experts=self.num_experts
        )
    
    def test_fusion_weights_generation(self):
        """Test fusion weights generation."""
        batch_size = 16
        observations = torch.randn(batch_size, self.obs_dim)
        motion_encodings = torch.randn(batch_size, self.latent_dim)
        
        weights = self.controller.generate_fusion_weights(observations, motion_encodings)
        
        self.assertEqual(weights.shape, (batch_size, self.num_experts))
        
        # Check weights are positive and sum to 1
        self.assertTrue(torch.all(weights >= 0))
        weight_sums = torch.sum(weights, dim=1)
        self.assertTrue(torch.allclose(weight_sums, torch.ones(batch_size), atol=1e-6))
    
    def test_action_fusion(self):
        """Test action fusion from multiple experts."""
        batch_size = 8
        action_dim = 20
        expert_actions = torch.randn(batch_size, self.num_experts, action_dim)
        fusion_weights = torch.softmax(torch.randn(batch_size, self.num_experts), dim=1)
        
        fused_action = self.controller.fuse_actions(expert_actions, fusion_weights)
        
        self.assertEqual(fused_action.shape, (batch_size, action_dim))
    
    def test_quality_prediction(self):
        """Test fusion quality prediction."""
        batch_size = 12
        observations = torch.randn(batch_size, self.obs_dim)
        fusion_weights = torch.softmax(torch.randn(batch_size, self.num_experts), dim=1)
        
        quality = self.controller.predict_fusion_quality(observations, fusion_weights)
        
        self.assertEqual(quality.shape, (batch_size, 1))
        self.assertTrue(torch.all(quality >= 0) and torch.all(quality <= 1))

class TestMultiExpertActor(unittest.TestCase):
    """Test cases for multi-expert actor."""
    
    def setUp(self):
        self.obs_dim = 150
        self.action_dim = 25
        self.latent_dim = 128
        self.num_experts = 6
        self.actor = MultiExpertActor(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            latent_dim=self.latent_dim,
            num_experts=self.num_experts
        )
    
    def test_expert_action_generation(self):
        """Test individual expert action generation."""
        batch_size = 10
        observations = torch.randn(batch_size, self.obs_dim)
        
        # Test each expert
        for expert_id in range(self.num_experts):
            actions = self.actor.get_expert_actions(observations, expert_id)
            self.assertEqual(actions.shape, (batch_size, self.action_dim))
    
    def test_gating_network(self):
        """Test gating network for expert selection."""
        batch_size = 16
        observations = torch.randn(batch_size, self.obs_dim)
        motion_encodings = torch.randn(batch_size, self.latent_dim)
        
        gates = self.actor.compute_gates(observations, motion_encodings)
        
        self.assertEqual(gates.shape, (batch_size, self.num_experts))
        
        # Check gates sum to 1
        gate_sums = torch.sum(gates, dim=1)
        self.assertTrue(torch.allclose(gate_sums, torch.ones(batch_size), atol=1e-6))
    
    def test_forward_pass(self):
        """Test complete forward pass."""
        batch_size = 8
        observations = torch.randn(batch_size, self.obs_dim)
        motion_encodings = torch.randn(batch_size, self.latent_dim)
        
        actions, gates, expert_actions = self.actor(observations, motion_encodings)
        
        self.assertEqual(actions.shape, (batch_size, self.action_dim))
        self.assertEqual(gates.shape, (batch_size, self.num_experts))
        self.assertEqual(expert_actions.shape, (batch_size, self.num_experts, self.action_dim))

class TestFusionMetrics(unittest.TestCase):
    """Test cases for fusion quality metrics."""
    
    def setUp(self):
        self.metrics = FusionMetrics()
    
    def test_motion_similarity_calculation(self):
        """Test motion similarity metric calculation."""
        # Mock trajectory data
        reference_traj = torch.randn(100, 50)  # 100 timesteps, 50 features
        generated_traj = reference_traj + 0.1 * torch.randn_like(reference_traj)
        
        similarity = self.metrics.calculate_motion_similarity(generated_traj, reference_traj)
        
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_transition_smoothness(self):
        """Test transition smoothness calculation."""
        # Mock motion with transitions
        motion_sequence = torch.randn(200, 30)  # 200 timesteps, 30 features
        transition_points = [50, 100, 150]
        
        smoothness = self.metrics.calculate_transition_smoothness(
            motion_sequence, transition_points
        )
        
        self.assertIsInstance(smoothness, float)
        self.assertGreaterEqual(smoothness, 0.0)
    
    def test_innovation_score(self):
        """Test innovation score calculation."""
        # Mock fusion weights and motion types
        fusion_weights = torch.softmax(torch.randn(100, 6), dim=1)
        motion_types = torch.randint(0, 6, (100,))
        
        innovation = self.metrics.calculate_innovation_score(fusion_weights, motion_types)
        
        self.assertIsInstance(innovation, float)
        self.assertGreaterEqual(innovation, 0.0)

class TestConfigurationFiles(unittest.TestCase):
    """Test configuration files are valid."""
    
    def setUp(self):
        self.config_dir = Path(__file__).parent / "config"
    
    def test_yaml_validity(self):
        """Test all YAML configuration files are valid."""
        yaml_files = list(self.config_dir.rglob("*.yaml"))
        
        self.assertGreater(len(yaml_files), 0, "No YAML files found")
        
        for yaml_file in yaml_files:
            with self.subTest(file=yaml_file.name):
                try:
                    with open(yaml_file, 'r') as f:
                        yaml.safe_load(f)
                except yaml.YAMLError as e:
                    self.fail(f"Invalid YAML in {yaml_file}: {e}")
    
    def test_required_configs_exist(self):
        """Test required configuration files exist."""
        required_configs = [
            "multimodal_base.yaml",
            "env/multimodal_motion_tracking.yaml", 
            "algo/multimodal_ppo.yaml",
            "exp/taichi_boxing_fusion.yaml",
            "exp/full_multimodal.yaml"
        ]
        
        for config in required_configs:
            config_path = self.config_dir / config
            self.assertTrue(config_path.exists(), f"Required config {config} not found")

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_end_to_end_pipeline(self):
        """Test the complete training pipeline can be initialized."""
        # This test verifies that all components can work together
        # without actually running training (which would be too slow for unit tests)
        
        # Mock configuration
        mock_config = {
            'multimodal': {
                'motion_types': ['taichi', 'boxing'],
                'fusion': {'latent_dim': 128, 'temperature': 1.0},
                'training_phases': {
                    'pretraining': {'enabled': True, 'num_iterations': 100},
                    'multimodal': {'enabled': True, 'num_iterations': 200}
                }
            },
            'env': {
                'config': {
                    'multimodal_settings': {
                        'motion_types': ['taichi', 'boxing'],
                        'latent_dim': 128
                    }
                }
            }
        }
        
        # Test component initialization with mock config
        try:
            encoder = MotionEncoder(input_dim=100, latent_dim=128, num_motion_types=2)
            controller = FusionController(obs_dim=200, latent_dim=128, num_experts=2)
            actor = MultiExpertActor(obs_dim=150, action_dim=25, latent_dim=128, num_experts=2)
            
            # Test they can work together
            batch_size = 4
            obs = torch.randn(batch_size, 100)
            obs_extended = torch.randn(batch_size, 150)
            
            # Encode motion
            encoded = encoder.encode(obs)
            
            # Generate actions
            actions, gates, expert_actions = actor(obs_extended, encoded)
            
            # Predict quality
            quality = controller.predict_fusion_quality(obs_extended[:, :200], gates)
            
            # Verify shapes
            self.assertEqual(actions.shape[0], batch_size)
            self.assertEqual(quality.shape[0], batch_size)
            
        except Exception as e:
            self.fail(f"Integration test failed: {e}")

def run_tests():
    """Run all tests and return results."""
    print("🧪 Running multimodal motion fusion test suite...")
    
    # Create test suite
    test_classes = [
        TestMotionEncoder,
        TestFusionController, 
        TestMultiExpertActor,
        TestFusionMetrics,
        TestConfigurationFiles,
        TestIntegration
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        suite.addTest(unittest.makeSuite(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Results:")
    print(f"  ✅ Tests run: {result.testsRun}")
    print(f"  ❌ Failures: {len(result.failures)}")
    print(f"  ⚠️  Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print(f"\n⚠️  Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\\n')[-2]}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print(f"\n🎉 All tests passed! System is ready for training.")
    else:
        print(f"\n💥 Some tests failed. Please fix issues before training.")
    
    return success

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
