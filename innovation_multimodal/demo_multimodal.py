#!/usr/bin/env python3
"""
Demo script for multimodal motion fusion learning system.
Demonstrates key features and capabilities without full training.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add PBHC to path
PBHC_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PBHC_ROOT))

try:
    from innovation_multimodal.motion_encoder import MotionEncoder, MotionCompatibilityMatrix
    from innovation_multimodal.fusion_controller import FusionController, FusionMetrics
    from innovation_multimodal.multimodal_ppo import MultiExpertActor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please run the integration script first: python integrate_multimodal.py")
    sys.exit(1)

class MultimodalDemo:
    """Demonstrates multimodal motion fusion capabilities."""
    
    def __init__(self):
        self.motion_types = ["taichi", "boxing", "dance", "karate", "yoga", "gymnastics"]
        self.num_motion_types = len(self.motion_types)
        self.latent_dim = 128
        self.obs_dim = 200
        self.action_dim = 25
        
        # Initialize components
        self.setup_components()
        
    def setup_components(self):
        """Initialize all multimodal components."""
        print("🔧 Initializing multimodal components...")
        
        # Motion encoder
        self.motion_encoder = MotionEncoder(
            input_dim=self.obs_dim,
            latent_dim=self.latent_dim,
            num_motion_types=self.num_motion_types
        )
        
        # Fusion controller
        self.fusion_controller = FusionController(
            obs_dim=self.obs_dim,
            latent_dim=self.latent_dim,
            num_experts=self.num_motion_types
        )
        
        # Multi-expert actor
        self.multi_expert_actor = MultiExpertActor(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            latent_dim=self.latent_dim,
            num_experts=self.num_motion_types
        )
        
        # Compatibility matrix
        self.compatibility_matrix = MotionCompatibilityMatrix(self.num_motion_types)
        
        # Metrics calculator
        self.metrics = FusionMetrics()
        
        print("✅ Components initialized successfully!")
    
    def demo_motion_encoding(self):
        """Demonstrate motion encoding and latent space representation."""
        print("\n🎭 Demonstrating motion encoding...")
        
        # Generate sample motion data for different types
        batch_size = 32
        sample_motions = {}
        
        for i, motion_type in enumerate(self.motion_types):
            # Simulate different motion characteristics
            base_motion = torch.randn(batch_size, self.obs_dim)
            
            # Add motion-specific patterns
            if motion_type == "taichi":
                base_motion += 0.3 * torch.sin(torch.linspace(0, 4*np.pi, self.obs_dim))
            elif motion_type == "boxing":
                base_motion += 0.5 * torch.sign(torch.randn(self.obs_dim))
            elif motion_type == "dance":
                base_motion += 0.4 * torch.cos(torch.linspace(0, 6*np.pi, self.obs_dim))
            
            sample_motions[motion_type] = base_motion
        
        # Encode all motions
        encoded_motions = {}
        for motion_type, motion_data in sample_motions.items():
            encoded = self.motion_encoder.encode(motion_data)
            encoded_motions[motion_type] = encoded
            print(f"  ✓ Encoded {motion_type}: {motion_data.shape} -> {encoded.shape}")
        
        # Visualize latent space (2D projection)
        self.visualize_latent_space(encoded_motions)
        
        return encoded_motions
    
    def demo_fusion_control(self):
        """Demonstrate fusion weight generation and control."""
        print("\n🎛️ Demonstrating fusion control...")
        
        batch_size = 16
        observations = torch.randn(batch_size, self.obs_dim)
        motion_encodings = torch.randn(batch_size, self.latent_dim)
        
        # Generate fusion weights
        fusion_weights = self.fusion_controller.generate_fusion_weights(
            observations, motion_encodings
        )
        
        print(f"  ✓ Generated fusion weights: {fusion_weights.shape}")
        print(f"  📊 Weight distribution (mean per expert):")
        mean_weights = torch.mean(fusion_weights, dim=0)
        for i, motion_type in enumerate(self.motion_types):
            print(f"    {motion_type:12}: {mean_weights[i]:.3f}")
        
        # Predict fusion quality
        quality = self.fusion_controller.predict_fusion_quality(
            observations, fusion_weights
        )
        
        print(f"  🎯 Average predicted fusion quality: {torch.mean(quality):.3f}")
        
        # Visualize fusion patterns
        self.visualize_fusion_patterns(fusion_weights)
        
        return fusion_weights
    
    def demo_multi_expert_action(self):
        """Demonstrate multi-expert action generation."""
        print("\n🤖 Demonstrating multi-expert action generation...")
        
        batch_size = 8
        observations = torch.randn(batch_size, self.obs_dim)
        motion_encodings = torch.randn(batch_size, self.latent_dim)
        
        # Generate actions from all experts
        actions, gates, expert_actions = self.multi_expert_actor(
            observations, motion_encodings
        )
        
        print(f"  ✓ Generated fused actions: {actions.shape}")
        print(f"  ✓ Expert gates: {gates.shape}")
        print(f"  ✓ Individual expert actions: {expert_actions.shape}")
        
        # Analyze expert utilization
        mean_gates = torch.mean(gates, dim=0)
        print(f"  📊 Expert utilization:")
        for i, motion_type in enumerate(self.motion_types):
            print(f"    {motion_type:12}: {mean_gates[i]:.3f}")
        
        # Visualize expert contributions
        self.visualize_expert_contributions(gates, expert_actions)
        
        return actions, gates, expert_actions
    
    def demo_compatibility_learning(self):
        """Demonstrate motion compatibility matrix learning."""
        print("\n🔗 Demonstrating motion compatibility learning...")
        
        # Simulate training data for compatibility learning
        num_samples = 1000
        motion_pairs = []
        compatibility_scores = []
        
        for _ in range(num_samples):
            # Random motion pair
            motion_a = torch.randint(0, self.num_motion_types, (1,)).item()
            motion_b = torch.randint(0, self.num_motion_types, (1,)).item()
            
            # Simulate compatibility score based on motion characteristics
            if motion_a == motion_b:
                score = 1.0  # Same motion is perfectly compatible
            elif (motion_a in [0, 4] and motion_b in [0, 4]) or \
                 (motion_a in [1, 3] and motion_b in [1, 3]):
                # Taichi-Yoga or Boxing-Karate pairs are more compatible
                score = 0.8 + 0.2 * np.random.random()
            else:
                # Other pairs have variable compatibility
                score = 0.3 + 0.4 * np.random.random()
            
            motion_pairs.append([motion_a, motion_b])
            compatibility_scores.append(score)
        
        # Learn compatibility matrix
        motion_pairs = torch.tensor(motion_pairs)
        compatibility_scores = torch.tensor(compatibility_scores)
        
        self.compatibility_matrix.update_compatibility(
            motion_pairs, compatibility_scores
        )
        
        # Visualize learned compatibility matrix
        self.visualize_compatibility_matrix()
        
        print(f"  ✅ Learned compatibility from {num_samples} motion pairs")
    
    def demo_innovation_metrics(self):
        """Demonstrate innovation and quality metrics calculation."""
        print("\n🎯 Demonstrating innovation metrics...")
        
        # Generate sample trajectory data
        seq_length = 200
        feature_dim = 30
        
        # Simulate motion with transitions
        trajectory = torch.randn(seq_length, feature_dim)
        
        # Add realistic motion patterns
        t = torch.linspace(0, 4*np.pi, seq_length)
        trajectory[:, 0] += torch.sin(t)  # Rhythmic component
        trajectory[:, 1] += 0.5 * torch.cos(2*t)  # Higher frequency
        
        # Define transition points
        transition_points = [50, 100, 150]
        
        # Calculate metrics
        smoothness = self.metrics.calculate_transition_smoothness(
            trajectory, transition_points
        )
        
        # Generate fusion weights for innovation calculation
        fusion_weights = torch.softmax(torch.randn(seq_length, self.num_motion_types), dim=1)
        motion_types = torch.randint(0, self.num_motion_types, (seq_length,))
        
        innovation = self.metrics.calculate_innovation_score(
            fusion_weights, motion_types
        )
        
        print(f"  📊 Transition smoothness: {smoothness:.3f}")
        print(f"  💡 Innovation score: {innovation:.3f}")
        
        # Visualize metrics over time
        self.visualize_metrics_timeline(trajectory, fusion_weights, transition_points)
    
    def visualize_latent_space(self, encoded_motions):
        """Visualize motion encodings in 2D latent space."""
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_motion_types))
        
        for i, (motion_type, encodings) in enumerate(encoded_motions.items()):
            # Project to 2D using PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            encodings_2d = pca.fit_transform(encodings.detach().numpy())
            
            plt.scatter(
                encodings_2d[:, 0], encodings_2d[:, 1],
                c=[colors[i]], label=motion_type, alpha=0.7, s=50
            )
        
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('Motion Encodings in Latent Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('multimodal_latent_space.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("  📊 Saved latent space visualization: multimodal_latent_space.png")
    
    def visualize_fusion_patterns(self, fusion_weights):
        """Visualize fusion weight patterns."""
        plt.figure(figsize=(12, 8))
        
        # Heatmap of fusion weights
        weights_np = fusion_weights.detach().numpy()
        
        plt.subplot(2, 2, 1)
        sns.heatmap(weights_np.T, cmap='viridis', cbar=True)
        plt.xlabel('Sample Index')
        plt.ylabel('Motion Type')
        plt.title('Fusion Weights Heatmap')
        plt.yticks(range(self.num_motion_types), self.motion_types, rotation=0)
        
        # Average weights per motion type
        plt.subplot(2, 2, 2)
        mean_weights = np.mean(weights_np, axis=0)
        bars = plt.bar(self.motion_types, mean_weights, color=plt.cm.Set3(np.linspace(0, 1, self.num_motion_types)))
        plt.xlabel('Motion Type')
        plt.ylabel('Average Weight')
        plt.title('Average Fusion Weights')
        plt.xticks(rotation=45)
        
        # Weight distribution
        plt.subplot(2, 2, 3)
        for i, motion_type in enumerate(self.motion_types):
            plt.hist(weights_np[:, i], alpha=0.6, label=motion_type, bins=20)
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.title('Weight Distribution')
        plt.legend()
        
        # Motion combination patterns
        plt.subplot(2, 2, 4)
        dominant_motions = np.argmax(weights_np, axis=1)
        unique, counts = np.unique(dominant_motions, return_counts=True)
        motion_labels = [self.motion_types[i] for i in unique]
        plt.pie(counts, labels=motion_labels, autopct='%1.1f%%')
        plt.title('Dominant Motion Distribution')
        
        plt.tight_layout()
        plt.savefig('multimodal_fusion_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("  📊 Saved fusion patterns visualization: multimodal_fusion_patterns.png")
    
    def visualize_expert_contributions(self, gates, expert_actions):
        """Visualize expert contributions and action diversity."""
        plt.figure(figsize=(15, 10))
        
        gates_np = gates.detach().numpy()
        expert_actions_np = expert_actions.detach().numpy()
        
        # Expert gate patterns
        plt.subplot(2, 3, 1)
        sns.heatmap(gates_np.T, cmap='plasma', cbar=True)
        plt.xlabel('Sample Index')
        plt.ylabel('Expert')
        plt.title('Expert Gate Activations')
        plt.yticks(range(self.num_motion_types), self.motion_types, rotation=0)
        
        # Expert utilization
        plt.subplot(2, 3, 2)
        mean_gates = np.mean(gates_np, axis=0)
        bars = plt.bar(self.motion_types, mean_gates)
        plt.xlabel('Expert Type')
        plt.ylabel('Average Activation')
        plt.title('Expert Utilization')
        plt.xticks(rotation=45)
        
        # Action diversity per expert
        plt.subplot(2, 3, 3)
        action_stds = np.std(expert_actions_np, axis=0)  # Std across samples
        mean_action_stds = np.mean(action_stds, axis=1)   # Mean across action dims
        
        plt.bar(self.motion_types, mean_action_stds)
        plt.xlabel('Expert Type')
        plt.ylabel('Action Diversity (Std)')
        plt.title('Action Diversity by Expert')
        plt.xticks(rotation=45)
        
        # Action correlation between experts
        plt.subplot(2, 3, 4)
        action_means = np.mean(expert_actions_np, axis=0)  # Mean across samples
        correlation_matrix = np.corrcoef(action_means)
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.xlabel('Expert')
        plt.ylabel('Expert')
        plt.title('Inter-Expert Action Correlation')
        plt.xticks(range(self.num_motion_types), self.motion_types, rotation=45)
        plt.yticks(range(self.num_motion_types), self.motion_types, rotation=0)
        
        # Gate entropy (measure of multimodality)
        plt.subplot(2, 3, 5)
        gate_entropy = -np.sum(gates_np * np.log(gates_np + 1e-8), axis=1)
        plt.hist(gate_entropy, bins=20, alpha=0.7)
        plt.xlabel('Gate Entropy')
        plt.ylabel('Frequency')
        plt.title('Multimodality Measure')
        plt.axvline(np.mean(gate_entropy), color='red', linestyle='--', label=f'Mean: {np.mean(gate_entropy):.2f}')
        plt.legend()
        
        # Action magnitude comparison
        plt.subplot(2, 3, 6)
        action_magnitudes = np.linalg.norm(expert_actions_np, axis=2)  # L2 norm across action dims
        mean_magnitudes = np.mean(action_magnitudes, axis=0)
        
        plt.bar(self.motion_types, mean_magnitudes)
        plt.xlabel('Expert Type')
        plt.ylabel('Average Action Magnitude')
        plt.title('Action Magnitude by Expert')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('multimodal_expert_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("  📊 Saved expert analysis visualization: multimodal_expert_analysis.png")
    
    def visualize_compatibility_matrix(self):
        """Visualize learned motion compatibility matrix."""
        plt.figure(figsize=(10, 8))
        
        compatibility = self.compatibility_matrix.get_compatibility_matrix()
        compatibility_np = compatibility.detach().numpy()
        
        # Create symmetric matrix for visualization
        symmetric_matrix = (compatibility_np + compatibility_np.T) / 2
        
        sns.heatmap(
            symmetric_matrix,
            annot=True,
            cmap='RdYlGn',
            center=0.5,
            xticklabels=self.motion_types,
            yticklabels=self.motion_types,
            square=True
        )
        
        plt.xlabel('Motion Type B')
        plt.ylabel('Motion Type A')
        plt.title('Learned Motion Compatibility Matrix')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('multimodal_compatibility_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("  📊 Saved compatibility matrix: multimodal_compatibility_matrix.png")
    
    def visualize_metrics_timeline(self, trajectory, fusion_weights, transition_points):
        """Visualize metrics evolution over time."""
        plt.figure(figsize=(15, 12))
        
        seq_length = trajectory.shape[0]
        time_steps = np.arange(seq_length)
        
        # Motion trajectory
        plt.subplot(3, 2, 1)
        plt.plot(time_steps, trajectory[:, :3].detach().numpy())
        plt.xlabel('Time Step')
        plt.ylabel('Motion Features')
        plt.title('Sample Motion Trajectory')
        plt.legend(['Feature 1', 'Feature 2', 'Feature 3'])
        
        for tp in transition_points:
            plt.axvline(tp, color='red', linestyle='--', alpha=0.7)
        
        # Fusion weights over time
        plt.subplot(3, 2, 2)
        fusion_weights_np = fusion_weights.detach().numpy()
        
        for i, motion_type in enumerate(self.motion_types):
            plt.plot(time_steps, fusion_weights_np[:, i], label=motion_type, alpha=0.8)
        
        plt.xlabel('Time Step')
        plt.ylabel('Fusion Weight')
        plt.title('Fusion Weights Evolution')
        plt.legend()
        
        for tp in transition_points:
            plt.axvline(tp, color='red', linestyle='--', alpha=0.7)
        
        # Motion velocity
        plt.subplot(3, 2, 3)
        velocity = torch.diff(trajectory, dim=0)
        velocity_magnitude = torch.norm(velocity, dim=1)
        
        plt.plot(time_steps[1:], velocity_magnitude.detach().numpy())
        plt.xlabel('Time Step')
        plt.ylabel('Velocity Magnitude')
        plt.title('Motion Velocity')
        
        for tp in transition_points:
            if tp > 0:
                plt.axvline(tp, color='red', linestyle='--', alpha=0.7)
        
        # Acceleration (smoothness indicator)
        plt.subplot(3, 2, 4)
        acceleration = torch.diff(velocity, dim=0)
        acceleration_magnitude = torch.norm(acceleration, dim=1)
        
        plt.plot(time_steps[2:], acceleration_magnitude.detach().numpy())
        plt.xlabel('Time Step')
        plt.ylabel('Acceleration Magnitude')
        plt.title('Motion Acceleration (Smoothness)')
        
        for tp in transition_points:
            if tp > 1:
                plt.axvline(tp, color='red', linestyle='--', alpha=0.7)
        
        # Fusion entropy
        plt.subplot(3, 2, 5)
        fusion_entropy = -torch.sum(fusion_weights * torch.log(fusion_weights + 1e-8), dim=1)
        
        plt.plot(time_steps, fusion_entropy.detach().numpy())
        plt.xlabel('Time Step')
        plt.ylabel('Fusion Entropy')
        plt.title('Multimodality Level')
        
        for tp in transition_points:
            plt.axvline(tp, color='red', linestyle='--', alpha=0.7)
        
        # Dominant motion over time
        plt.subplot(3, 2, 6)
        dominant_motion = torch.argmax(fusion_weights, dim=1)
        
        # Create a color map for motion types
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_motion_types))
        for i in range(seq_length):
            plt.scatter(time_steps[i], dominant_motion[i], 
                       c=[colors[dominant_motion[i]]], s=20, alpha=0.7)
        
        plt.xlabel('Time Step')
        plt.ylabel('Dominant Motion Type')
        plt.title('Motion Type Evolution')
        plt.yticks(range(self.num_motion_types), self.motion_types)
        
        for tp in transition_points:
            plt.axvline(tp, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('multimodal_metrics_timeline.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("  📊 Saved metrics timeline: multimodal_metrics_timeline.png")
    
    def run_complete_demo(self):
        """Run complete demonstration of all features."""
        print("🎬 Starting complete multimodal motion fusion demo...")
        print("=" * 60)
        
        # Run all demonstrations
        encoded_motions = self.demo_motion_encoding()
        fusion_weights = self.demo_fusion_control()
        actions, gates, expert_actions = self.demo_multi_expert_action()
        self.demo_compatibility_learning()
        self.demo_innovation_metrics()
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("\n📊 Generated visualizations:")
        print("  - multimodal_latent_space.png")
        print("  - multimodal_fusion_patterns.png")
        print("  - multimodal_expert_analysis.png")
        print("  - multimodal_compatibility_matrix.png")
        print("  - multimodal_metrics_timeline.png")
        print("\n🚀 Ready for full training! Run: ./run_training.sh")

def main():
    """Main demo function."""
    print("🎭 Multimodal Motion Fusion Learning System Demo")
    print("=" * 60)
    
    # Create and run demo
    demo = MultimodalDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()
