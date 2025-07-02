#!/usr/bin/env python3
"""
Integration script for multimodal motion fusion learning system.
This script integrates the multimodal system with the existing PBHC training pipeline.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import yaml
import subprocess
from typing import Dict, List, Optional

# Add PBHC to Python path
PBHC_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PBHC_ROOT))

class MultimodalIntegrator:
    """Integrates multimodal motion fusion system with PBHC."""
    
    def __init__(self):
        self.pbhc_root = PBHC_ROOT
        self.multimodal_dir = self.pbhc_root / "innovation_multimodal"
        self.config_dir = self.multimodal_dir / "config"
        self.humanoidverse_config = self.pbhc_root / "humanoidverse" / "config"
        
    def setup_environment(self):
        """Set up the environment for multimodal training."""
        print("🔧 Setting up multimodal environment...")
        
        # Create necessary directories
        dirs_to_create = [
            self.multimodal_dir / "pretrained",
            self.multimodal_dir / "logs",
            self.multimodal_dir / "evaluation",
            self.multimodal_dir / "checkpoints"
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created directory: {dir_path}")
    
    def install_dependencies(self):
        """Install additional dependencies for multimodal learning."""
        print("📦 Installing multimodal dependencies...")
        
        requirements = [
            "scikit-learn>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "tensorboard>=2.8.0",
            "plotly>=5.0.0",
            "scipy>=1.7.0"
        ]
        
        for req in requirements:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", req], 
                             check=True, capture_output=True)
                print(f"  ✓ Installed: {req}")
            except subprocess.CalledProcessError as e:
                print(f"  ⚠️  Warning: Failed to install {req}: {e}")
    
    def create_symbolic_links(self):
        """Create symbolic links to integrate with PBHC config system."""
        print("🔗 Creating configuration links...")
        
        # Link multimodal configs to humanoidverse config
        links_to_create = [
            (self.config_dir / "env" / "multimodal_motion_tracking.yaml",
             self.humanoidverse_config / "env" / "multimodal_motion_tracking.yaml"),
            (self.config_dir / "algo" / "multimodal_ppo.yaml",
             self.humanoidverse_config / "algo" / "multimodal_ppo.yaml"),
            (self.config_dir / "multimodal_base.yaml",
             self.humanoidverse_config / "multimodal_base.yaml")
        ]
        
        for src, dst in links_to_create:
            if dst.exists():
                dst.unlink()
            try:
                dst.symlink_to(src.resolve())
                print(f"  ✓ Linked: {src.name} -> {dst}")
            except OSError as e:
                # Fallback to copy if symlink fails
                shutil.copy2(src, dst)
                print(f"  ✓ Copied: {src.name} -> {dst}")
    
    def validate_integration(self) -> bool:
        """Validate that the integration is working correctly."""
        print("🔍 Validating integration...")
        
        validation_checks = [
            ("Multimodal environment", self.validate_multimodal_env),
            ("Motion encoder", self.validate_motion_encoder),
            ("Fusion controller", self.validate_fusion_controller),
            ("PPO algorithm", self.validate_multimodal_ppo),
            ("Configuration files", self.validate_configs)
        ]
        
        all_passed = True
        for check_name, check_func in validation_checks:
            try:
                result = check_func()
                status = "✓" if result else "✗"
                print(f"  {status} {check_name}")
                if not result:
                    all_passed = False
            except Exception as e:
                print(f"  ✗ {check_name}: {e}")
                all_passed = False
        
        return all_passed
    
    def validate_multimodal_env(self) -> bool:
        """Validate multimodal environment can be imported."""
        try:
            from innovation_multimodal.multimodal_env import MultimodalMotionTrackingEnv
            return True
        except ImportError:
            return False
    
    def validate_motion_encoder(self) -> bool:
        """Validate motion encoder can be imported and initialized."""
        try:
            from innovation_multimodal.motion_encoder import MotionEncoder
            encoder = MotionEncoder(input_dim=100, latent_dim=128, num_motion_types=6)
            return True
        except (ImportError, Exception):
            return False
    
    def validate_fusion_controller(self) -> bool:
        """Validate fusion controller can be imported."""
        try:
            from innovation_multimodal.fusion_controller import FusionController
            return True
        except ImportError:
            return False
    
    def validate_multimodal_ppo(self) -> bool:
        """Validate multimodal PPO can be imported."""
        try:
            from innovation_multimodal.multimodal_ppo import MultimodalPPO
            return True
        except ImportError:
            return False
    
    def validate_configs(self) -> bool:
        """Validate configuration files are valid YAML."""
        config_files = [
            self.config_dir / "multimodal_base.yaml",
            self.config_dir / "env" / "multimodal_motion_tracking.yaml",
            self.config_dir / "algo" / "multimodal_ppo.yaml"
        ]
        
        for config_file in config_files:
            if not config_file.exists():
                return False
            try:
                with open(config_file, 'r') as f:
                    yaml.safe_load(f)
            except yaml.YAMLError:
                return False
        
        return True
    
    def create_training_script(self):
        """Create a training script for multimodal learning."""
        script_content = '''#!/usr/bin/env python3
"""
Multimodal motion fusion training script.
"""

import hydra
from omegaconf import DictConfig
import sys
from pathlib import Path

# Add PBHC to path
PBHC_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PBHC_ROOT))

from humanoidverse.train_agent import train_agent

@hydra.main(version_base=None, config_path="config", config_name="multimodal_base")
def main(cfg: DictConfig) -> None:
    """Main training function for multimodal motion fusion."""
    print("🚀 Starting multimodal motion fusion training...")
    print(f"📝 Experiment: {cfg.experiment_name}")
    print(f"🎯 Motion types: {cfg.multimodal.motion_types}")
    
    # Run training
    train_agent(cfg)

if __name__ == "__main__":
    main()
'''
        
        script_path = self.multimodal_dir / "train_multimodal.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)
        print(f"  ✓ Created training script: {script_path}")
    
    def create_evaluation_script(self):
        """Create an evaluation script for multimodal models."""
        script_content = '''#!/usr/bin/env python3
"""
Multimodal motion fusion evaluation script.
"""

import hydra
from omegaconf import DictConfig
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

# Add PBHC to path
PBHC_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PBHC_ROOT))

from humanoidverse.eval_agent import eval_agent
from innovation_multimodal.multimodal_env import MultimodalMotionTrackingEnv
from innovation_multimodal.fusion_controller import FusionMetrics

class MultimodalEvaluator:
    """Evaluates multimodal motion fusion models."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.metrics = FusionMetrics()
        self.results = {}
    
    def evaluate_fusion_quality(self, model_path: str) -> Dict:
        """Evaluate fusion quality across different motion combinations."""
        print("🔍 Evaluating fusion quality...")
        
        # Load model and run evaluation
        results = eval_agent(self.cfg, model_path)
        
        # Analyze fusion-specific metrics
        fusion_results = {
            'motion_similarity': self.metrics.calculate_motion_similarity(results),
            'transition_smoothness': self.metrics.calculate_transition_smoothness(results),
            'innovation_score': self.metrics.calculate_innovation_score(results)
        }
        
        return fusion_results
    
    def create_visualization(self, results: Dict):
        """Create visualizations of evaluation results."""
        print("📊 Creating evaluation visualizations...")
        
        # Implementation would create plots for:
        # - Fusion quality heatmap
        # - Motion compatibility matrix
        # - Transition smoothness over time
        # - Innovation score distribution
        
        pass

@hydra.main(version_base=None, config_path="config", config_name="multimodal_base")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    evaluator = MultimodalEvaluator(cfg)
    
    # Evaluate model
    model_path = cfg.get("model_path", "checkpoints/latest.pt")
    results = evaluator.evaluate_fusion_quality(model_path)
    
    # Create visualizations
    evaluator.create_visualization(results)
    
    print("✅ Evaluation complete!")

if __name__ == "__main__":
    main()
'''
        
        script_path = self.multimodal_dir / "eval_multimodal.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)
        print(f"  ✓ Created evaluation script: {script_path}")

def main():
    """Main integration function."""
    parser = argparse.ArgumentParser(description="Integrate multimodal motion fusion system with PBHC")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation")
    parser.add_argument("--no-deps", action="store_true", help="Skip dependency installation")
    args = parser.parse_args()
    
    integrator = MultimodalIntegrator()
    
    if args.validate_only:
        print("🔍 Running validation only...")
        success = integrator.validate_integration()
        sys.exit(0 if success else 1)
    
    print("🎯 Integrating multimodal motion fusion system with PBHC...")
    
    # Run integration steps
    integrator.setup_environment()
    
    if not args.no_deps:
        integrator.install_dependencies()
    
    integrator.create_symbolic_links()
    integrator.create_training_script()
    integrator.create_evaluation_script()
    
    # Validate integration
    if integrator.validate_integration():
        print("\n✅ Integration successful!")
        print("\n🚀 Next steps:")
        print("1. Run training: cd innovation_multimodal && python train_multimodal.py")
        print("2. Monitor progress: tensorboard --logdir logs")
        print("3. Evaluate results: python eval_multimodal.py")
    else:
        print("\n❌ Integration failed. Please check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
