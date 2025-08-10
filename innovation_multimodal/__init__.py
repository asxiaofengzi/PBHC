# innovation_multimodal/__init__.py
from .multimodal_ppo import MultimodalPPO
from .multimodal_env import MultimodalMotionTrackingEnv
from .motion_encoder import MotionEncoder, MotionType
from .fusion_controller import FusionController

__all__ = [
    'MultimodalPPO',
    'MultimodalMotionTrackingEnv', 
    'MotionEncoder',
    'MotionType',
    'FusionController'
]