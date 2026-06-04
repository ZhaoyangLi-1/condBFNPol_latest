"""
Training script for BFN-Policy and Diffusion Policy experiments.
Wrapper around diffusion_policy/train.py with proper config path handling.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib

# Register custom resolvers
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path="config",  # Relative to this script
    config_name="train_bfn_hybrid_lift"  # Default config
)
def main(cfg: OmegaConf):
    """Main training function."""
    # Resolve immediately so all the ${now:} resolvers use the same time
    OmegaConf.resolve(cfg)
    
    # Import workspace class
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
