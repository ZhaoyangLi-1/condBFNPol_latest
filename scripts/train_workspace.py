"""
Unified training script for all policy workspaces.

Usage:
    # Train diffusion policy
    python scripts/train_workspace.py --config-name=train_diffusion_unet_hybrid
    
    # Train BFN policy
    python scripts/train_workspace.py --config-name=train_bfn
    
    # Train Conditional BFN policy
    python scripts/train_workspace.py --config-name=train_conditional_bfn
    
    # Train Guided BFN policy
    python scripts/train_workspace.py --config-name=train_guided_bfn
    
    # Override parameters
    python scripts/train_workspace.py --config-name=train_bfn training.num_epochs=50 dataloader.batch_size=128
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import hydra
from omegaconf import OmegaConf, DictConfig

# Register custom resolver for eval expressions (used in diffusion_policy configs)
OmegaConf.register_new_resolver("eval", eval, replace=True)


def _patch_async_vector_env_shared_memory_api_mismatch() -> None:
    """
    Patch diffusion_policy's AsyncVectorEnv to work with newer Gym versions.

    diffusion_policy.gym_util.async_vector_env imports Gym shared-memory helpers
    but uses an older argument order for read/write. Newer Gym expects:
    - read_from_shared_memory(space, shared_memory, n=...)
    - write_to_shared_memory(space, index, value, shared_memory)

    We cannot modify diffusion_policy itself, so we monkey-patch the module-level
    references used by AsyncVectorEnv to accept both orders while keeping
    shared_memory=True.
    """
    try:
        from gym.spaces import Space
        from gym.vector.utils import shared_memory as gym_shared_memory
        import diffusion_policy.gym_util.async_vector_env as dp_async_vector_env
    except Exception as exc:  # pragma: no cover - best-effort runtime patch
        print(f"[warn] AsyncVectorEnv shared-memory patch not applied: {exc}")
        return

    if getattr(dp_async_vector_env, "_shared_memory_api_patched", False):
        return

    def _patched_read(space_or_mem, mem_or_space, *args, **kwargs):
        # Handle the reversed argument order used by diffusion_policy.
        if (not isinstance(space_or_mem, Space)) and isinstance(mem_or_space, Space):
            space_or_mem, mem_or_space = mem_or_space, space_or_mem
        return gym_shared_memory.read_from_shared_memory(
            space_or_mem, mem_or_space, *args, **kwargs
        )

    def _patched_write(space_or_index, index_or_value, value_or_mem, mem_or_space, *args, **kwargs):
        # diffusion_policy calls write_to_shared_memory(index, value, mem, space).
        if isinstance(mem_or_space, Space) and not isinstance(space_or_index, Space):
            space = mem_or_space
            index = space_or_index
            value = index_or_value
            shared_memory = value_or_mem
            return gym_shared_memory.write_to_shared_memory(
                space, index, value, shared_memory, *args, **kwargs
            )
        return gym_shared_memory.write_to_shared_memory(
            space_or_index, index_or_value, value_or_mem, mem_or_space, *args, **kwargs
        )

    dp_async_vector_env.read_from_shared_memory = _patched_read
    dp_async_vector_env.write_to_shared_memory = _patched_write

    # Gym's VectorEnv.reset now forwards seed/return_info/options kwargs.
    async_vec_cls = dp_async_vector_env.AsyncVectorEnv
    if not getattr(async_vec_cls, "_reset_api_patched", False):
        _orig_reset_async = async_vec_cls.reset_async
        _orig_reset_wait = async_vec_cls.reset_wait

        def _reset_async_with_kwargs(self, seed=None, return_info=False, options=None):
            if seed is not None:
                self.seed(seed)
            return _orig_reset_async(self)

        def _reset_wait_with_kwargs(self, timeout=None, seed=None, return_info=False, options=None):
            obs = _orig_reset_wait(self, timeout=timeout)
            if return_info:
                # Shared-memory reset doesn't return infos; provide empty dicts.
                return obs, [{} for _ in range(self.num_envs)]
            return obs

        async_vec_cls.reset_async = _reset_async_with_kwargs
        async_vec_cls.reset_wait = _reset_wait_with_kwargs
        async_vec_cls._reset_api_patched = True

    dp_async_vector_env._shared_memory_api_patched = True


@hydra.main(
    version_base=None,
    config_path='../config',
    config_name=None,
)
def main(cfg: DictConfig):
    """
    Main training function. Instantiates the workspace based on config
    and runs training.
    
    The workspace class is determined by the `_target_` field in the config.
    """
    # Ensure AsyncVectorEnv shared memory works with the installed Gym API.
    _patch_async_vector_env_shared_memory_api_mismatch()

    # Resolve all interpolations
    OmegaConf.resolve(cfg)
    
    # Print config for debugging
    print("=" * 60)
    print("Configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    # Get workspace class from config target
    workspace_cls = hydra.utils.get_class(cfg._target_)
    
    # Create and initialize workspace
    workspace = workspace_cls(cfg)
    
    # Run training
    workspace.run()


if __name__ == "__main__":
    main()
