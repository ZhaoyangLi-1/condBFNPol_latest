# Policies

Visuomotor policy implementations.

<br>

## BFN

| Policy | Description |
|:--|:--|
| `bfn_policy.py` | Base BFN |
| `bfn_unet_hybrid_image_policy.py` | Image-conditioned |
| `bfn_hybrid_action_policy.py` | Hybrid discrete/continuous |

<br>

## Diffusion

| Policy | Description |
|:--|:--|
| `diffusion_policy.py` | Base diffusion |
| `diffusion_unet_hybrid_image_policy.py` | Image-conditioned |

<br>

## Interface

```python
policy.predict_action(obs)    # Inference
policy.compute_loss(batch)    # Training
policy.set_normalizer(norm)   # Setup
```
