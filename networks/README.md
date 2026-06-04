# Networks

Core neural network implementations.

<br>

## BFN

```python
from networks.conditional_bfn import ContinuousBFN

bfn = ContinuousBFN(backbone=unet, action_dim=7, sigma=0.01)
```

<br>

## Modules

| File | Description |
|:--|:--|
| `conditional_bfn.py` | BFN training & inference |
| `unet.py` | 1D U-Net backbone |
| `multi_image_obs_encoder.py` | Multi-view image encoder |
| `normalizer.py` | Action normalization |
