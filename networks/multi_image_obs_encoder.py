"""Multi-Image Observation Encoder and Crop Randomizer.

This module provides a robust, self-contained implementation of a multi-image
encoder that supports shared or independent backbones, resizing, random cropping
(for data augmentation), and normalization.

It removes dependencies on `ModuleAttrMixin` and `tensor_util`, using standard
PyTorch idioms instead.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as ttf
from einops import rearrange

# Attempt to import replace_submodules, fallback to simple identity if missing
try:
    from diffusion_policy.common.pytorch_util import replace_submodules
except ImportError:

    def replace_submodules(root_module, predicate, func):
        return root_module


logger = logging.getLogger(__name__)

__all__ = ["CropRandomizer", "MultiImageObsEncoder"]


# =============================================================================
# Helper Functions
# =============================================================================


def _crop_image_from_indices(
    images: torch.Tensor, crop_indices: torch.Tensor, crop_height: int, crop_width: int
) -> torch.Tensor:
    """Crops images at specified indices using advanced indexing."""
    assert crop_indices.shape[-1] == 2
    device = images.device
    img_c, img_h, img_w = images.shape[-3:]

    # Grid of offsets [CH, CW]
    grid_h, grid_w = torch.meshgrid(
        torch.arange(crop_height, device=device),
        torch.arange(crop_width, device=device),
        indexing="ij",
    )
    # [CH, CW, 2]
    grid = torch.stack([grid_h, grid_w], dim=-1)

    # Expand to match batch
    indices_expanded = crop_indices.unsqueeze(-2).unsqueeze(-2)

    # Add dimensions for Batch... and N to grid
    # Fix: Properly handle dimensions for reshaping grid
    batch_ndim = crop_indices.ndim - 2  # ... (batch dims)

    view_shape = [1] * (batch_ndim + 1) + [crop_height, crop_width, 2]
    grid_expanded = grid.view(*view_shape)

    # Absolute coordinates
    pixel_coords = indices_expanded + grid_expanded

    # Flatten for gather
    flat_pixel_indices = pixel_coords[..., 0] * img_w + pixel_coords[..., 1]
    flat_pixel_indices = flat_pixel_indices.view(*flat_pixel_indices.shape[:-2], -1)

    # Expand for channels: (..., N, CH*CW) -> (..., N, C, CH*CW)
    flat_pixel_indices = flat_pixel_indices.unsqueeze(-2)
    target_shape = list(flat_pixel_indices.shape)
    target_shape[-2] = img_c
    flat_pixel_indices = flat_pixel_indices.expand(*target_shape)

    # Prepare images: [B..., C, HW]
    images_flat = images.view(*images.shape[:-2], -1)
    num_crops = crop_indices.shape[-2]

    # Expand images: [B..., C, HW] -> [B..., N, C, HW]
    # FIX: We insert N at dimension -3.
    # We must expand using *batch_dims, N, C, HW
    images_flat = images_flat.unsqueeze(-3)

    # Correct expansion shape logic:
    # We want to match images_flat which is [B..., 1, C, HW]
    # to [B..., N, C, HW]
    expand_shape = list(images_flat.shape)
    expand_shape[-3] = num_crops

    images_flat = images_flat.expand(*expand_shape)

    crops_flat = torch.gather(images_flat, -1, flat_pixel_indices)
    crops = crops_flat.view(*crops_flat.shape[:-1], crop_height, crop_width)

    return crops


def _sample_random_image_crops(
    images: torch.Tensor,
    crop_height: int,
    crop_width: int,
    num_crops: int,
    pos_enc: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Samples random crops from images."""
    device = images.device
    source_im = images

    if pos_enc:
        h, w = source_im.shape[-2:]
        pos_y, pos_x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        pos_y = pos_y.float() / h
        pos_x = pos_x.float() / w
        pos_enc_map = torch.stack((pos_y, pos_x), dim=0)

        batch_dims = source_im.shape[:-3]
        # Robust expansion logic
        if len(batch_dims) > 0:
            pos_enc_map = pos_enc_map.view(*([1] * len(batch_dims)), 2, h, w)
            pos_enc_map = pos_enc_map.expand(*batch_dims, 2, h, w)

        source_im = torch.cat((source_im, pos_enc_map), dim=-3)

    img_h, img_w = source_im.shape[-2:]
    max_y = img_h - crop_height
    max_x = img_w - crop_width
    batch_dims = source_im.shape[:-3]

    rand_y = (torch.rand(*batch_dims, num_crops, device=device) * max_y).long()
    rand_x = (torch.rand(*batch_dims, num_crops, device=device) * max_x).long()

    crop_indices = torch.stack((rand_y, rand_x), dim=-1)
    crops = _crop_image_from_indices(source_im, crop_indices, crop_height, crop_width)

    return crops, crop_indices


# =============================================================================
# Crop Randomizer
# =============================================================================


class CropRandomizer(nn.Module):
    """Randomly samples crops from inputs for augmentation."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        crop_height: int,
        crop_width: int,
        num_crops: int = 1,
        pos_enc: bool = False,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops
        self.pos_enc = pos_enc

    def output_shape_in(self, input_shape=None) -> List[int]:
        """Returns shape after cropping, before pooling."""
        c = self.input_shape[0] + 2 if self.pos_enc else self.input_shape[0]
        return [c, self.crop_height, self.crop_width]

    def output_shape_out(self, input_shape=None) -> List[int]:
        """Returns shape after pooling (same as input structure)."""
        return list(input_shape)

    def forward_in(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training:
            crops, _ = _sample_random_image_crops(
                images=inputs,
                crop_height=self.crop_height,
                crop_width=self.crop_width,
                num_crops=self.num_crops,
                pos_enc=self.pos_enc,
            )
            return crops.flatten(0, 1)
        else:
            out = ttf.center_crop(inputs, (self.crop_height, self.crop_width))
            if self.num_crops > 1:
                b, c, h, w = out.shape
                out = (
                    out.unsqueeze(1)
                    .expand(b, self.num_crops, c, h, w)
                    .reshape(-1, c, h, w)
                )
            return out

    def forward_out(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.num_crops <= 1:
            return inputs
        b_times_n = inputs.shape[0]
        batch_size = b_times_n // self.num_crops
        out = inputs.view(batch_size, self.num_crops, *inputs.shape[1:])
        return out.mean(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_in(inputs)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"input_shape={self.input_shape}, "
            f"crop_size=[{self.crop_height}, {self.crop_width}], "
            f"num_crops={self.num_crops})"
        )


# =============================================================================
# Multi-Image Observation Encoder
# =============================================================================


class MultiImageObsEncoder(nn.Module):
    """Encodes multiple image streams and low-dim observations."""

    def __init__(
        self,
        shape_meta: Dict[str, Any],
        rgb_model: Union[nn.Module, Dict[str, nn.Module]],
        resize_shape: Optional[
            Union[Tuple[int, int], Dict[str, Tuple[int, int]]]
        ] = None,
        crop_shape: Optional[Union[Tuple[int, int], Dict[str, Tuple[int, int]]]] = None,
        random_crop: bool = True,
        use_group_norm: bool = False,
        share_rgb_model: bool = False,
        imagenet_norm: bool = False,
        output_dim: Optional[int] = None,
        n_obs_steps: int = 1,
        feature_aggregation: Optional[str] = None,
        downsample_ratio: int = 32,
    ):
        super().__init__()

        self.shape_meta = shape_meta
        self.share_rgb_model = share_rgb_model
        self.output_dim = output_dim

        rgb_keys = []
        low_dim_keys = []
        self.key_model_map = nn.ModuleDict()
        self.key_transform_map = nn.ModuleDict()

        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            if use_group_norm:
                rgb_model = replace_submodules(
                    root_module=rgb_model,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(
                        num_groups=max(1, x.num_features // 16),
                        num_channels=x.num_features,
                    ),
                )
            self.key_model_map["rgb"] = rgb_model

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            type_ = attr.get("type", "low_dim")

            if type_ == "rgb":
                rgb_keys.append(key)
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        this_model = rgb_model[key]
                    else:
                        this_model = copy.deepcopy(rgb_model)

                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=max(1, x.num_features // 16),
                                num_channels=x.num_features,
                            ),
                        )
                    self.key_model_map[key] = this_model

                transforms_list = []
                input_shape = shape
                if resize_shape is not None:
                    h, w = (
                        resize_shape[key]
                        if isinstance(resize_shape, dict)
                        else resize_shape
                    )
                    transforms_list.append(
                        torchvision.transforms.Resize((h, w), antialias=True)
                    )
                    input_shape = (shape[0], h, w)

                if crop_shape is not None:
                    h, w = (
                        crop_shape[key] if isinstance(crop_shape, dict) else crop_shape
                    )
                    if random_crop:
                        transforms_list.append(CropRandomizer(input_shape, h, w))
                    else:
                        transforms_list.append(
                            torchvision.transforms.CenterCrop((h, w))
                        )

                if imagenet_norm:
                    transforms_list.append(
                        torchvision.transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        )
                    )

                self.key_transform_map[key] = nn.Sequential(*transforms_list)

            elif type_ == "low_dim":
                low_dim_keys.append(key)

        self.rgb_keys = sorted(rgb_keys)
        self.low_dim_keys = sorted(low_dim_keys)

        if self.output_dim is not None:
            self.projector = nn.LazyLinear(self.output_dim)
        else:
            self.projector = nn.Identity()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = []
        batch_size = None

        if self.share_rgb_model:
            imgs = []
            for key in self.rgb_keys:
                img = obs_dict[key]
                if img.ndim == 5:
                    b, t = img.shape[:2]
                    img = img.view(b * t, *img.shape[2:])

                if batch_size is None:
                    batch_size = img.shape[0]

                img = self.key_transform_map[key](img)
                imgs.append(img)

            if len(imgs) > 0:
                imgs_cat = torch.cat(imgs, dim=0)
                feats = self.key_model_map["rgb"](imgs_cat)
                feats = feats.view(len(imgs), batch_size, -1)
                feats = feats.permute(1, 0, 2)
                feats = feats.reshape(batch_size, -1)
                features.append(feats)
        else:
            for key in self.rgb_keys:
                img = obs_dict[key]
                if img.ndim == 5:
                    b, t = img.shape[:2]
                    img = img.view(b * t, *img.shape[2:])
                if batch_size is None:
                    batch_size = img.shape[0]

                transform = self.key_transform_map[key]
                img = transform(img)
                feat = self.key_model_map[key](img)

                for m in transform:
                    if isinstance(m, CropRandomizer):
                        feat = m.forward_out(feat)
                features.append(feat)

        for key in self.low_dim_keys:
            data = obs_dict[key]
            if data.ndim == 3:
                b, t = data.shape[:2]
                data = data.view(b * t, -1)
            if batch_size is None:
                batch_size = data.shape[0]
            features.append(data)

        result = torch.cat(features, dim=-1)
        result = self.projector(result)

        return result
