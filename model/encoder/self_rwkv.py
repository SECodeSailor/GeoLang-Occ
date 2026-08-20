from functools import partial

from mmseg.registry import MODELS
import torch
from mmengine.model import BaseModule
from .rwkv_utils.block_v6 import Block
from .utils import cartesian, reverse_cartesian
import torch.nn as nn

@MODELS.register_module()
class Self_RWVK(BaseModule):
    supports_geometry_cache = True
    returns_residual_delta = True

    def __init__(self,
                 pc_range,
                 grid_size,
                 n_embd,
                 n_head,
                 n_layer,
                 num_layers=2,
                 shift_model='q_shift',
                 channel_gamma=1 / 4,
                 shift_pixel=1,
                 drop_path=0.,
                 hidden_rate=4,
                 conv_embed_channels=128, kernel_size=[1, 5], dilation=[1, 5],
                 **kwargs):
        super().__init__()
        self.n_embd = n_embd
        self.rwkv_block = nn.ModuleList([Block(
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            layer_id=0,
            shift_mode=shift_model,
            channel_gamma=channel_gamma,
            shift_pixel=shift_pixel,
            drop_path=drop_path,
            hidden_rate=hidden_rate,
            init_mode='fancy',
            post_norm=False,
            key_norm=False,
            init_values=None,
            with_cp=False,
            conv_embed_channels=conv_embed_channels, kernel_size=kernel_size, dilation=dilation,
            pc_range=pc_range, grid_size=grid_size,
        ) for _ in range(num_layers)])

        self.get_xyz = partial(cartesian, pc_range=pc_range)
        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float))
        self.register_buffer('grid_size', torch.tensor(grid_size, dtype=torch.float))

    def forward(
        self,
        instance_feature,
        means3d,
        metas=None,
        geometry_cache=None,
        decoder_index=0,
        **kwargs
    ):
        device = instance_feature.device
        bs, num_anchor, _ = means3d.shape

        anchor_xyz = cartesian(means3d, pc_range=self.pc_range)

        indices = anchor_xyz - self.pc_range[None, :3]
        indices = indices / self.grid_size[None, :]
        coords = indices.to(torch.int32).to(device)
        coords_batched = coords.view(bs, num_anchor, -1)

        if geometry_cache is not None:
            detached_coords = coords_batched.detach()
            geometry_cache["current_base_coords"] = detached_coords
            geometry_cache["base_grid_size"] = self.grid_size.detach()
            if "initial_base_coords" not in geometry_cache:
                geometry_cache["initial_base_coords"] = detached_coords


        spatial_shape = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size
        spatial_shape = spatial_shape.to(torch.int32)

        x = instance_feature
        residual_delta = None
        for layer in self.rwkv_block:
            layer_delta = layer.forward_delta(
                x,
                spatial_shape,
                means3d,
                self.pc_range,
                self.grid_size,
            )
            x = x + layer_delta
            residual_delta = (
                layer_delta
                if residual_delta is None
                else residual_delta + layer_delta
            )

        if residual_delta is None:
            return torch.zeros_like(instance_feature)
        return residual_delta