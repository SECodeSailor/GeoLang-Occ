import torch
import torch.nn as nn
from torch.nn import functional as F
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from spconv import pytorch as spconv
from .utils import safe_inverse_sigmoid, cartesian

@MODELS.register_module()
class OmniSpatialShift(BaseModule):
    def __init__(self, input_dim, conv_embed_channels, kernel_size, dilation,
                 pc_range=None, grid_size=None, ratio=8):
        super(OmniSpatialShift, self).__init__()

        self.branches = nn.ModuleList([
            spconv.SubMConv3d(input_dim, conv_embed_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            spconv.SubMConv3d(input_dim, conv_embed_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            spconv.SubMConv3d(input_dim, conv_embed_channels, kernel_size=3, padding=4, dilation=4, bias=False)
        ])

        num_branch = len(self.branches)
        total_channels = conv_embed_channels * num_branch

        self.squeeze = nn.AdaptiveAvgPool1d(1)
        
        reduced_channels = max(1, total_channels // ratio)
        self.excitation = nn.Sequential(
            nn.Conv1d(total_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_channels, total_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.output_proj = nn.Linear(total_channels, conv_embed_channels)

        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float))
        self.register_buffer('grid_size', torch.tensor(grid_size, dtype=torch.float))

    def forward(self, x, xyz, pc_range=None, grid_size=None):
        bs, n, _ = x.shape
        anchor_xyz = cartesian(xyz, pc_range=self.pc_range).flatten(0, 1)
        indices = anchor_xyz - self.pc_range[None, :3]
        indices = indices / self.grid_size[None, :]
        indices = indices.to(torch.int32)
        batched_indices = torch.cat([
            torch.arange(bs, device=indices.device, dtype=torch.int32).reshape(
                bs, 1, 1).expand(-1, n, -1).flatten(0, 1), indices], dim=-1)
        spatial_shape = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size
        spatial_shape = spatial_shape.to(torch.int32)
        input = spconv.SparseConvTensor(
            x.flatten(0, 1),
            indices=batched_indices,
            spatial_shape=spatial_shape,
            batch_size=bs
        )

        branch_outputs = [branch(input).features for branch in self.branches]
        concat_features = torch.cat(branch_outputs, dim=1)
        squeezed_features = concat_features.transpose(0,1).unsqueeze(0)

        channel_descriptor = self.squeeze(squeezed_features)
        channel_weight = self.excitation(channel_descriptor)
        scaled_features = concat_features * channel_weight.squeeze(0).transpose(0,1)

        fianl_out = self.output_proj(scaled_features)

        return fianl_out.unflatten(0, (bs, n))
    
class GateFusion(nn.Module):
    def __init__(self, detial_channels, context_channels, fianl_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.proj_detial = spconv.SubMConv3d(
            detial_channels, fianl_channels, kernel_size=1, bias=False
        )

        if context_channels != fianl_channels:
            self.proj_context = spconv.SubMConv3d(
                context_channels, fianl_channels, kernel_size=1, bias=False
            )
        else:
            self.proj_context = nn.Identity()

        self.gate_conv = spconv.SparseSequential(
                spconv.SubMConv3d(detial_channels + context_channels, fianl_channels // 2, kernel_size=1, bias=False),
                nn.LayerNorm(fianl_channels // 2),
                nn.ReLU(True),
                spconv.SubMConv3d(fianl_channels // 2, 1, kernel_size=1, bias=False),
            )
        
    def forward(self, x_detial, x_context):

        concat_feats = torch.cat([x_detial.features, x_context.features], dim=1)
        x_concat = x_detial.replace_feature(concat_feats)

        gate_feats = self.gate_conv(x_concat).features
        gate = torch.sigmoid(gate_feats)

        detail_aligned = self.proj_detial(x_detial).features
        context_aligned = self.proj_context(x_context).features

        fused_feats = (1 - gate) * context_aligned + gate * detail_aligned

        return x_detial.replace_feature(fused_feats)


@MODELS.register_module()
class Sparse3DBlock_SE(BaseModule):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size=None,
                 se_reduction_ratio=16,
                 pc_range=None, grid_size=None,
                 init_cfg = None):
        super().__init__(init_cfg)

        if isinstance(kernel_size, list):
            self.kernel_size = kernel_size
        else:
            self.kernel_size = [kernel_size, kernel_size]

        self.conv1 = Separable3DConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size[0]
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

        self.conv2 = Separable3DConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size[1]
        )
        self.norm2 = nn.BatchNorm1d(out_channels)
        # self.act2 = nn.ReLU(inplace=True)


        self.funsion_conv = spconv.SubMConv3d(
            out_channels*2,
            out_channels,
            kernel_size=1
        )
        self.norm_funsion = nn.BatchNorm1d(out_channels)

        self.se_block = SparseSEBlock(
            out_channels,
            reduction_ratio=se_reduction_ratio
        )

        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float))
        self.register_buffer('grid_size', torch.tensor(grid_size, dtype=torch.float))

    def forward(self, x, xyz, pc_range=None, grid_size=None):
        bs, n, _ = x.shape
        anchor_xyz = cartesian(xyz, pc_range=self.pc_range).flatten(0, 1)
        indices = anchor_xyz - self.pc_range[None, :3]
        indices = indices / self.grid_size[None, :]
        indices = indices.to(torch.int32)
        batched_indices = torch.cat([
            torch.arange(bs, device=indices.device, dtype=torch.int32).reshape(
                bs, 1, 1).expand(-1, n, -1).flatten(0, 1), indices], dim=-1)
        spatial_shape = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size
        spatial_shape = spatial_shape.to(torch.int32)
        input = spconv.SparseConvTensor(
            x.flatten(0, 1),
            indices=batched_indices,
            spatial_shape=spatial_shape,
            batch_size=bs
        )

        # identity = input

        out_3x3 = self.conv1(input)
        out_3x3 = out_3x3.replace_feature(self.norm1(out_3x3.features))
        # out_3x3 = out_3x3.replace_feature(self.act1(out_3x3.features))


        out_5x5 = self.conv2(input)
        out_5x5 = out_5x5.replace_feature(self.norm2(out_5x5.features))

        concat_features = torch.cat([out_3x3.features, out_5x5.features], dim=1)
        fused_out = input.replace_feature(concat_features)

        fused_out = self.funsion_conv(fused_out)
        fused_out = fused_out.replace_feature(self.norm_funsion(fused_out.features))
        fused_out = fused_out.replace_feature(self.act(fused_out.features))

        se_out = self.se_block(fused_out)

        # final_out = se_out.replace_feature(se_out.features + identity.features)
        # final_out = final_out.replace_feature(self.act(final_out.features))

        final_out = se_out.features.unflatten(0, (bs, n))

        return final_out



class Separable3DConv(BaseModule):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 indice_key=None,
                 bias=False,
                 init_cfg = None):
        super().__init__(init_cfg)

        assert kernel_size % 2 == 1
        self.in_channels = in_channels
        self.out_channels = out_channels

        # self.depthwise = spconv.SubMConv3d(
        #     in_channels,
        #     out_channels,
        #     kernel_size=kernel_size,
        #     padding=(kernel_size - 1) // 2,
        #     groups=in_channels,
        #     bias=bias
        # )

        self.depthwise = nn.ModuleList([
            spconv.SubMConv3d(
                1,
                1,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=bias
            ) for _ in range(in_channels)
        ])

        self.pointwise = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        # x = self.depthwise(x)

        depthwise_outputs = []

        for c in range(self.in_channels):
            cur_channel = x.features[:, c:c+1]
            cur_sparse = x.replace_feature(cur_channel)
            output_channel = self.depthwise[c](cur_sparse)

            depthwise_outputs.append(output_channel.features)

        concat_features = torch.cat(depthwise_outputs, dim=1)
        x_depthwise_out = x.replace_feature(concat_features)


        x_pointwise_out = self.pointwise(x_depthwise_out)

        return x_pointwise_out
    

class SparseSEBlock(BaseModule):
    def __init__(self, 
                 channels,
                 reduction_ratio=16,
                 init_cfg = None):
        super().__init__(init_cfg)

        reduced_channels = max(1, channels // reduction_ratio)

        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = x.features

        # Squeeze
        squeezed_feats = features.transpose(0, 1).unsqueeze(0)
        channel_descriptor = self.squeeze(squeezed_feats)
        channel_descriptors = channel_descriptor.view(channel_descriptor.shape[0], channel_descriptor.shape[1])

        # excitation
        channel_weight = self.excitation(channel_descriptors)

        # scale
        scaled_feats = features * channel_weight

        return x.replace_feature(scaled_feats)
    


if __name__ == '__main__':
    pc_range = torch.tensor([-50.0, -50.0, -5.0, 50.0, 50.0, 3.0])
    grid_size = torch.tensor([0.5, 0.5, 0.5])
    model = Spconv3DShift(128, 128, [1, 3, 5], [1, 3, 5])
    xyz = safe_inverse_sigmoid(torch.randn(1, 10, 3))
    feats = torch.rand(1, 10, 128)
    anchor_xyz = cartesian(xyz, pc_range=pc_range).flatten(0, 1)
    indices = anchor_xyz - pc_range[None, :3]
    indices = indices / grid_size[None, :]
    indices = indices.to(torch.int32)
    batched_indices = torch.cat([
        torch.arange(1, device=indices.device, dtype=torch.int32).reshape(
            1, 1, 1).expand(-1, 10, -1).flatten(0, 1), indices], dim=-1)
    spatial_shape = (pc_range[3:] - pc_range[:3]) / grid_size
    spatial_shape = spatial_shape.to(torch.int32)
    input = spconv.SparseConvTensor(
        feats.flatten(0, 1),
        indices=batched_indices,
        spatial_shape=spatial_shape,
        batch_size=1
    )
    out = model(input)
    print(out)