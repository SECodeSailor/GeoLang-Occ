import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.registry import MODELS

from model.utils.safe_ops import safe_sigmoid


def _expand_per_scale(value, num_scales, name):
    if isinstance(value, (int, float)) or value is None:
        return [value] * num_scales
    value = list(value)
    if len(value) != num_scales:
        raise ValueError(
            f"{name} must contain {num_scales} values, got {len(value)}."
        )
    return value


@MODELS.register_module()
class CascadeLanguageGuidance(BaseModule):

    supports_topology_cache = True

    def __init__(
        self,
        embed_dim,
        text_dim=512,
        voxel_sizes=(4.0, 2.0, 1.0),
        route_ratios=(0.5, 0.25, 0.125),
        min_routed_regions=32,
        max_routed_regions=None,
        pc_range=None,
        semantic_start=11,
        semantic_dim=17,
        valid_label_start=1,
        num_language_classes=16,
        visibility_saturation=2.0,
        semantic_uncertainty_weight=1.0,
        observation_uncertainty_weight=1.0,
        semantic_change_weight=0.25,
        prototype_temperature=0.2,
        language_temperature=0.07,
        residual_scale_init=0.1,
        topology_refresh_interval=0,
        drop_out=0.0,
        semantic_channel_start=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.embed_dim = int(embed_dim)
        self.semantic_start = int(semantic_start)
        self.semantic_dim = int(semantic_dim)
        self.valid_label_start = int(valid_label_start)
        self.num_language_classes = int(num_language_classes)

        if semantic_channel_start is None:
            semantic_channel_start = valid_label_start
        self.semantic_channel_start = int(semantic_channel_start)

        self.visibility_saturation = float(visibility_saturation)
        self.semantic_uncertainty_weight = float(
            semantic_uncertainty_weight
        )
        self.observation_uncertainty_weight = float(
            observation_uncertainty_weight
        )
        self.semantic_change_weight = float(semantic_change_weight)
        self.prototype_temperature = float(prototype_temperature)
        self.language_temperature = float(language_temperature)
        self.topology_refresh_interval = int(topology_refresh_interval)
        pc_range = list(pc_range)
        self.voxel_sizes = [float(size) for size in voxel_sizes]
        self.spatial_shapes = [
            tuple(
                int(math.ceil((pc_range[axis + 3] - pc_range[axis]) / size))
                for axis in range(3)
            )
            for size in self.voxel_sizes
        ]

        num_scales = len(self.voxel_sizes)
        self.route_ratios = [
            float(v)
            for v in _expand_per_scale(
                route_ratios, num_scales, "route_ratios"
            )
        ]
        self.min_routed_regions = [
            int(v)
            for v in _expand_per_scale(
                min_routed_regions, num_scales, "min_routed_regions"
            )
        ]
        self.max_routed_regions = _expand_per_scale(
            max_routed_regions, num_scales, "max_routed_regions"
        )
        self.max_routed_regions = [
            None if v is None else int(v) for v in self.max_routed_regions
        ]

        self.register_buffer(
            "pc_range",
            torch.tensor(pc_range, dtype=torch.float32),
            persistent=False,
        )

        self.text_key_projection = nn.Linear(
            text_dim, embed_dim, bias=False
        )
        self.text_value_projection = nn.Linear(
            text_dim, embed_dim, bias=False
        )

        self.geometry_projections = nn.ModuleList()
        self.region_norms = nn.ModuleList()
        self.query_projections = nn.ModuleList()
        self.context_projections = nn.ModuleList()
        self.fusion_gates = nn.ModuleList()
        self.fusion_norms = nn.ModuleList()
        self.delta_projections = nn.ModuleList()

        for _ in self.voxel_sizes:
            self.geometry_projections.append(
                nn.Sequential(
                    nn.Linear(6, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, embed_dim),
                )
            )
            self.region_norms.append(nn.LayerNorm(embed_dim))
            self.query_projections.append(
                nn.Linear(embed_dim, embed_dim, bias=False)
            )
            self.context_projections.append(
                nn.Linear(embed_dim, embed_dim, bias=False)
            )
            self.fusion_gates.append(
                nn.Sequential(
                    nn.Linear(embed_dim * 2 + 2, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, embed_dim),
                    nn.Sigmoid(),
                )
            )
            self.fusion_norms.append(nn.LayerNorm(embed_dim))
            self.delta_projections.append(
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        self.scale_logits = nn.Parameter(torch.zeros(num_scales))
        self.dropout = nn.Dropout(drop_out)

        self.output = nn.Linear(embed_dim, embed_dim, bias=False)
        self.layer_scale = nn.Parameter(
            torch.full((embed_dim,), float(residual_scale_init))
        )

    def _semantic_state(self, anchor, decoder_index):
        batch_size, num_anchor = anchor.shape[:2]
        if decoder_index == 0:
            probabilities = anchor.new_full(
                (
                    batch_size,
                    num_anchor,
                    self.num_language_classes,
                ),
                1.0 / self.num_language_classes,
            )
            uncertainty = anchor.new_ones(
                (batch_size, num_anchor, 1)
            )
            return uncertainty, probabilities

        semantic_end = self.semantic_start + self.semantic_dim
        if anchor.shape[-1] < semantic_end:
            raise ValueError(
                f"Anchor dimension {anchor.shape[-1]} is too small for "
                f"semantic slice [{self.semantic_start}:{semantic_end}]."
            )

        valid_start = (
            self.semantic_start + self.semantic_channel_start
        )
        valid_end = valid_start + self.num_language_classes
        logits = anchor[..., valid_start:valid_end]
        probabilities = torch.softmax(logits.float(), dim=-1).clamp_min(1e-6)
        entropy = -torch.sum(
            probabilities * torch.log(probabilities), dim=-1, keepdim=True
        )
        entropy = entropy / math.log(self.num_language_classes)
        return (
            entropy.to(dtype=anchor.dtype),
            probabilities.to(dtype=anchor.dtype),
        )

    def _observation_confidence(
        self,
        anchor,
        metas,
        observation_count=None,
    ):
        batch_size, num_anchor = anchor.shape[:2]
        if observation_count is not None:
            if observation_count.shape[:2] != (batch_size, num_anchor):
                raise ValueError(
                    "observation_count must have shape [B, N] or [B, N, 1], "
                    f"got {tuple(observation_count.shape)}."
                )
            if observation_count.ndim == 2:
                observation_count = observation_count[..., None]
            return (
                observation_count.to(
                    device=anchor.device, dtype=anchor.dtype
                )
                / self.visibility_saturation
            ).clamp(0.0, 1.0)

        if metas is None or "projection_mat" not in metas:
            return anchor.new_ones((batch_size, num_anchor, 1))

        unit_xyz = safe_sigmoid(anchor[..., :3])
        pc_range = self.pc_range.to(
            device=anchor.device, dtype=anchor.dtype
        )
        physical_xyz = (
            unit_xyz * (pc_range[3:] - pc_range[:3]) + pc_range[:3]
        )
        homogeneous_xyz = torch.cat(
            [physical_xyz, torch.ones_like(physical_xyz[..., :1])], dim=-1
        )

        projection_mat = metas["projection_mat"].to(
            device=anchor.device, dtype=anchor.dtype
        )
        projected = torch.einsum(
            "bcij,bnj->bcni", projection_mat, homogeneous_xyz
        )
        depth = projected[..., 2]
        image_xy = projected[..., :2] / depth.clamp_min(1e-5)[..., None]

        image_wh = metas.get("image_wh")
        if image_wh is not None:
            image_wh = image_wh.to(
                device=anchor.device, dtype=anchor.dtype
            )
            image_xy = image_xy / image_wh[:, :, None, :]

        valid = (
            (depth > 1e-5)
            & (image_xy[..., 0] > 0)
            & (image_xy[..., 0] < 1)
            & (image_xy[..., 1] > 0)
            & (image_xy[..., 1] < 1)
        )
        visible_cameras = valid.sum(dim=1).to(anchor.dtype)
        return (
            visible_cameras[..., None] / self.visibility_saturation
        ).clamp(0.0, 1.0)

    def _sample_language_targets(self, normalized_xyz, metas):
        """Sample labels 1--16 and remap them to prototype indices 0--15."""
        batch_size, num_points = normalized_xyz.shape[:2]
        ignore_index = -100
        if metas is None or "occ_label" not in metas:
            return torch.full(
                (batch_size, num_points),
                ignore_index,
                dtype=torch.long,
                device=normalized_xyz.device,
            )

        labels = metas["occ_label"].to(device=normalized_xyz.device)
        if labels.ndim != 4:
            raise ValueError(
                f"Expected occ_label [B, X, Y, Z], got {tuple(labels.shape)}."
            )

        unit_xyz = ((normalized_xyz + 1.0) * 0.5).clamp(0.0, 1.0)
        grid_shape = torch.tensor(
            labels.shape[1:4],
            device=normalized_xyz.device,
            dtype=unit_xyz.dtype,
        )
        indices = torch.floor(unit_xyz * grid_shape).long()
        indices[..., 0].clamp_(0, labels.shape[1] - 1)
        indices[..., 1].clamp_(0, labels.shape[2] - 1)
        indices[..., 2].clamp_(0, labels.shape[3] - 1)

        batch_indices = torch.arange(
            batch_size, device=normalized_xyz.device
        )[:, None].expand(-1, num_points)
        targets = labels[
            batch_indices,
            indices[..., 0],
            indices[..., 1],
            indices[..., 2],
        ].long()

        valid = (
            (targets >= self.valid_label_start)
            & (
                targets
                < self.valid_label_start + self.num_language_classes
            )
        )
        occ_mask = metas.get("occ_cam_mask")
        if occ_mask is not None:
            occ_mask = occ_mask.to(device=normalized_xyz.device)
            sampled_mask = occ_mask[
                batch_indices,
                indices[..., 0],
                indices[..., 1],
                indices[..., 2],
            ].bool()
            valid = valid & sampled_mask

        targets = targets - self.valid_label_start
        targets[~valid] = ignore_index
        return targets

    def _build_hash_topology(
        self,
        anchor,
        voxel_size,
        scale_index,
        decoder_index,
        topology_cache,
    ):
        batch_size, num_anchor = anchor.shape[:2]
        cache_root = topology_cache.setdefault("utp_topologies", {})
        cache_key = f"scale_{scale_index}_{voxel_size:g}"
        cached = cache_root.get(cache_key)

        refresh = cached is None
        if (
            not refresh
            and self.topology_refresh_interval > 0
            and decoder_index > 0
            and decoder_index % self.topology_refresh_interval == 0
        ):
            refresh = True
        if not refresh:
            refresh = (
                cached["batch_size"] != batch_size
                or cached["num_anchor"] != num_anchor
                or cached["inverse"].device != anchor.device
            )

        if not refresh:
            return cached, True

        pc_range = self.pc_range.to(device=anchor.device)
        spatial_extent = pc_range[3:] - pc_range[:3]
        spatial_shape_values = self.spatial_shapes[scale_index]
        spatial_shape = torch.tensor(
            spatial_shape_values,
            device=anchor.device,
            dtype=torch.long,
        )
        base_coords = topology_cache.get("current_base_coords")
        base_grid_size = topology_cache.get("base_grid_size")
        coords = None
        if base_coords is not None and base_grid_size is not None:
            base_grid_size = base_grid_size.to(
                device=anchor.device, dtype=torch.float32
            )
            scale_factor = voxel_size / base_grid_size
            rounded_factor = torch.round(scale_factor)
            if (
                base_coords.shape[:2] == (batch_size, num_anchor)
                and torch.allclose(
                    scale_factor,
                    rounded_factor,
                    atol=1e-5,
                    rtol=0.0,
                )
            ):
                coords = torch.div(
                    base_coords.long(),
                    rounded_factor.long().clamp_min(1),
                    rounding_mode="floor",
                )

        if coords is None:
            unit_xyz = safe_sigmoid(anchor[..., :3]).detach()
            physical_xyz = (
                unit_xyz * spatial_extent.to(unit_xyz.dtype)
                + pc_range[:3].to(unit_xyz.dtype)
            )
            coords = torch.floor(
                (physical_xyz - pc_range[:3].to(unit_xyz.dtype))
                / voxel_size
            ).long()

        coords = coords.detach()
        for axis in range(3):
            coords[..., axis].clamp_(
                0, spatial_shape_values[axis] - 1
            )

        batch_ids = torch.arange(
            batch_size, device=anchor.device, dtype=torch.long
        )[:, None].expand(-1, num_anchor)
        dim_x, dim_y, dim_z = spatial_shape.unbind()
        flat_keys = (
            (
                (batch_ids * dim_x + coords[..., 0]) * dim_y
                + coords[..., 1]
            )
            * dim_z
            + coords[..., 2]
        ).reshape(-1)
        unique_keys, inverse = torch.unique(
            flat_keys, sorted=True, return_inverse=True
        )
        cells_per_batch = dim_x * dim_y * dim_z
        region_batch = torch.div(
            unique_keys,
            cells_per_batch,
            rounding_mode="floor",
        ).long()
        counts = torch.bincount(
            inverse, minlength=unique_keys.numel()
        ).long()

        cached = {
            "inverse": inverse,
            "region_batch": region_batch,
            "counts": counts,
            "num_regions": int(unique_keys.numel()),
            "batch_size": batch_size,
            "num_anchor": num_anchor,
        }
        cache_root[cache_key] = cached
        return cached, False

    @staticmethod
    def _region_mean(values, inverse, counts):
        num_regions = counts.numel()
        output = values.new_zeros((num_regions, values.shape[-1]))
        output.index_add_(0, inverse, values)
        return output / counts.to(values.dtype).clamp_min(1)[:, None]

    @staticmethod
    def _region_difficulty(values, inverse, counts):
        """Combine mean and maximum difficulty for robust small-object routing."""
        num_regions = counts.numel()
        flat_values = values.reshape(-1)
        region_sum = values.new_zeros(num_regions)
        region_sum.index_add_(0, inverse, flat_values)
        region_mean = region_sum / counts.to(values.dtype).clamp_min(1)

        region_max = values.new_full((num_regions,), -torch.inf)
        region_max.scatter_reduce_(
            0,
            inverse,
            flat_values,
            reduce="amax",
            include_self=True,
        )
        return 0.5 * (region_mean + region_max)

    def _select_regions(
        self,
        region_difficulty,
        region_batch,
        scale_index,
        batch_size,
    ):
        selected = []
        ratio = self.route_ratios[scale_index]
        minimum = self.min_routed_regions[scale_index]
        maximum = self.max_routed_regions[scale_index]

        for batch_index in range(batch_size):
            candidates = torch.nonzero(
                region_batch == batch_index, as_tuple=False
            ).flatten()
            if candidates.numel() == 0:
                continue
            budget = max(
                minimum,
                int(math.ceil(candidates.numel() * ratio)),
            )
            if maximum is not None:
                budget = min(budget, maximum)
            budget = min(budget, candidates.numel())
            local_score = region_difficulty[candidates]
            local_selected = torch.topk(
                local_score,
                k=budget,
                largest=True,
                sorted=False,
            ).indices
            selected.append(candidates[local_selected])

        if not selected:
            return region_batch.new_empty((0,), dtype=torch.long)
        return torch.cat(selected, dim=0)

    @staticmethod
    def _route_points(inverse, selected, num_regions):
        """Map points in selected regions to compact routed-region indices."""
        region_to_routed = inverse.new_full((num_regions,), -1)
        region_to_routed[selected] = torch.arange(
            selected.numel(),
            device=inverse.device,
            dtype=inverse.dtype,
        )
        point_to_routed = region_to_routed[inverse]
        routed_point_indices = torch.nonzero(
            point_to_routed >= 0, as_tuple=False
        ).flatten()
        routed_inverse = point_to_routed[routed_point_indices]
        return routed_point_indices, routed_inverse

    def _region_targets(self, point_targets, inverse, num_regions):
        valid = point_targets != -100
        if not torch.any(valid):
            return torch.full(
                (num_regions,),
                -100,
                dtype=torch.long,
                device=point_targets.device,
            )

        flattened_vote_index = (
            inverse[valid] * self.num_language_classes
            + point_targets[valid]
        )
        votes = torch.bincount(
            flattened_vote_index,
            minlength=num_regions * self.num_language_classes,
        ).reshape(num_regions, self.num_language_classes)
        targets = votes.argmax(dim=-1)
        targets[votes.sum(dim=-1) == 0] = -100
        return targets

    def forward(
        self,
        instance_feature,
        anchor,
        language_features,
        metas=None,
        decoder_index=0,
        return_diagnostics=False,
        topology_cache: Optional[Dict] = None,
        observation_count=None,
    ):
        batch_size, num_anchor, channels = instance_feature.shape
        if channels != self.embed_dim:
            raise ValueError(
                f"Expected feature dimension {self.embed_dim}, got {channels}."
            )
        if topology_cache is None:
            topology_cache = {}

        unit_xyz = safe_sigmoid(anchor[..., :3])
        normalized_xyz = 2.0 * unit_xyz - 1.0
        geometry_descriptor = torch.cat(
            [unit_xyz, safe_sigmoid(anchor[..., 3:6])], dim=-1
        )

        uncertainty, probabilities = self._semantic_state(
            anchor, decoder_index
        )
        observation_confidence = self._observation_confidence(
            anchor,
            metas,
            observation_count=observation_count,
        )

        previous_probabilities = topology_cache.get(
            "previous_semantic_probabilities"
        )
        if (
            previous_probabilities is None
            or previous_probabilities.shape != probabilities.shape
        ):
            semantic_change = torch.zeros_like(uncertainty)
        else:
            semantic_change = 0.5 * torch.sum(
                torch.abs(
                    probabilities
                    - previous_probabilities.to(probabilities.dtype)
                ),
                dim=-1,
                keepdim=True,
            )
        topology_cache["previous_semantic_probabilities"] = (
            probabilities.detach()
        )

        difficulty_weight_sum = (
            self.semantic_uncertainty_weight
            + self.observation_uncertainty_weight
            + self.semantic_change_weight
        )
        if difficulty_weight_sum <= 0:
            raise ValueError("At least one difficulty weight must be positive.")
        difficulty = (
            self.semantic_uncertainty_weight * uncertainty
            + self.observation_uncertainty_weight
            * (1.0 - observation_confidence)
            + self.semantic_change_weight * semantic_change
        ) / difficulty_weight_sum
        difficulty = difficulty.clamp(0.0, 1.0)

        compute_language_targets = self.training or return_diagnostics
        point_targets = None
        if compute_language_targets:
            point_targets = self._sample_language_targets(
                normalized_xyz, metas
            ).reshape(-1)

        text_key = self.text_key_projection(language_features)
        text_value = self.text_value_projection(language_features)
        normalized_text_key = F.normalize(text_key.float(), dim=-1).to(
            text_key.dtype
        )

        original_feature = instance_feature
        working_feature = instance_feature
        scale_weights = torch.softmax(self.scale_logits, dim=0)

        language_logits = []
        language_targets = []
        diagnostics = []
        flat_language_evidence = instance_feature.new_zeros(
            (batch_size * num_anchor, self.num_language_classes)
        )
        flat_language_weight = instance_feature.new_zeros(
            (batch_size * num_anchor, 1)
        )

        flat_geometry = geometry_descriptor.reshape(-1, 6)
        flat_uncertainty = uncertainty.reshape(-1, 1)
        flat_observation = observation_confidence.reshape(-1, 1)
        flat_difficulty = difficulty.reshape(-1, 1)

        for scale_index, voxel_size in enumerate(self.voxel_sizes):
            topology, cache_reused = self._build_hash_topology(
                anchor,
                voxel_size,
                scale_index,
                decoder_index,
                topology_cache,
            )
            inverse = topology["inverse"]
            counts = topology["counts"]
            region_batch = topology["region_batch"]
            num_regions = topology["num_regions"]
            region_difficulty = self._region_difficulty(
                flat_difficulty, inverse, counts
            )
            selected = self._select_regions(
                region_difficulty,
                region_batch,
                scale_index,
                batch_size,
            )

            routed_point_indices, routed_inverse = self._route_points(
                inverse, selected, num_regions
            )

            attention = None
            learned_gate = None
            selected_targets = None
            selected_language_confidence = None
            selected_language_reliability = None
            selected_geometry = flat_geometry.new_empty((0, 6))

            if selected.numel() > 0:
                selected_counts = counts[selected]
                selected_batch = region_batch[selected]
                selected_difficulty = region_difficulty[selected, None]
                flat_feature = working_feature.reshape(
                    -1, self.embed_dim
                )
                selected_feature = self._region_mean(
                    flat_feature.index_select(0, routed_point_indices),
                    routed_inverse,
                    selected_counts,
                )
                selected_geometry = self._region_mean(
                    flat_geometry.index_select(0, routed_point_indices),
                    routed_inverse,
                    selected_counts,
                )
                selected_uncertainty = self._region_mean(
                    flat_uncertainty.index_select(0, routed_point_indices),
                    routed_inverse,
                    selected_counts,
                )
                selected_observation = self._region_mean(
                    flat_observation.index_select(0, routed_point_indices),
                    routed_inverse,
                    selected_counts,
                )
                selected_feature = self.region_norms[scale_index](
                    selected_feature
                    + self.geometry_projections[scale_index](
                        selected_geometry
                    )
                )

                query = F.normalize(
                    self.query_projections[scale_index](
                        selected_feature
                    ).float(),
                    dim=-1,
                ).to(selected_feature.dtype)
                selected_key = normalized_text_key[selected_batch]
                attention_logits = torch.einsum(
                    "mc,mkc->mk", query, selected_key
                )
                attention = torch.softmax(
                    attention_logits / self.prototype_temperature,
                    dim=-1,
                )
                language_context = torch.einsum(
                    "mk,mkc->mc",
                    attention,
                    text_value[selected_batch],
                )
                language_context = self.context_projections[scale_index](
                    language_context
                )

                gate_input = torch.cat(
                    [
                        selected_feature,
                        language_context,
                        selected_uncertainty,
                        selected_observation,
                    ],
                    dim=-1,
                )
                learned_gate = self.fusion_gates[scale_index](gate_input)
                attention_float = attention.float().clamp_min(1e-6)
                selected_language_confidence = (
                    1.0
                    + torch.sum(
                        attention_float * torch.log(attention_float),
                        dim=-1,
                        keepdim=True,
                    )
                    / math.log(self.num_language_classes)
                ).to(attention.dtype)
                selected_language_reliability = (
                    selected_difficulty
                    * learned_gate.mean(dim=-1, keepdim=True)
                    * selected_language_confidence
                ).clamp(0.0, 1.0)
                raw_language_delta = (
                    learned_gate
                    * selected_difficulty
                    * (language_context - selected_feature)
                )

                fusion_norm = self.fusion_norms[scale_index]
                baseline_selected = fusion_norm(selected_feature)
                fused_selected = fusion_norm(
                    selected_feature
                    + self.dropout(raw_language_delta)
                )
                selected_delta = self.delta_projections[scale_index](
                    fused_selected - baseline_selected
                )

                if compute_language_targets:
                    selected_targets = self._region_targets(
                        point_targets.index_select(
                            0, routed_point_indices
                        ),
                        routed_inverse,
                        selected.numel(),
                    )
                    normalized_feature = F.normalize(
                        fused_selected.float(), dim=-1
                    )
                    logits = torch.einsum(
                        "mc,mkc->mk",
                        normalized_feature,
                        F.normalize(
                            text_key[selected_batch].float(), dim=-1
                        ),
                    )
                    language_logits.append(
                        logits / self.language_temperature
                    )
                    language_targets.append(selected_targets)

                routed_language_reliability = (
                    scale_weights[scale_index].to(attention.dtype)
                    * selected_language_reliability[routed_inverse]
                )
                flat_language_evidence = torch.index_add(
                    flat_language_evidence,
                    0,
                    routed_point_indices,
                    routed_language_reliability
                    * attention[routed_inverse],
                )
                flat_language_weight = torch.index_add(
                    flat_language_weight,
                    0,
                    routed_point_indices,
                    routed_language_reliability,
                )
                routed_delta = selected_delta[routed_inverse]
                flat_feature = torch.index_add(
                    flat_feature,
                    0,
                    routed_point_indices,
                    scale_weights[scale_index].to(flat_feature.dtype)
                    * routed_delta,
                )
                working_feature = flat_feature.reshape(
                    batch_size, num_anchor, self.embed_dim
                )

            if return_diagnostics:
                diagnostic_region_uncertainty = self._region_mean(
                    flat_uncertainty, inverse, counts
                )
                diagnostic_region_observation = self._region_mean(
                    flat_observation, inverse, counts
                )
                diagnostics.append(
                    {
                        "voxel_size": voxel_size,
                        "num_regions": num_regions,
                        "num_routed_regions": int(selected.numel()),
                        "num_routed_points": int(
                            routed_point_indices.numel()
                        ),
                        "cache_reused": cache_reused,
                        "selected_regions": selected.detach(),
                        "region_batch": region_batch.detach(),
                        "difficulty": region_difficulty.detach(),
                        "uncertainty": (
                            diagnostic_region_uncertainty.detach()
                        ),
                        "observation_confidence": (
                            diagnostic_region_observation.detach()
                        ),
                        "gate": (
                            learned_gate.detach()
                            if learned_gate is not None
                            else None
                        ),
                        "attention": (
                            attention.detach()
                            if attention is not None
                            else None
                        ),
                        "language_confidence": (
                            selected_language_confidence.detach()
                            if selected_language_confidence is not None
                            else None
                        ),
                        "language_reliability": (
                            selected_language_reliability.detach()
                            if selected_language_reliability is not None
                            else None
                        ),
                        "points": (
                            2.0 * selected_geometry[:, :3].detach() - 1.0
                        ),
                        "targets": (
                            selected_targets.detach()
                            if selected_targets is not None
                            else None
                        ),
                    }
                )

        language_context = working_feature - original_feature
        output = self.output(language_context)
        output = output * self.layer_scale[None, None, :]
        flat_language_posterior = (
            flat_language_evidence
            / flat_language_weight.clamp_min(1e-6)
        )
        language_posterior = flat_language_posterior.reshape(
            batch_size, num_anchor, self.num_language_classes
        )
        language_reliability = flat_language_weight.clamp(0.0, 1.0).reshape(
            batch_size, num_anchor, 1
        )

        return output, {
            "language_logits": language_logits,
            "language_targets": language_targets,
            "diagnostics": diagnostics,
            "language_posterior": language_posterior,
            "language_reliability": language_reliability,
            "support_difficulty": difficulty,
        }