import numpy as np
import torch, torch.nn as nn

from mmengine.registry import MODELS
from .base_head import BaseTaskHead
from ..utils.utils import get_rotation_matrix


@MODELS.register_module()
class GaussianHead(BaseTaskHead):
    def __init__(
        self, 
        init_cfg=None,
        apply_loss_type=None,
        num_classes=18,
        empty_args=None,
        with_empty=False,
        cuda_kwargs=None,
        dataset_type='nusc',
        empty_label=17,
        use_localaggprob=False,
        use_localaggprob_fast=False,
        g2v_backend="prob",
        combine_geosem=False,
        language_label_start=1,
        num_language_classes=16,
        **kwargs,
    ):
        super().__init__(init_cfg)
        
        self.num_classes = num_classes
        self.use_localaggprob = use_localaggprob
        self.g2v_backend = str(g2v_backend).lower()
        self.use_ulr_g2v = (
            use_localaggprob and self.g2v_backend == "ulr"
        )
        aggregator_kwargs = dict(cuda_kwargs)
        if use_localaggprob:
            if self.g2v_backend == "ulr":
                import local_aggregate_ulr
                self.aggregator = local_aggregate_ulr.LocalAggregator(
                    **aggregator_kwargs
                )
            elif self.g2v_backend == "s2go":
                for key in (
                    "adaptive_support",
                    "support_threshold",
                    "support_beta",
                    "support_min_threshold",
                ):
                    aggregator_kwargs.pop(key, None)
                import local_aggregate_s2go
                self.aggregator = local_aggregate_s2go.LocalAggregator(
                    **aggregator_kwargs
                )
            elif self.g2v_backend == "prob" and use_localaggprob_fast:
                for key in (
                    "adaptive_support",
                    "support_threshold",
                    "support_beta",
                    "support_min_threshold",
                ):
                    aggregator_kwargs.pop(key, None)
                import local_aggregate_prob_fast
                self.aggregator = local_aggregate_prob_fast.LocalAggregator(
                    **aggregator_kwargs
                )
            elif self.g2v_backend == "prob":
                for key in (
                    "adaptive_support",
                    "support_threshold",
                    "support_beta",
                    "support_min_threshold",
                ):
                    aggregator_kwargs.pop(key, None)
                import local_aggregate_prob
                self.aggregator = local_aggregate_prob.LocalAggregator(
                    **aggregator_kwargs
                )
            else:
                raise ValueError(
                    "g2v_backend must be one of {'prob', 's2go', 'ulr'}, "
                    f"got {g2v_backend!r}."
                )
        else:
            import local_aggregate
            self.aggregator = local_aggregate.LocalAggregator(
                **aggregator_kwargs
            )
        
        self.combine_geosem = combine_geosem
        if with_empty:
            self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float) * 10.0)
            self.register_buffer('empty_mean', torch.tensor(empty_args['mean'])[None, None, :])
            self.register_buffer('empty_scale', torch.tensor(empty_args['scale'])[None, None, :])
            self.register_buffer('empty_rot', torch.tensor([1., 0., 0., 0.])[None, None, :])
            self.register_buffer('empty_sem', torch.zeros(self.num_classes)[None, None, :])
            self.register_buffer('empty_opa', torch.ones(1)[None, None, :])
        self.with_emtpy = with_empty
        self.empty_args = empty_args
        self.dataset_type = dataset_type
        self.empty_label = empty_label
        self.language_label_start = int(language_label_start)
        self.num_language_classes = int(num_language_classes)
        language_label_end = (
            self.language_label_start + self.num_language_classes
        )
        if (
            self.language_label_start < 0
            or language_label_end > self.num_classes
        ):
            raise ValueError(
                "The language class range must fit inside num_classes."
            )

        if apply_loss_type == 'all':
            self.apply_loss_type = 'all'
        elif 'random' in apply_loss_type:
            self.apply_loss_type = 'random'
            self.random_apply_loss_layers = int(apply_loss_type.split('_')[1])
        elif 'fixed' in apply_loss_type:
            self.apply_loss_type = 'fixed'
            self.fixed_apply_loss_layers = [int(item) for item in apply_loss_type.split('_')[1:]]
            print(f"Supervised fixed layers: {self.fixed_apply_loss_layers}")
        else:
            raise NotImplementedError
        self.register_buffer('zero_tensor', torch.zeros(1, dtype=torch.float))

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def _sampling(self, gt_xyz, gt_label, gt_mask=None):
        if gt_mask is None:
            gt_label = gt_label.flatten(1)
            gt_xyz = gt_xyz.flatten(1, 3)
        else:
            assert gt_label.shape[0] == 1, "OccLoss does not support bs > 1"
            gt_label = gt_label[gt_mask].reshape(1, -1)
            gt_xyz = gt_xyz[gt_mask].reshape(1, -1, 3)
        return gt_xyz, gt_label

    def prepare_gaussian_args(self, gaussians):
        means = gaussians.means # b, g, 3
        scales = gaussians.scales # b, g, 3
        rotations = gaussians.rotations # b, g, 4
        opacities = gaussians.semantics # b, g, c
        origi_opa = gaussians.opacities # b, g, 1
        if origi_opa.numel() == 0:
            origi_opa = torch.ones_like(opacities[..., :1], requires_grad=False)
        if self.with_emtpy:
            assert opacities.shape[-1] == self.num_classes - 1
            if 'kitti' in self.dataset_type:
                opacities = torch.cat([torch.zeros_like(opacities[..., :1]), opacities], dim=-1)
            else:
                opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1])], dim=-1)
            means = torch.cat([means, self.empty_mean], dim=1)
            scales = torch.cat([scales, self.empty_scale], dim=1)
            rotations = torch.cat([rotations, self.empty_rot], dim=1)
            empty_sem = self.empty_sem.clone()
            empty_sem[..., self.empty_label] += self.empty_scalar
            opacities = torch.cat([opacities, empty_sem], dim=1)
            origi_opa = torch.cat([origi_opa, self.empty_opa], dim=1)
        elif self.use_localaggprob:
            assert opacities.shape[-1] == self.num_classes - 1
            opacities = opacities.softmax(dim=-1)
            if 'kitti' in self.dataset_type:
                opacities = torch.cat([torch.zeros_like(opacities[..., :1]), opacities], dim=-1)
            else:
                opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1])], dim=-1)

        R = get_rotation_matrix(rotations) # b, g, 3, 3
        inv_scale_sq = scales.reciprocal().square()
        CovInv = torch.matmul(
            R.transpose(-1, -2),
            torch.matmul(torch.diag_embed(inv_scale_sq), R),
        )
        return means, origi_opa, opacities, scales, CovInv

    def prepare_ulr_args(self, representation_item, semantics):
        batch_size, num_gaussians, _ = semantics.shape
        posterior = representation_item.get("language_posterior")
        reliability = representation_item.get("language_reliability")
        difficulty = representation_item.get("support_difficulty")

        if posterior is None:
            return (
                torch.zeros_like(semantics),
                semantics.new_zeros((batch_size, num_gaussians, 1)),
                semantics.new_zeros((batch_size, num_gaussians, 1)),
            )

        expected_shape = (
            batch_size,
            num_gaussians,
            self.num_language_classes,
        )
        if tuple(posterior.shape) != expected_shape:
            raise ValueError(
                "language_posterior must have shape "
                f"{expected_shape}, got {tuple(posterior.shape)}."
            )
        if reliability is None or tuple(reliability.shape) != (
            batch_size,
            num_gaussians,
            1,
        ):
            raise ValueError(
                "language_reliability must have shape "
                f"{(batch_size, num_gaussians, 1)}."
            )
        if difficulty is None or tuple(difficulty.shape) != (
            batch_size,
            num_gaussians,
            1,
        ):
            raise ValueError(
                "support_difficulty must have shape "
                f"{(batch_size, num_gaussians, 1)}."
            )

        language_end = (
            self.language_label_start + self.num_language_classes
        )
        language_semantics = torch.cat(
            [
                semantics.new_zeros(
                    (
                        batch_size,
                        num_gaussians,
                        self.language_label_start,
                    )
                ),
                posterior.to(semantics.dtype),
                semantics.new_zeros(
                    (
                        batch_size,
                        num_gaussians,
                        self.num_classes - language_end,
                    )
                ),
            ],
            dim=-1,
        )
        language_residual = language_semantics - semantics
        return (
            language_residual,
            reliability.to(semantics.dtype).clamp(0.0, 1.0),
            difficulty.to(semantics.dtype).clamp(0.0, 1.0),
        )

    def forward(
        self,
        representation,
        metas=None,
        **kwargs
    ):
        num_decoder = len(representation)
        if not self.training:
            apply_loss_layers = [num_decoder - 1]
        elif self.apply_loss_type == "all":
            apply_loss_layers = list(range(num_decoder))
        elif self.apply_loss_type == "random":
            if self.random_apply_loss_layers > 1:
                apply_loss_layers = np.random.choice(num_decoder - 1, self.random_apply_loss_layers - 1, False)
                apply_loss_layers = apply_loss_layers.tolist() + [num_decoder - 1]
            else:
                apply_loss_layers = [num_decoder - 1]
        elif self.apply_loss_type == 'fixed':
            apply_loss_layers = self.fixed_apply_loss_layers
        else:
            raise NotImplementedError

        prediction = []
        bin_logits = []
        density = []
        occ_xyz = metas['occ_xyz'].to(self.zero_tensor.device)
        occ_label = metas['occ_label'].to(self.zero_tensor.device)
        occ_cam_mask = metas['occ_cam_mask'].to(self.zero_tensor.device)
        sampled_xyz, sampled_label = self._sampling(occ_xyz, occ_label, None)
        for idx in apply_loss_layers:
            representation_item = representation[idx]
            gaussians = representation_item['gaussian']

            means, origi_opa, opacities, scales, CovInv = self.prepare_gaussian_args(gaussians)
            bs, g = means.shape[:2]

            if self.use_ulr_g2v:
                (
                    language_residual,
                    language_reliability,
                    support_difficulty,
                ) = self.prepare_ulr_args(
                    representation_item, opacities
                )
                semantics = self.aggregator(
                    sampled_xyz.clone().float(),
                    means,
                    origi_opa.reshape(bs, g),
                    opacities,
                    language_residual,
                    language_reliability,
                    support_difficulty,
                    scales,
                    CovInv,
                )
            else:
                semantics = self.aggregator(
                    sampled_xyz.clone().float(), 
                    means, 
                    origi_opa.reshape(bs, g),
                    opacities,
                    scales,
                    CovInv) # 1, c, n
            if self.use_localaggprob:
                if self.combine_geosem:
                    if 'kitti' in self.dataset_type:
                        sem = semantics[0][:, 1:] * semantics[1].unsqueeze(-1)  # 取出第1类及之后作为前景
                        geo = 1 - semantics[1].unsqueeze(-1)  # 计算空（empty）的概率
                        geosem = torch.cat([geo, sem], dim=-1)  # 把空的概率拼到最前面
                    else:
                        sem = semantics[0][:, :-1] * semantics[1].unsqueeze(-1)
                        geo = 1 - semantics[1].unsqueeze(-1)
                        geosem = torch.cat([sem, geo], dim=-1)
                else:
                    geosem = semantics[0]
                    
                prediction.append(geosem[None].transpose(1, 2))
                bin_logits.append(semantics[1][None])
                density.append(semantics[2][None])
            else:
                prediction.append(semantics[None].transpose(1, 2))
        
        if self.use_localaggprob and not self.combine_geosem:
            threshold = kwargs.get("sigmoid_thresh", 0.5)
            final_semantics = prediction[-1].argmax(dim=1)
            final_occupancy = bin_logits[-1] > threshold
            final_prediction = torch.ones_like(final_semantics) * self.empty_label
            final_prediction[final_occupancy] = final_semantics[final_occupancy]
        else:
            final_prediction = prediction[-1].argmax(dim=1)
        
        return {
            'pred_occ': prediction,
            'bin_logits': bin_logits,
            'density': density,
            'sampled_label': sampled_label,
            'sampled_xyz': sampled_xyz,
            'occ_mask': occ_cam_mask,
            'final_occ': final_prediction,
            'gaussian': representation[-1]['gaussian'],
            'gaussians': [r['gaussian'] for r in representation]
        }