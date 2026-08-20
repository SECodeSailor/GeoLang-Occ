from typing import List, Optional
import torch, torch.nn as nn
from mmseg.registry import MODELS
from mmengine import build_from_cfg
from .base_encoder import BaseEncoder


@MODELS.register_module()
class GaussianOccEncoder(BaseEncoder):
    def __init__(
        self,
        anchor_encoder: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        mid_refine_layer: dict = None,
        spconv_layer: dict = None,
        self_rwkv_layer=None,
        clg_layer=None,
        num_decoder: int = 6,
        operation_order: Optional[List[str]] = None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.num_decoder = num_decoder

        if operation_order is None:
            operation_order = [
                "spconv",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.anchor_encoder = build(anchor_encoder, MODELS)
        self.op_config_map = {
            "norm": [norm_layer, MODELS],
            "ffn": [ffn, MODELS],
            "deformable": [deformable_model, MODELS],
            "refine": [refine_layer, MODELS],
            "mid_refine":[mid_refine_layer, MODELS],
            "spconv": [spconv_layer, MODELS],
            "self_rwkv": [self_rwkv_layer, MODELS],
            "clg": [clg_layer, MODELS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "sfmodeulation":
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(
        self,
        representation,
        rep_features,
        language_features=None,
        T_proj_features=None,
        T_orig_features=None,
        ms_img_feats=None,
        metas=None,
        vl_feats=None,
        **kwargs
    ):
        feature_maps = ms_img_feats
        vl_feats_maps = vl_feats
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        instance_feature = rep_features
        anchor = representation

        anchor_embed = self.anchor_encoder(anchor)

        prediction = []
        t_proj_features = []
        language_logits = []
        language_targets = []
        clg_diagnostics = []
        return_diagnostics = kwargs.get("return_diagnostics", False)
        # Shared only within this frame. Self-RWKV contributes its sparse
        # coordinates and every UTP-CLG stage reuses the resulting topology.
        geometry_cache = {}
        observation_count = None
        language_posterior = None
        language_reliability = None
        support_difficulty = None
        for i, op in enumerate(self.operation_order):
            # if op == 'spconv':
            #     instance_feature = self.layers[i](
            #         instance_feature,
            #         anchor)
            if op == 'self_rwkv':
                rwkv_kwargs = {}
                if getattr(
                    self.layers[i], "supports_geometry_cache", False
                ):
                    rwkv_kwargs.update(
                        geometry_cache=geometry_cache,
                        decoder_index=len(prediction),
                    )
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    metas=metas,
                    **rwkv_kwargs,
                )
            elif op == 'clg':
                clg_kwargs = {}
                if getattr(
                    self.layers[i], "supports_topology_cache", False
                ):
                    clg_kwargs.update(
                        topology_cache=geometry_cache,
                        observation_count=observation_count,
                    )
                instance_feature, clg_aux = self.layers[i](
                    instance_feature,
                    anchor,
                    language_features,
                    metas=metas,
                    decoder_index=len(prediction),
                    return_diagnostics=return_diagnostics,
                    **clg_kwargs,
                )
                language_logits.extend(clg_aux["language_logits"])
                language_targets.extend(clg_aux["language_targets"])
                language_posterior = clg_aux.get("language_posterior")
                language_reliability = clg_aux.get(
                    "language_reliability"
                )
                support_difficulty = clg_aux.get("support_difficulty")
                if return_diagnostics:
                    clg_diagnostics.append(clg_aux["diagnostics"])
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "identity":
                identity = instance_feature
            elif op == "add":
                instance_feature = instance_feature + identity
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                )
                observation_count = getattr(
                    self.layers[i], "last_observation_count", None
                )
            elif "refine" in op:
                anchor, gaussian = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                )
                prediction_item = {"gaussian": gaussian}
                if language_posterior is not None:
                    prediction_item["language_posterior"] = (
                        language_posterior
                    )
                    prediction_item["language_reliability"] = (
                        language_reliability
                    )
                    prediction_item["support_difficulty"] = (
                        support_difficulty
                    )
                prediction.append(prediction_item)
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
            else:
                raise NotImplementedError(f"{op} is not supported.")

        return {"representation": prediction,
                "T_proj_features": t_proj_features,
                "language_logits": language_logits,
                "language_targets": language_targets,
                "language_loss_reference": instance_feature,
                "clg_diagnostics": clg_diagnostics}