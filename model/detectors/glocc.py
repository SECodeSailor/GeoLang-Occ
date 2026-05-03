
from mmseg.models import SEGMENTORS
from mmseg.models import build_backbone
from mmengine.model import BaseModule
import torch, time
import clip
import torch.nn as nn
from ..utils.safe_ops import linear_relu_ln
from mmcv.cnn import Scale
# from ..utils.utils import generate_descriptive_features, build_adj_matrix_with_sbert
# from ..encoder.gaussian_encoder.gat_layer import CategoryGATEnhancer


@SEGMENTORS.register_module()
class GeoLangOcc(BaseModule):

    def __init__(
        self,
        img_backbone=None,
        img_neck=None,
        lifter=None,
        encoder=None,
        head=None, 
        freeze_img_backbone=False,
        freeze_img_neck=False,
        freeze_lifter=False,
        img_backbone_out_indices=[1, 2, 3],
        extra_img_backbone=None,
        label=None,
        use_text=False,
        # use_post_fusion=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # self.fp16_enabled = False
        self.freeze_img_backbone = freeze_img_backbone
        self.freeze_img_neck = freeze_img_neck
        self.img_backbone_out_indices = img_backbone_out_indices
        # self.use_post_fusion = use_post_fusion
        self.use_text = use_text

        if img_backbone is not None:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            try:
                self.img_neck = builder.build_neck(img_neck)
            except:
                self.img_neck = MODELS.build(img_neck)
        if lifter is not None:
            self.lifter = builder.build_head(lifter)
        if encoder is not None:
            self.encoder = builder.build_head(encoder)
        if head is not None:
            self.head = builder.build_head(head)

        if freeze_img_backbone:
            self.img_backbone.requires_grad_(False)
        if freeze_img_neck:
            self.img_neck.requires_grad_(False)
        if freeze_lifter:
            self.lifter.requires_grad_(False)
            if hasattr(self.lifter, "random_anchors"):
                self.lifter.random_anchors.requires_grad = True
        if extra_img_backbone is not None:
            self.extra_img_backbone = build_backbone(extra_img_backbone)


        if self.use_text:
            print('Loading CLIP model......')
            self.clip_pretrained, _ = clip.load("ViT-B/32", device='cuda', jit=False, download_root='/ckpt/clip')
            self._freeze(self.clip_pretrained)
            self.clip_pretrained.eval()

            self.img_backbone_out_indices = img_backbone_out_indices
            self.label = label
            self.label_len = len(label)
            self.texts = []
            for class_i in range(self.label_len):
                text = self.label[class_i]
                text_feat = clip.tokenize(text)
                self.texts.append(text_feat)

        #     self.adj_matrix = build_adj_matrix_with_sbert(self.label, similarity_threshold=0.5)

        # self.gat_enhancer = CategoryGATEnhancer(embed_dim=512)

    def _freeze(self, model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def extract_img_feat(self, imgs, **kwargs):
        """Extract features of images."""
        B = imgs.size(0)
        result = {}
        device = imgs.device

        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats = []
        for idx in self.img_backbone_out_indices:
            img_feats.append(img_feats_backbone[idx])
        img_feats = self.img_neck(img_feats)
        if isinstance(img_feats, dict):
            secondfpn_out = img_feats["secondfpn_out"][0]
            BN, C, H, W = secondfpn_out.shape
            secondfpn_out = secondfpn_out.view(B, int(BN / B), C, H, W)
            img_feats = img_feats["fpn_out"]
            result.update({"secondfpn_out": secondfpn_out})

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            # if self.use_post_fusion:
            #     img_feats_reshaped.append(img_feat.unsqueeze(1))
            # else:
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        result.update({'ms_img_feats': img_feats_reshaped})


        if self.use_text:
            """Extract features of texts."""
            with torch.no_grad():
                text_features = [self.clip_pretrained.encode_text(text.to(device)) for text in self.texts]
            if isinstance(text_features, list):
                text_features = torch.cat(text_features, dim=0).to(img_feats_reshaped[0])
            if text_features.dim() != 3:
                language_features = text_features.unsqueeze(0).expand(B, -1, -1).to(img_feats_reshaped[0])

            result.update({'language_features': language_features})
            # result.update({'T_proj_features': None,
            #                'T_orig_features': text_features})

        return result
    
    def forward_extra_img_backbone(self, imgs, **kwargs):
        """Extract features of images."""
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.extra_img_backbone(imgs)

        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())

        img_feats_backbone_reshaped = []
        for img_feat_backbone in img_feats_backbone:
            BN, C, H, W = img_feat_backbone.size()
            img_feats_backbone_reshaped.append(
                img_feat_backbone.view(B, int(BN / B), C, H, W))
        return img_feats_backbone_reshaped

    def forward(self,
                imgs=None,
                metas=None,
                points=None,
                extra_backbone=False,
                occ_only=False,
                rep_only=False,
                **kwargs,
        ):
        """Forward training function.
        """
        if extra_backbone:
            return self.forward_extra_img_backbone(imgs=imgs)
        
        results = {
            'imgs': imgs,
            'metas': metas,
            'points': points
        }
        results.update(kwargs)
        outs = self.extract_img_feat(**results)
        results.update(outs)

        outs = self.lifter(**results)

        results.update(outs)
        outs = self.encoder(**results)
        if rep_only:
            return outs['representation']
        results.update(outs)
        if occ_only and hasattr(self.head, "forward_occ"):
            outs = self.head.forward_occ(**results)
        else:
            outs = self.head(**results)
        results.update(outs)
        return results
