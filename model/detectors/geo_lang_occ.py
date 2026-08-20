
from mmseg.models import SEGMENTORS
from mmseg.models import build_backbone
from .base_segmentor import CustomBaseSegmentor
import torch
import clip
import torch.nn.functional as F

@SEGMENTORS.register_module()
class GeoLangOcc(CustomBaseSegmentor):

    def __init__(
        self,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        img_backbone_out_indices=[1, 2, 3],
        extra_img_backbone=None,
        label=None,
        use_text=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.freeze_img_backbone = freeze_img_backbone
        self.freeze_img_neck = freeze_img_neck
        self.img_backbone_out_indices = img_backbone_out_indices
        self.use_text = use_text

        if freeze_img_backbone:
            self.img_backbone.requires_grad_(False)
        if freeze_img_neck:
            self.img_neck.requires_grad_(False)
        if extra_img_backbone is not None:
            self.extra_img_backbone = build_backbone(extra_img_backbone)


        if self.use_text:
            print('Loading CLIP model......')
            self.clip_pretrained, _ = clip.load("ViT-B/32", device='cuda', jit=False, download_root='/ckpts/clip')
            self._freeze(self.clip_pretrained)
            self.clip_pretrained.eval()

            self.img_backbone_out_indices = img_backbone_out_indices
            self.label = label
            self.label_len = len(label)
            tokenized_texts = clip.tokenize(self.label).to(
                next(self.clip_pretrained.parameters()).device
            )
            with torch.no_grad():
                text_features = self.clip_pretrained.encode_text(tokenized_texts)
                text_features = F.normalize(text_features.float(), dim=-1)
            self.register_buffer(
                "cached_language_features",
                text_features.detach(),
                persistent=False,
            )

            del self.clip_pretrained

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
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        result.update({'ms_img_feats': img_feats_reshaped})


        if self.use_text:
            language_features = self.cached_language_features.to(
                device=device,
                dtype=img_feats_reshaped[0].dtype,
            )
            language_features = language_features.unsqueeze(0).expand(B, -1, -1)

            result.update({'language_features': language_features})

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