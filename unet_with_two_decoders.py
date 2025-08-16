# nnUNet\nnunetv2\dynamic_network_architectures\architectures\unet_with_two_decoders.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import weakref


def conv3x3(in_ch, out_ch, ks=(3, 3, 3), bias=True):
    return nn.Conv3d(in_ch, out_ch, kernel_size=ks, padding=tuple(k // 2 for k in ks), bias=bias)


class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, ks=(3, 3, 3), bias=True,
                 norm=nn.InstanceNorm3d, nonlin=nn.LeakyReLU):
        super().__init__()
        self.need_proj = in_ch != out_ch
        self.conv1 = conv3x3(in_ch, out_ch, ks, bias=bias)
        self.bn1 = norm(out_ch, eps=1e-5, affine=True)
        self.act1 = nonlin(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch, ks, bias=bias)
        self.bn2 = norm(out_ch, eps=1e-5, affine=True)
        self.proj = nn.Conv3d(in_ch, out_ch, 1, bias=False) if self.need_proj else nn.Identity()
        self.act2 = nonlin(inplace=True)

    def forward(self, x):
        idt = self.proj(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act2(x + idt)


class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=(2, 2, 2)):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=stride, stride=stride)
    def forward(self, x):
        return self.up(x)


class DecoderShim(nn.Module):
    """
    Gives a `.decoder.deep_supervision` property so nnUNetTrainer can toggle it.
    """
    def __init__(self, owner: "TwoHeadUNet3D"):
        super().__init__()
        self._owner_ref = weakref.ref(owner)
    
    @property
    def deep_supervision(self):
        owner = self._owner_ref()
        return getattr(owner, "_deep_supervision", True) if owner is not None else True
    
    @deep_supervision.setter
    def deep_supervision(self, v: bool):
        owner = self._owner_ref()
        if owner is not None:
            setattr(owner, "_deep_supervision", bool(v))


class TwoHeadUNet3D(nn.Module):
    """
    Shared encoder → (A) segmentation decoder (deep supervision) + (B) classification decoder.
    forward(x) -> (seg_maps: List[Tensor], cls_logits: Tensor[B, 3])
    """
    def __init__(self, plans_manager, dataset_json: Dict[str, Any], configuration_manager):
        super().__init__()
        self.pm = plans_manager
        self.dj = dataset_json or {}
        self.cm = configuration_manager

        self._deep_supervision = True
        self.decoder = DecoderShim(self)

        # arch params with defaults (ResEnc-M 3d_fullres)
        arch = getattr(self.cm, "architecture", None) if self.cm is not None else None
        if isinstance(arch, dict) and "arch_kwargs" in arch:
            cfg = arch["arch_kwargs"]
        else:
            print("[TwoHeads] WARNING: configuration_manager.architecture['arch_kwargs'] not found; using defaults.")
            cfg = {
                "n_stages": 6,
                "features_per_stage": [32, 64, 128, 256, 320, 320],
                "kernel_sizes": [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
                "conv_bias": True,
            }

        n_stages: int = cfg.get("n_stages", 6)
        f_per: List[int] = cfg.get("features_per_stage", [32, 64, 128, 256, 320, 320])
        ks_list: List[List[int]] = cfg.get("kernel_sizes", [[1,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]])
        strides: List[List[int]] = cfg.get("strides", [[1,1,1],[1,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]])
        nconv_dec: List[int] = cfg.get("n_conv_per_stage_decoder", [2,2,2,2,2])
        bias: bool = cfg.get("conv_bias", True)

        # infer I/O
        in_ch = len(self.dj.get("modality", self.dj.get("modality_names", [0])))  # CT=1
        if "labels" in self.dj:
            num_seg_classes = len(self.dj["labels"])  # includes background
        else:
            num_seg_classes = 3
        num_cls = 3  # subtypes
        self.num_classes = num_cls

        # encoder 
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = in_ch
        for s in range(n_stages):
            out_ch = f_per[s]
            self.enc_blocks.append(ResidualBlock3D(ch, out_ch, tuple(ks_list[s]), bias=bias))
            ch = out_ch
            if s < n_stages - 1:
                self.downs.append(nn.Conv3d(ch, ch, kernel_size=strides[s + 1], stride=strides[s + 1], bias=False))
        bott_ch = f_per[-1]

        # segmentation decoder
        dec_levels = min(5, n_stages - 1)
        self.seg_ups = nn.ModuleList()
        self.seg_dec_blocks = nn.ModuleList()
        self.seg_heads = nn.ModuleList()

        dec_in = bott_ch
        for i in range(dec_levels):
            skip_stage = n_stages - 2 - i
            skip_ch = f_per[skip_stage]
            self.seg_ups.append(Up3D(dec_in, skip_ch, stride=tuple(strides[skip_stage + 1])))
            dec_block_in = skip_ch + skip_ch
            dec_block_out = skip_ch
            blocks = [ResidualBlock3D(dec_block_in, dec_block_out, tuple(ks_list[skip_stage]), bias=bias)]
            for _ in range(max(0, nconv_dec[min(i, len(nconv_dec)-1)] - 1)):
                blocks.append(ResidualBlock3D(dec_block_out, dec_block_out, tuple(ks_list[skip_stage]), bias=bias))
            self.seg_dec_blocks.append(nn.Sequential(*blocks))
            self.seg_heads.append(nn.Conv3d(dec_block_out, num_seg_classes, kernel_size=1, bias=True))
            dec_in = dec_block_out

        # classification decoder
        cls_up1_out = f_per[-2]
        cls_up2_out = f_per[-3] if n_stages >= 3 else cls_up1_out
        self.cls_up1 = Up3D(bott_ch, cls_up1_out, stride=tuple(strides[-1]))
        self.cls_block1 = ResidualBlock3D(cls_up1_out + f_per[-2], cls_up1_out, tuple(ks_list[-2]), bias=bias)
        self.cls_up2 = Up3D(cls_up1_out, cls_up2_out, stride=tuple(strides[-2]))
        self.cls_block2 = ResidualBlock3D(cls_up2_out + f_per[-3], cls_up2_out, tuple(ks_list[-3]), bias=bias)
        self.cls_drop = nn.Dropout3d(p=0.25)
        self._ensure_cls_head(bott_ch)
        self.cls_head.train(self.training)

    # subroutines
    def _encode(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips: List[torch.Tensor] = []
        for s, block in enumerate(self.enc_blocks):
            x = block(x)
            skips.append(x)
            if s < len(self.downs):
                x = self.downs[s](x)
        return x, skips  # bottleneck, skips

    def _seg_decode(self, bottleneck: torch.Tensor, skips: List[torch.Tensor]) -> List[torch.Tensor]:
        x = bottleneck
        seg_maps: List[torch.Tensor] = []
        for i in range(len(self.seg_ups)):
            skip_stage = len(skips) - 2 - i
            x = self.seg_ups[i](x)
            x = torch.cat([x, skips[skip_stage]], dim=1)
            x = self.seg_dec_blocks[i](x)
            seg_maps.append(self.seg_heads[i](x))
        seg_maps = list(reversed(seg_maps))  # highest-res first
        return seg_maps

    @staticmethod
    def _masked_gap(feat: torch.Tensor, seg_logits_hr: torch.Tensor) -> torch.Tensor:
        """
        Pancreas-aware masked global average pooling:
        mask = softmax(seg_logits_hr)[classes>0].sum(dim=1)  (detach)
        """
        with torch.no_grad():
            probs = torch.softmax(seg_logits_hr, dim=1)
            if probs.shape[1] >= 3:
                # use all non-background classes as FG
                fg = probs[:, 1:, ...].sum(dim=1, keepdim=True)
            else:
                # binary case: FG = 1 - P(background)
                fg = 1.0 - probs[:, 0:1, ...]                     
            # Resize mask to feature spatial size
            fg = F.interpolate(fg, size=feat.shape[2:], mode='trilinear', align_corners=False)
        num = (feat * fg).sum(dim=(2, 3, 4))                      
        den = fg.sum(dim=(2, 3, 4)).clamp_min(1e-6)               
        gap = num / den                                           
        return gap

    def _ensure_cls_head(self, in_ch: int):
        """
        (Re)build classification head if the bottleneck channel count changes
        or hasn't been set yet. Keeps things robust across architectures/plans.
        """
        if not hasattr(self, "_cls_in_ch") or self._cls_in_ch != in_ch:
            self._cls_in_ch = int(in_ch)
            hidden = 256  
            self.cls_head = nn.Sequential(
                nn.Linear(self._cls_in_ch, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(hidden, self.num_classes)
                )
            self.cls_head.train(self.training)

    def _cls_decode(self, bottleneck: torch.Tensor, skips: List[torch.Tensor], seg_maps: List[torch.Tensor]) -> torch.Tensor:
        """
        Produce classification logits from encoder features, masked by the predicted pancreas.
        `seg_maps` may be a list (deep supervision) or a single tensor; we want the **highest-res** logits.
        """
        # Grab the highest-res seg logits
        if isinstance(seg_maps, (list, tuple)):
            seg_logits_fullres = seg_maps[0]
        else:
            seg_logits_fullres = seg_maps

        x = bottleneck  

        # Masked GAP over pancreas foreground
        gap = self._masked_gap(x, seg_logits_fullres)             
        
        # Make sure cls_head matches the actual Cb
        self._ensure_cls_head(gap.shape[1])

        # MLP head -> class logits
        cls_logits = self.cls_head(gap)                           
        return cls_logits

    # api 
    def forward(self, x: torch.Tensor):
        bottleneck, skips = self._encode(x)
        seg_maps = self._seg_decode(bottleneck, skips)
        cls_logits = self._cls_decode(bottleneck, skips, seg_maps)
        if self.training:
            return seg_maps, cls_logits
        else:
            return seg_maps[0]


def UNetResEncMTwoHeads(plans_manager, dataset_json: Dict[str, Any], configuration_manager) -> TwoHeadUNet3D:
    return TwoHeadUNet3D(plans_manager, dataset_json, configuration_manager)
