# nnUNet\nnunetv2\training\nnUNetTrainer\nnUNetTrainer_TwoHeads.py
import os
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
import torch.nn.functional as F

from importlib import import_module
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

class nnUNetTrainer_TwoHeads(nnUNetTrainer):
    """
    ResEnc-M with an extra classification decoder head.
    - Seg loss: Dice+CE (nnU-Net default)
    - Cls loss: class-balanced CE with label smoothing (epsilon=0.05)
    - Loss weights: ramp λ_cls 0.1 -> 0.3 over first 10 epochs, keep λ_seg=1.0 then settle λ_seg=0.7, λ_cls=0.3
    """

    # nnU-Net hooks
    def build_network(self):
        arch = import_module('nnunetv2.dynamic_network_architectures.architectures.unet_with_two_decoders')
        TwoHead = getattr(arch, 'UNetResEncMTwoHeads', None) or getattr(arch, 'TwoHeadUNet3D')
        # Log exactly which class/file is being used
        try:
            self.print_to_log_file(f"[CLS] build_network -> using {TwoHead.__name__} from {arch.__file__}")
        except Exception:
            pass
        return TwoHead(self.plans_manager, self.dataset_json, self.configuration_manager)

    def initialize(self):
        super().initialize()
        
        self._orig_loss = self.loss
        self.loss = self._twoheads_loss_wrapper
        
        if not hasattr(self.network, "cls_head"):
            arch = import_module('nnunetv2.dynamic_network_architectures.architectures.unet_with_two_decoders')
            TwoHead = getattr(arch, 'UNetResEncMTwoHeads', None) or getattr(arch, 'TwoHeadUNet3D')
            new_net = TwoHead(self.plans_manager, self.dataset_json, self.configuration_manager).to(self.device)
            try:
                if getattr(self, "is_compiled", False) and hasattr(torch, "compile"):
                   new_net = torch.compile(new_net)
            except Exception:
                pass
            self.network = new_net
            self._rebind_optimizer_to_new_network()

        try:
            self.print_to_log_file(f"[CLS] after initialize: net={self.network.__class__.__name__}, "
                                   f"has_cls_head={hasattr(self.network, 'cls_head')}")
        except Exception:
            pass
        assert hasattr(self.network, 'cls_head'), "Two-head model not active after initialize!"

        # training length & checkpointing 
        self.max_num_epochs = 300
        self.save_every = 10                
        self.disable_checkpointing = False
        self.save_best_checkpoint = True
        self.save_final_checkpoint = True

        # locate dataset root
        start = Path(self.preprocessed_dataset_folder)
        ds_root = None
        probe = start
        for _ in range(6):
            if (probe / "splits_final.json").is_file():
                ds_root = probe
                break
            probe = probe.parent
        if ds_root is None:
            raise FileNotFoundError(f"Could not find splits_final.json starting at {start}")

        splits = json.loads((ds_root / "splits_final.json").read_text())
        split_idx = self.fold if isinstance(self.fold, int) and 0 <= self.fold < len(splits) else 0
        split = splits[split_idx]
        self.train_ids = set(split["train"])
        self.val_ids = set(split["val"])

        # subtypes (0/1/2) for classification
        self.caseid_to_subtype: Dict[str, int] = {}
        st_path = ds_root / "subtypes.json"
        if st_path.is_file():
            self.caseid_to_subtype = json.loads(st_path.read_text())
        else:
            self.print_to_log_file(f"[WARN] subtypes.json not found at {st_path}. Classification metrics disabled.")

        # class weights (inverse freq on TRAIN only)
        counts = [0, 0, 0]
        for cid in self.train_ids:
            if cid in self.caseid_to_subtype:
                c = int(self.caseid_to_subtype[cid])
                if 0 <= c <= 2:
                    counts[c] += 1
        tot = sum(counts) if sum(counts) > 0 else 1
        # avoid zeros
        freq = [(c if c > 0 else 1) / tot for c in counts]
        inv = [1.0 / f for f in freq]
        w = torch.tensor(inv, dtype=torch.float32, device=self.device)
        w = w / w.sum() * 3.0  # normalize around 1
        self.cls_weights = w
        self.print_to_log_file(f"[CLS] class weights (train only): {self.cls_weights.tolist()}")

        # loss weights (will ramp in training loop early on) 
        self.lambda_seg = 1.0
        self.lambda_cls = 0.10  

    # core training 
    def _cls_loss_label_smoothed_weighted(self, logits: torch.Tensor, target: torch.Tensor, eps: float = 0.05):
        """
        Weighted label-smoothed CE:
        - logits: [B, C]
        - target: [B] long
        """
        num_classes = logits.shape[1]
        logp = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logp)
            true_dist.fill_(eps / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - eps)
        per_sample = -(true_dist * logp).sum(dim=1)
        sample_w = self.cls_weights[target]
        return (sample_w * per_sample).mean()

    def _compute_losses_trainbatch(self, data_dict, seg_outputs: List[torch.Tensor], cls_logits: torch.Tensor):
        device = seg_outputs[0].device if isinstance(seg_outputs, (list, tuple)) else seg_outputs.device
        target = data_dict['target']
        if isinstance(target, (list, tuple)):
            target = [t.to(device, non_blocking=True) for t in target]
        else:
            target = target.to(device, non_blocking=True)
            
        seg_loss = self._orig_loss(seg_outputs, target)
        cls_loss = torch.zeros(1, device=seg_outputs[0].device)

        keys = data_dict.get('keys', None)
        if keys is not None and getattr(self, "caseid_to_subtype", None):
            labels = []
            ok = True
            for k in keys:
                cid = Path(k).stem.replace("_0000", "").replace(".nii", "")
                if (cid in self.caseid_to_subtype) and (cid in self.train_ids):
                    labels.append(int(self.caseid_to_subtype[cid]))
                else:
                    ok = False
                    break
            if ok and len(labels) == cls_logits.shape[0]:
                target = torch.as_tensor(labels, dtype=torch.long, device=cls_logits.device)
                cls_loss = self._cls_loss_label_smoothed_weighted(cls_logits, target, eps=0.05)

        total = self.lambda_seg * seg_loss + self.lambda_cls * cls_loss
        return total, seg_loss.detach(), cls_loss.detach()

    def train_step(self, batch):
        # simple ramp for λ_cls over first 5 epochs; then settle λ_seg=0.6, λ_cls=0.4
        if self.current_epoch < 5:
            frac = (self.current_epoch + 1) / 5.0
            self.lambda_seg = 1.0
            self.lambda_cls = 0.10 + 0.30 * frac
        else:
            self.lambda_seg = 0.6
            self.lambda_cls = 0.4

        data = batch['data'].to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)

        seg_outputs, cls_logits = self.network(data)
        loss, seg_loss, cls_loss = self._compute_losses_trainbatch(batch, seg_outputs, cls_logits)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12.0)
        self.optimizer.step()
        
        return {
            'loss': float(loss.detach().cpu().item()),
            'seg_loss': float(seg_loss.detach().cpu().item()),
            'cls_loss': float(cls_loss.detach().cpu().item()),
            }
    
    def validation_step(self, batch):
        """
        Segmentation-only validation compatible with nnU-Net's collator:
            - eval() -> model returns seg logits [B,C,...]
            - targets normalized to [B,1,...] on device
            - use original deep-supervision loss
            - return collatable types and hard counts: tp_hard/fp_hard/fn_hard
        """
        self.network.eval()
        data = batch['data'].to(self.device, non_blocking=True)

        with torch.no_grad():
            seg_logits = self.network(data)
            
        target = batch['target']
        if isinstance(target, (list, tuple)):
            target = target[0]                      # highest-res target
        target = target.to(seg_logits.device, non_blocking=True)
        if target.ndim == seg_logits.ndim - 1:
            target = target.unsqueeze(1)           # [B,1,...]

        # loss: original deep-supervision expects lists 
        seg_loss = self._orig_loss([seg_logits], [target])

        probs = softmax_helper_dim1(seg_logits)          
        hard = probs.argmax(1, keepdim=True)             
        pred_onehot = torch.zeros_like(probs)
        pred_onehot.scatter_(1, hard, 1)
        axes = [0] + list(range(2, seg_logits.ndim))     
        tp_h, fp_h, fn_h, _ = get_tp_fp_fn_tn(pred_onehot, target, axes=axes)

        # Return only collatable types
        return {
            'loss'    : float(seg_loss.detach().cpu().item()),
            'tp_hard' : tp_h.detach().cpu().numpy(),
            'fp_hard' : fp_h.detach().cpu().numpy(),
            'fn_hard' : fn_h.detach().cpu().numpy(),
            }

    def on_train_start(self):
        super().on_train_start()
        # Prove the architecture is correct or crash loudly
        self.print_to_log_file(f"[CLS] model class: {self.network.__class__.__name__}")
        assert hasattr(self.network, "cls_head"), "Two-head model not active: cls_head missing!"
        self.print_to_log_file(f"[CLS] model has cls_head: {hasattr(self.network, 'cls_head')}")
    
    def _twoheads_loss_wrapper(self, output, target):
        """
        Accepts output either as (seg_maps, cls_logits), list[Tensor], or single Tensor.
        Always convert to list[Tensor] for the base deep-supervision loss and
        move targets to the same device. Optionally add classification CE.
        """
        # Unpack (training path) or keep as-is (validation path)
        seg_maps, cls_logits = output, None
        if isinstance(output, (tuple, list)) and len(output) == 2:
            seg_maps, cls_logits = output

        if isinstance(seg_maps, torch.Tensor):
            seg_maps = [seg_maps]
        elif isinstance(seg_maps, tuple):
            seg_maps = list(seg_maps)
        dev = seg_maps[0].device

        if isinstance(target, (list, tuple)):
            target = [t.to(dev, non_blocking=True) for t in target]
        else:
            target = [target.to(dev, non_blocking=True)]

        seg_loss = self._orig_loss(seg_maps, target)

        cls_loss = torch.zeros((), device=dev)
        if cls_logits is not None and isinstance(target, list):
            pass

        lam = getattr(self, "lambda_cls", 0.0)
        total = seg_loss + lam * cls_loss
        try:
            if getattr(self, "iteration", 0) % 200 == 0:
                self.print_to_log_file(
                    f"[CLS] seg_loss={float(seg_loss):.4f} cls_loss={float(cls_loss):.4f} lam={lam:.2f}"
                    )
        except Exception:
            pass
        return total
    
    def _rebind_optimizer_to_new_network(self):
        """
        Some nnU-Net versions don't expose initialize_optimizer_and_scheduler().
        Rebind existing optimizer param groups to the new network parameters.
        """
        opt = getattr(self, 'optimizer', None)
        if opt is None:
            return
        params = list(self.network.parameters())
        if len(opt.param_groups) == 0:
            opt.add_param_group({'params': params})
        else:
            opt.param_groups[0]['params'] = params
            for g in opt.param_groups[1:]:
                g['params'] = []
        try:
            self.print_to_log_file("[CLS] optimizer param groups rebound to new network")
        except Exception:
            pass

    # validation (seg: base; cls: macro-F1 summary only) 
    def validate(self, do_mirroring: bool = True):
        seg_val_out = super().validate(do_mirroring)
        return seg_val_out
