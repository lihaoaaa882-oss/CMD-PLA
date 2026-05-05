import os, sys
from typing import Optional
import torch
from tqdm.auto import trange

UNI_ROOT = os.path.dirname(__file__)
if UNI_ROOT not in sys.path:
    sys.path.insert(0, UNI_ROOT)

from utils.bio_utils import ATOM_TYPE, BIO_ATOM_TYPE

def _z_to_idx(z_1b: torch.Tensor) -> torch.Tensor:

    if (z_1b <= 0).any():
        bad = z_1b[z_1b <= 0].unique().tolist()
        raise RuntimeError(f"[CMD] invalid atomic numbers (<=0): {bad}")
    return z_1b.long() - 1

_SUPPORTED_ATIDX = torch.tensor(
    [ATOM_TYPE.index(sym) for sym in BIO_ATOM_TYPE],
    dtype=torch.long
)

_COLLISION_THRESHOLD = 1.6

_MAX_LIG_GLOBAL_RMSD = 0.6

def _mask_supported(atype_idx: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, "isin"):
        return torch.isin(atype_idx, _SUPPORTED_ATIDX.to(atype_idx.device))
    mask = torch.zeros_like(atype_idx, dtype=torch.bool)
    for v in _SUPPORTED_ATIDX:
        mask |= (atype_idx == v.to(atype_idx.device))
    return mask


class CMDTensorRunner:
    def __init__(self, ckpt_path: str, gpu: int = -1, task: str = "ligand", map_location: Optional[str] = None):
        if map_location is None:
            if gpu is None or gpu < 0 or not torch.cuda.is_available():
                map_location = "cpu"
                self.device = torch.device("cpu")
            else:
                map_location = f"cuda:{gpu}"
                self.device = torch.device(map_location)
        else:
            self.device = torch.device(map_location)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[CMDTensorRunner] ckpt not found: {ckpt_path}")

        print(f"[CMDTensorRunner] loading ckpt from {ckpt_path} on {self.device} (task={task}) ...")

        model = torch.load(ckpt_path, map_location="cpu")
        if hasattr(model, "to"):
            model = model.to(self.device)
        model.eval()
        self.model = model
        self.task = task

    @torch.no_grad()
    def infer_tensors_batch(
            self,
            z_1b_cat: torch.Tensor,
            x_cat: torch.Tensor,
            abid_cat: torch.Tensor,
            *,
            sde_step: int,
            inf_step: int,
            temperature: float,
            guidance: float,
            env_cat: torch.Tensor,
            edge_mask: Optional[torch.Tensor] = None,
            freeze_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if z_1b_cat.numel() == 0:
            print("空批直接返回")
            return x_cat

        out_device = x_cat.device

        z_1b = z_1b_cat.to(self.device, non_blocking=True)
        x0 = x_cat.to(self.device, non_blocking=True)  # Å
        abid = abid_cat.to(self.device, non_blocking=True)

        atype_all = _z_to_idx(z_1b)

        keep = _mask_supported(atype_all)
        if not keep.any():
            print("没有 BIO 支持的原子")
            return x_cat

        atype = atype_all[keep]
        xt = x0[keep].clone()
        abid_ = abid[keep]

        if edge_mask is not None:
            edge_mask_dev = edge_mask.to(self.device, non_blocking=True).long()[keep]
        else:
            edge_mask_dev = torch.ones_like(atype, dtype=torch.long, device=self.device)

        if freeze_mask is not None:
            freeze_mask_dev = freeze_mask.to(self.device, non_blocking=True).bool()[keep]
        else:
            freeze_mask_dev = None

        env_full = env_cat.to(self.device, non_blocking=True).long()
        env_geom = env_full[keep]
        env_vec = env_geom.clone()

        num_env = 1
        for k, v in self.model.state_dict().items():
            if ("env" in k or "domain" in k) and v.dim() == 2:
                num_env = v.shape[0]
                break
        if num_env > 1:
            env_vec = env_vec.clamp_(0, num_env - 1)
        else:
            env_vec = torch.zeros_like(env_vec, dtype=torch.long, device=self.device)

        uniq, inv, counts = torch.unique(abid_, return_inverse=True, return_counts=True)
        centers = torch.zeros((uniq.numel(), 3), device=self.device, dtype=xt.dtype)
        centers.index_add_(0, inv, xt)
        centers = centers / counts.unsqueeze(-1)
        xt = xt - centers[inv]

        xt_init = xt.clone()  # (ΣN_keep, 3)

        if freeze_mask_dev is not None:
            is_frozen = freeze_mask_dev
        else:
            is_frozen = torch.zeros_like(env_geom, dtype=torch.bool, device=self.device)

        for step_idx in trange(inf_step, desc=f"{self.task} inf_step", leave=False):
            xt_prev = xt

            batch = {
                "x0": xt_prev,
                "atype": atype,
                "abid": abid_,
                "edge_mask": edge_mask_dev,
                "env": env_vec,
            }

            xt_next = self.model.sde(batch, sde_step=sde_step, temp=temperature, guidance=guidance)


            if freeze_mask_dev is not None and freeze_mask_dev.any():
                xt_next[freeze_mask_dev] = xt_prev[freeze_mask_dev]

            if not torch.isfinite(xt_next).all():
                xt_next = xt_prev

            xt_candidate = xt_next.clone()

            for b in uniq:
                comp_mask = (abid_ == b)

                lig_mask = comp_mask & (env_geom == 0) & (~is_frozen)
                prot_mask = comp_mask & (env_geom == 1)

                if lig_mask.sum() == 0:
                    continue

                lig_init = xt_init[lig_mask]
                lig_cand = xt_candidate[lig_mask]

                disp_global = lig_cand - lig_init
                if disp_global.numel() > 0:
                    global_rmsd = torch.sqrt(torch.mean((disp_global ** 2).sum(dim=-1)))
                else:
                    global_rmsd = torch.tensor(0.0, device=self.device)

                too_far = (global_rmsd > _MAX_LIG_GLOBAL_RMSD)

                if prot_mask.sum() > 0:
                    prot_cand = xt_candidate[prot_mask]
                    dist_lp = torch.cdist(lig_cand, prot_cand, p=2)  # (nL, nP)
                    min_dist = dist_lp.min()
                    collision = (min_dist < _COLLISION_THRESHOLD)
                else:
                    collision = torch.tensor(False, device=self.device)

                if too_far or collision:
                    xt_candidate[comp_mask] = xt_prev[comp_mask]

            xt = xt_candidate

        xt = xt + centers[inv]

        x_out = x0.clone()
        x_out[keep] = xt

        return x_out.to(out_device, non_blocking=True)
