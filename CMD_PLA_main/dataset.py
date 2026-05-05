import math
import os
import pickle
import warnings
import multiprocessing
from itertools import repeat

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from rdkit import Chem
from rdkit import RDLogger
from scipy.spatial import distance_matrix
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data
from tqdm import tqdm
import networkx as nx
from typing import Optional
from CMD_core.CMD_batch_tensor import CMDTensorRunner

CMD_MOL_CKPT = "molecule.ckpt"
CMD_GPU = 1

lig_runner = CMDTensorRunner(ckpt_path=CMD_MOL_CKPT, gpu=CMD_GPU, task="ligand")

CMD_LIG_ARGS_EVAL  = dict(sde_step=15, inf_step=10, temperature=300, guidance=0.05)
CMD_PROT_ARGS_EVAL = None
ALT_ROUNDS_EVAL = 1

USE_CMD_OFFLINE_DEFAULT = True


RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')

LETTER_TO_NUM = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                 'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                 'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                 'N': 2, 'Y': 18, 'M': 12, 'X': 20}
NUM_TO_LETTER = {v: k for k, v in LETTER_TO_NUM.items()}


@torch.no_grad()
def run_CMD_offline_for_comp_batch(
    comp_list,
    lig_runner: CMDTensorRunner,
    lig_cfg: Optional[dict] = None,
    alt_rounds: int = 1,
):
    assert len(comp_list) > 0, "comp_list 不能为空"
    assert lig_runner is not None, "当前只实现了 ligand，lig_runner 不能为空"
    assert lig_cfg is not None,    "lig_cfg 不能为空"

    device = lig_runner.device

    z_all, x_all, split_all, abid_all = [], [], [], []
    node_ranges = []
    offset = 0

    for b_id, comp in enumerate(comp_list):
        assert hasattr(comp, "pos") and hasattr(comp, "z") and hasattr(comp, "split"), \
            "comp_data 必须含有 pos/z/split"

        pos_i   = comp.pos
        z_i     = comp.z
        split_i = comp.split

        N = pos_i.size(0)
        node_ranges.append((offset, offset + N))

        z_all.append(z_i.to(device))
        x_all.append(pos_i.to(device))
        split_all.append(split_i.to(device))

        abid_all.append(torch.full((N,), b_id, dtype=torch.long, device=device))

        offset += N

    z_cat     = torch.cat(z_all,     dim=0)
    x_cat     = torch.cat(x_all,     dim=0)
    split_cat = torch.cat(split_all, dim=0)
    abid_cat  = torch.cat(abid_all,  dim=0)

    x_new = x_cat.clone()

    lig_all  = (split_cat == 0).long()
    prot_all = (split_cat == 1).long()

    for _ in range(alt_rounds):
        x_new = lig_runner.infer_tensors_batch(
            z_1b_cat=z_cat,
            x_cat=x_new,
            abid_cat=abid_cat,
            sde_step=lig_cfg["sde_step"],
            inf_step=lig_cfg["inf_step"],
            temperature=lig_cfg["temperature"],
            guidance=lig_cfg["guidance"],
            edge_mask=lig_all,
            freeze_mask=prot_all.bool(),
            env_cat=split_cat,
        )

    comp_cmd_list = []
    for (start, end), comp in zip(node_ranges, comp_list):
        pos_i_new = x_new[start:end].cpu()

        comp_i = comp.clone()
        comp_i.pos = pos_i_new
        comp_cmd_list.append(comp_i)

    return comp_cmd_list


@torch.no_grad()
def rebuild_inter_edges_from_pos(
    comp: Data,
    dis_threshold: float = 5.0,
) -> Data:

    pos = comp.pos
    split = comp.split

    device = pos.device
    lig_idx = torch.nonzero(split == 0, as_tuple=False).view(-1)
    prot_idx = torch.nonzero(split == 1, as_tuple=False).view(-1)

    if lig_idx.numel() == 0 or prot_idx.numel() == 0:
        comp.edge_index_inter = torch.zeros((2, 0), dtype=torch.long, device=device)
        return comp

    pos_l = pos[lig_idx]
    pos_p = pos[prot_idx]

    D = torch.cdist(pos_l, pos_p, p=2)
    mask = (D < dis_threshold)
    if not mask.any():
        comp.edge_index_inter = torch.zeros((2, 0), dtype=torch.long, device=device)
        return comp

    li, pi = torch.nonzero(mask, as_tuple=True)

    src = lig_idx[li]
    dst = prot_idx[pi]

    edge_u = torch.cat([src, dst], dim=0)
    edge_v = torch.cat([dst, src], dim=0)

    edge_index_inter = torch.stack([edge_u, edge_v], dim=0)
    comp.edge_index_inter = edge_index_inter.to(device)

    return comp

# 口袋图（GVP）特征工具
def featurize_pocket_graph(protein, name=None, num_pos_emb=16, num_rbf=16, contact_cutoff=8.):
    coords = torch.as_tensor(protein['coords'], dtype=torch.float32)
    seq = torch.as_tensor([LETTER_TO_NUM[a] for a in protein['seq']], dtype=torch.long)
    loaded = torch.load(protein['embed'], weights_only=False)
    lm = loaded['lm_pock_fea']
    seq_emb = lm.float() if isinstance(lm, torch.Tensor) else torch.from_numpy(lm).float()

    mask = torch.isfinite(coords.sum(dim=(1, 2)))
    coords[~mask] = np.inf

    X_ca = coords[:, 1]
    ca_mask = torch.isfinite(X_ca.sum(dim=(1))).float()
    ca_mask_2D = torch.unsqueeze(ca_mask, 0) * torch.unsqueeze(ca_mask, 1)
    dX_ca = torch.unsqueeze(X_ca, 0) - torch.unsqueeze(X_ca, 1)
    D_ca = ca_mask_2D * torch.sqrt(torch.sum(dX_ca ** 2, 2) + 1e-6)
    edge_index = torch.nonzero((D_ca < contact_cutoff) & (ca_mask_2D == 1)).t().contiguous()

    O_feature = _local_frame(X_ca, edge_index)
    pos_embeddings = _positional_embeddings(edge_index, num_embeddings=num_pos_emb)
    E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), D_count=num_rbf)

    dihedrals = _dihedrals(coords)
    orientations = _orientations(X_ca)
    sidechains = _sidechains(coords)

    node_s = dihedrals
    node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
    edge_s = torch.cat([rbf, O_feature, pos_embeddings], dim=-1)
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v))

    data = torch_geometric.data.Data(
        x=X_ca, seq=seq, name=name,
        node_s=node_s, node_v=node_v,
        edge_s=edge_s, edge_v=edge_v,
        edge_index=edge_index, mask=mask,
        seq_emb=seq_emb
    )
    return data


def _dihedrals(X, eps=1e-7):
    X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)
    cosD = torch.sum(n_2 * n_1, -1).clamp(-1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
    D = F.pad(D, [1, 2]).reshape([-1, 3])
    return torch.cat([torch.cos(D), torch.sin(D)], 1)


def _positional_embeddings(edge_index, num_embeddings=None, period_range=[2, 1000]):
    d = edge_index[0] - edge_index[1]
    frequency = torch.exp(torch.arange(0, num_embeddings, 2, dtype=torch.float32)
                          * -(np.log(10000.0) / num_embeddings))
    angles = d.unsqueeze(-1) * frequency
    return torch.cat((torch.cos(angles), torch.sin(angles)), -1)


def _orientations(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def _sidechains(X):
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec


def _normalize(tensor, dim=-1):
    return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    D_mu = torch.linspace(D_min, D_max, D_count, device=device).view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    return torch.exp(-((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2)


def _local_frame(X, edge_index, eps=1e-6):
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)
    o_1 = _normalize(u_2 - u_1, dim=-1)
    O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 1)
    O = F.pad(O, (0, 0, 0, 0, 1, 2), 'constant', 0)
    dX = _normalize(X[edge_index[1]] - X[edge_index[0]], dim=-1)
    dU = torch.bmm(O[edge_index[0]], dX.unsqueeze(2)).squeeze(2)
    R = torch.bmm(O[edge_index[0]].transpose(-1, -2), O[edge_index[1]])
    Q = _quaternions(R)
    return torch.cat((dU, Q), dim=-1)


def _quaternions(R):
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([Rxx - Ryy - Rzz,
                                                             - Rxx + Ryy - Rzz,
                                                             - Rxx - Ryy + Rzz], -1)))
    _R = lambda i, j: R[:, i, j]
    signs = torch.sign(torch.stack([_R(2, 1) - _R(1, 2),
                                    _R(0, 2) - _R(2, 0),
                                    _R(1, 0) - _R(0, 1)], -1))
    xyz = signs * magnitudes
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)
    return Q

# Small-molecule / pocket 图构建
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):
    for atom in mol.GetAtoms():
        results = (
            one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) +
            one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) +
            one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) +
            one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ]) +
            [atom.GetIsAromatic()]
        )
        if explicit_H:
            results += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        graph.add_node(atom.GetIdx(), feats=torch.tensor(np.array(results, dtype=np.float32)))


def get_edges(g):
    e = {}
    for n1, n2, d in g.edges(data=True):
        e_t = [int(d['b_type'] == x) for x in (
            Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC,
            Chem.rdchem.BondType.IONIC, Chem.rdchem.BondType.DATIVE,
            Chem.rdchem.BondType.HYDROGEN, Chem.rdchem.BondType.THREECENTER,
            Chem.rdchem.BondType.DATIVEL, Chem.rdchem.BondType.DATIVER)]
        e_t.append(int(d['IsConjugated'] == False))
        e_t.append(int(d['IsConjugated'] == True))
        e[(n1, n2)] = e_t
    edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
    edge_attr = torch.FloatTensor(list(e.values()))
    return edge_index, edge_attr


def mol2graph(mol):
    g = nx.Graph()
    atom_features(mol, g)
    g = g.to_directed()
    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            e_ij = mol.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                g.add_edge(i, j, b_type=e_ij.GetBondType(), IsConjugated=int(e_ij.GetIsConjugated()))
    x = torch.stack([feats['feats'] for _, feats in g.nodes(data=True)])
    edge_index, edge_attr = get_edges(g)
    return x, edge_index, edge_attr


def inter_graph(ligand, pocket, dis_threshold=5.0):
    atom_num_l = ligand.GetNumAtoms()
    g = nx.Graph()
    pos_l = ligand.GetConformer().GetPositions()
    pos_p = pocket.GetConformer().GetPositions()
    dis_matrix = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix < dis_threshold)
    for i, j in zip(node_idx[0], node_idx[1]):
        g.add_edge(i, j + atom_num_l)
    g = to_directed = g.to_directed()
    if g.number_of_edges() == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v in g.edges(data=False)]).T
    return edge_index_inter


def get_coord(residues):
    residues_coord = []
    for res in residues:
        try:
            N = [round(num, 3) for num in res.child_dict['N'].coord.tolist()]
            CA = [round(num, 3) for num in res.child_dict['CA'].coord.tolist()]
            C = [round(num, 3) for num in res.child_dict['C'].coord.tolist()]
            O = [round(num, 3) for num in res.child_dict['O'].coord.tolist()]
            res_coor = (N, CA, C, O)
        except Exception:
            print(res)
        residues_coord.append(res_coor)
    return residues_coord


def get_sequence(structure):
    residues_list = [residue for residue in structure.get_residues() if residue.get_id()[0] == ' ']
    residues = [res for res in residues_list if {'N', 'CA', 'C', 'O'}.issubset(res.child_dict.keys())]
    seqs = ''.join([seq1(res.get_resname()) for res in residues])
    coords = get_coord(residues)
    return residues, seqs, coords


def process_pock(prot_name, pock_file, pock_esm):
    parser = PDBParser()
    pocket = parser.get_structure(prot_name, pock_file)
    pock_res, pock_seq, pock_coords = get_sequence(pocket)
    pock_dic = {'name': prot_name, 'seq': pock_seq, 'coords': pock_coords, 'embed': pock_esm}
    return pock_dic


# ====== 严格路径解析（ESM 特征文件仍需路径） ======
def _resolve_pock_esm_path_strict(data_dir: str, cid: str) -> str:
    p = os.path.join(data_dir, "pock_5A_fea", f"{cid}_pock.pt")
    if not os.path.exists(p):
        raise FileNotFoundError(f"missing pocket ESM: {p}")
    return p


def mols2graphs(
    complex_path, label, save_path, cid, pock_path, pock_esm_path, lig_feat_path,
    dis_threshold=5.,
):

    with open(complex_path, 'rb') as f:
        ligand, pocket = pickle.load(f)

    pocket_graph_mol = Chem.RemoveHs(pocket)
    if pocket_graph_mol.GetNumAtoms() == 0:
        raise ValueError(f"{cid}: pocket_graph_mol has 0 atoms after RemoveHs.")

    pos_l = torch.from_numpy(np.asarray(ligand.GetConformer().GetPositions(), dtype=np.float32))
    pos_p = torch.from_numpy(np.asarray(pocket_graph_mol.GetConformer().GetPositions(), dtype=np.float32))

    z_l = torch.tensor([a.GetAtomicNum() for a in ligand.GetAtoms()], dtype=torch.long)
    z_p = torch.tensor([a.GetAtomicNum() for a in pocket_graph_mol.GetAtoms()], dtype=torch.long)

    x_l, edge_index_l, edge_attr_l = mol2graph(ligand)
    x_p, edge_index_p, edge_attr_p = mol2graph(pocket_graph_mol)

    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket_graph_mol.GetNumAtoms()
    edge_index_intra = torch.cat([edge_index_l, edge_index_p + atom_num_l], dim=-1)
    edge_index_inter = inter_graph(ligand, pocket_graph_mol, dis_threshold=dis_threshold)
    y = torch.tensor([label], dtype=torch.float32)
    pos = torch.cat([pos_l, pos_p], dim=0)
    split = torch.cat([
        torch.zeros((atom_num_l,), dtype=torch.long),
        torch.ones((atom_num_p,), dtype=torch.long),
    ], dim=0)

    pock_dic = process_pock(cid, pock_path, pock_esm_path)
    pock_new_fea = featurize_pocket_graph(pock_dic, cid, num_pos_emb=16, num_rbf=16, contact_cutoff=8.)

    drug_data = Data(
        x=x_l, edge_index=edge_index_l, edge_index_inter=edge_index_inter,
        edge_attr=edge_attr_l, pos=pos_l, y=y
    )
    drug_data.z = z_l

    pock_data = Data(
        x=x_p, edge_index=edge_index_p, edge_index_inter=edge_index_inter,
        edge_attr=edge_attr_p, pos=pos_p, y=y
    )
    pock_data.z = z_p

    comp_data = Data(
        x=torch.cat([x_l, x_p], dim=0),
        edge_index_intra=edge_index_intra,
        edge_index_inter=edge_index_inter,
        y=y, pos=pos, split=split
    )
    comp_data.z = torch.cat([z_l, z_p], dim=0)

    lig_emb = None
    if lig_feat_path is not None and os.path.exists(lig_feat_path):
        try:
            lig_loaded = torch.load(lig_feat_path, weights_only=False)
            lig_vec = lig_loaded.get("lig_mol2vec", None)
            if lig_vec is not None:
                lig_emb = torch.as_tensor(lig_vec, dtype=torch.float32)
        except Exception as e:
            print(f"[WARN] {cid} 加载配体特征失败: {e}")

    if lig_emb is not None:
        drug_data.lig_emb = lig_emb
        comp_data.lig_emb = lig_emb

    data = {'drug': drug_data, 'pock': pock_data, 'pock_new': pock_new_fea, 'comp': comp_data}
    torch.save(data, save_path)

# Dataset / DataLoader
class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


class GraphDataset(Dataset):
    def __init__(
            self,
            data_dir,
            data_df,
            dis_threshold=5,
            graph_type='Graph_HG',
            num_process=8,
            create=False,
            use_cmd_offline: bool = False,
            lig_runner: CMDTensorRunner = None,
            prot_runner: CMDTensorRunner = None,
            CMD_lig_args: dict = None,
            CMD_prot_args: dict = None,
            alt_rounds_CMD: int = 1,
            CMD_batch_size: int = 128,
    ):
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.graph_type = graph_type
        self.create = create
        self.graph_paths = None
        self.complex_ids = None
        self.num_process = num_process

        self.use_cmd_offline = use_cmd_offline
        self.lig_runner = lig_runner
        self.prot_runner = prot_runner
        self.CMD_lig_args = CMD_lig_args or {}
        self.CMD_prot_args = CMD_prot_args or {}
        self.alt_rounds_CMD = alt_rounds_CMD
        self.CMD_batch_size = CMD_batch_size

        self._pre_process()

    def _pre_process(self):
        data_dir = self.data_dir
        data_df = self.data_df
        dis_thresholds = repeat(self.dis_threshold, len(data_df))

        complex_path_list, pKa_list, graph_path_list = [], [], []
        cid_list, pock_path_list, pock_esm_path_list = [], [], []
        lig_feat_path_list = []

        for _, row in tqdm(data_df.iterrows(), total=len(data_df), ncols=80):
            cid, pKa = str(row['id']), float(row['affinity'])
            complex_dir = os.path.join(data_dir, cid)

            complex_path = os.path.join(complex_dir, f"{cid}_{self.dis_threshold}A.rdkit")
            graph_path = os.path.join(complex_dir, f"{cid}_fea.pyg")
            pock_path = os.path.join(complex_dir, "Pocket_5A.pdb")
            lig_feat_path = os.path.join(complex_dir, f"{cid}_lig.pt")

            try:
                pock_esm_path = _resolve_pock_esm_path_strict(data_dir, cid)
            except FileNotFoundError:
                if not os.path.exists(complex_path):
                    print(f"[MISS] {cid}: {complex_path}")
                if not os.path.exists(pock_path):
                    print(f"[MISS] {cid}: {pock_path}")
                print(f"[MISS] {cid}: {os.path.join(data_dir, 'pock_5A_fea', f'{cid}_pock.pt')}")
                continue

            have_inputs = (
                    os.path.exists(complex_path)
                    and os.path.exists(pock_path)
                    and os.path.exists(pock_esm_path)
                    and os.path.exists(lig_feat_path)
            )

            if self.create:
                if have_inputs:
                    complex_path_list.append(complex_path)
                    pKa_list.append(pKa)
                    graph_path_list.append(graph_path)
                    cid_list.append(cid)
                    pock_path_list.append(pock_path)
                    pock_esm_path_list.append(pock_esm_path)
                    lig_feat_path_list.append(lig_feat_path)
            else:
                if have_inputs and os.path.exists(graph_path):
                    complex_path_list.append(complex_path)
                    pKa_list.append(pKa)
                    graph_path_list.append(graph_path)
                    cid_list.append(cid)
                    pock_path_list.append(pock_path)
                    pock_esm_path_list.append(pock_esm_path)
                    lig_feat_path_list.append(lig_feat_path)

        if self.create and complex_path_list:
            print('Generate complex graph...')
            # ====== 阶段 A：只构图，不跑 CMD ======
            if self.use_cmd_offline:
                print("[Stage A] build graphs without CMD ...")
                for complex_path, pKa, graph_path, cid, pock_path, pock_esm_path, lig_feat_path, dis in zip(
                        complex_path_list,
                        pKa_list,
                        graph_path_list,
                        cid_list,
                        pock_path_list,
                        pock_esm_path_list,
                        lig_feat_path_list,
                        dis_thresholds
                ):
                    mols2graphs(
                        complex_path, pKa, graph_path, cid,
                        pock_path, pock_esm_path, lig_feat_path,
                        dis_threshold=dis,
                    )
            else:
                pool = multiprocessing.Pool(self.num_process)
                try:
                    pool.starmap(
                        mols2graphs,
                        zip(complex_path_list,
                            pKa_list,
                            graph_path_list,
                            cid_list,
                            pock_path_list,
                            pock_esm_path_list,
                            lig_feat_path_list,
                            dis_thresholds)
                    )
                finally:
                    pool.close()
                    pool.join()

            # ====== 阶段 B：按 batch 用 CMD 精修坐标 ======
            if self.create and self.use_cmd_offline:
                if self.lig_runner is None:
                    print("[WARN] use_cmd_offline=True 但 lig_runner 为空，跳过 CMD。")
                else:
                    print("[Stage B] CMD batched refinement (ligand-only) ...")
                    B = self.CMD_batch_size

                    num_batches = (len(graph_path_list) + B - 1) // B

                    for bi in tqdm(range(num_batches), desc=f"CMD refinement (B={B})"):
                        start = bi * B
                        paths_batch = graph_path_list[start:start + B]

                        data_batch = []
                        comp_batch = []
                        for p in paths_batch:
                            d = torch.load(p)
                            data_batch.append(d)
                            comp_batch.append(d['comp'])  

                        comp_cmd_list = run_CMD_offline_for_comp_batch(
                            comp_batch,
                            lig_runner=self.lig_runner,
                            lig_cfg=self.CMD_lig_args,
                            alt_rounds=self.alt_rounds_CMD,
                        )

                        for p, d, cu in zip(paths_batch, data_batch, comp_cmd_list):
                            cu = rebuild_inter_edges_from_pos(
                                cu,
                                dis_threshold=self.dis_threshold,
                            )

                            d['comp_cmd'] = cu
                            torch.save(d, p)

        self.graph_paths = [p for p in graph_path_list if os.path.exists(p)]
        print(f"done, total {len(self.graph_paths)} graphs.")
        if len(self.graph_paths) == 0:
            if self.create:
                print("[WARN] No graphs were generated. Check missing files above.")
            else:
                print("[WARN] No prebuilt graphs found. Set create=True to build.")

    def __getitem__(self, idx):
        return torch.load(self.graph_paths[idx])

    def collate_fn(self, batch):
        drug_batch = Batch.from_data_list([item['drug'] for item in batch])
        pock_batch = Batch.from_data_list([item['pock'] for item in batch])

        comp_list = []
        for item in batch:
            if 'comp_cmd' in item:
                comp_list.append(item['comp_cmd'])
            else:
                comp_list.append(item['comp'])
        comp_batch = Batch.from_data_list(comp_list)

        pock_new_batch = Batch.from_data_list([item['pock_new'] for item in batch])

        return drug_batch, pock_batch, comp_batch, pock_new_batch

    def __len__(self):
        return len(self.graph_paths)


if __name__ == '__main__':
    data_root = "data"
    toy_dir = os.path.join(data_root, 'toy_set')
    toy_df = pd.read_csv(os.path.join(data_root, "toy_examples.csv"))

    toy_set = GraphDataset(
        toy_dir,
        toy_df,
        graph_type='Graph_HG',
        dis_threshold=5,
        create=True,
        use_cmd_offline=True,
        lig_runner=lig_runner,
        CMD_lig_args=CMD_LIG_ARGS_EVAL,
        alt_rounds_CMD=ALT_ROUNDS_EVAL,
        CMD_batch_size=200,
    )
