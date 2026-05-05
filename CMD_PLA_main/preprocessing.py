# %%
import os
import warnings
from rdkit import Chem
import torch
import pickle
import csv
import esm
import pandas as pd
from tqdm import tqdm
import pymol
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings("ignore")
# %%


def generate_pocket(data_dir, distance):
    #complex_id = os.listdir(data_dir)
    all_subitems = os.listdir(data_dir)
    candidate_dirs = [item for item in all_subitems if os.path.isdir(os.path.join(data_dir, item))]
    valid_complex_ids = []
    for item in candidate_dirs:
        # 拼接当前目录下的蛋白质和配体文件路径
        protein_file = os.path.join(data_dir, item, f"{item}_protein.pdb")
        ligand_file = os.path.join(data_dir, item, f"{item}_ligand.mol2")
        # 校验两个文件是否都存在
        if os.path.exists(protein_file) and os.path.exists(ligand_file):
            valid_complex_ids.append(item)
        else:
            # 可选：打印跳过原因，便于排查
            print(
                f"跳过非复合物目录: {item} (缺少 {protein_file.split(os.sep)[-1]} 或 {ligand_file.split(os.sep)[-1]})")

    for cid in tqdm(valid_complex_ids, ncols=50):
        complex_dir = os.path.join(data_dir, cid)
        lig_native_path = os.path.join(complex_dir, f"{cid}_ligand.mol2")
        protein_path = os.path.join(complex_dir, f"{cid}_protein.pdb")
        if os.path.exists(os.path.join(complex_dir, f'Pocket_{distance}A.pdb')):
            continue

        pymol.cmd.load(protein_path)
        pymol.cmd.remove('resn HOH')
        pymol.cmd.load(lig_native_path)
        pymol.cmd.remove('hydrogens')
        try:
            pymol.cmd.select('Pocket', f'byres {cid}_ligand around {distance}')
            pymol.cmd.save(os.path.join(complex_dir, f'Pocket_{distance}A.pdb'), 'Pocket')
        except Exception:
            count = count + 1
            print(cid, f'{count} error')
        pymol.cmd.delete('all')

def generate_complex(data_dir, data_df, distance=5, input_ligand_format='mol2'):
    un = []
    pbar = tqdm(total=len(data_df), ncols=50)
    for i, row in data_df.iterrows():
        cid, pKa = row['id'], float(row['affinity'])
        pocket_path = os.path.join(data_dir, cid, f'Pocket_{distance}A.pdb')
        if not os.path.exists(os.path.join(f"{data_dir}/{cid}/{cid}_{distance}A.rdkit")):
            try:
                if input_ligand_format != 'pdb':
                    ligand_input_path = os.path.join(data_dir, cid, f'{cid}_ligand.{input_ligand_format}')
                    ligand_path = ligand_input_path.replace(f".{input_ligand_format}", ".pdb")
                    os.system(f'obabel {ligand_input_path} -O {ligand_path} -d')
                else:
                    ligand_path = os.path.join(data_dir, cid, f'{cid}_ligand.pdb')

                save_path = os.path.join(f"{data_dir}/{cid}/{cid}_{distance}A.rdkit")
                ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)
                if ligand == None:
                    print(f"Unable to process ligand of {cid}")
                    continue

                pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)
                if pocket == None:
                    print(f"Unable to process protein of {cid}")
                    continue
            except:
                un.append(cid)
            complex = (ligand, pocket)
            with open(save_path, 'wb') as f:
                pickle.dump(complex, f)

            pbar.update(1)
    print(un, 'Unprocessed')

def get_Ca_coord(residues):
    residues_coord = []
    for res in residues:
        try:
            N_coord = [round(num, 3) for num in res.child_dict['N'].coord.tolist()]
            CA_coord = [round(num, 3) for num in res.child_dict['CA'].coord.tolist()]
            C_coord = [round(num, 3) for num in res.child_dict['C'].coord.tolist()]
            O_coord = [round(num, 3) for num in res.child_dict['O'].coord.tolist()]
            res_coor = [N_coord, CA_coord, C_coord, O_coord]
        except:
            print(res)
        residues_coord.append(res_coor)
    return residues_coord

def get_sequence(structure):
    residues_list = [residue for residue in structure.get_residues() if residue.get_id()[0] == ' ']

    residues = []
    for res in residues_list:
        if 'N' in res.child_dict.keys() and 'CA' in res.child_dict.keys() and 'C' in res.child_dict.keys() and 'O' in res.child_dict.keys():
            residues.append(res)
    seqs = ''.join([seq1(residue.get_resname()) for residue in residues])
    coords = get_Ca_coord(residues)
    return residues, seqs, coords

def generate_mask(seq_res, sequence, pocket_res):
    mask = len(sequence) * [0]
    pock_f_id = [res_i.full_id for res_i in pocket_res]
    prot_f_id = [res_j.full_id for res_j in seq_res]
    for res_i in pocket_res:
        for res_j in seq_res:
            if res_i.full_id == res_j.full_id:
                index = seq_res.index(res_j)
                mask[index] = 1
    ## valid num
    idx_1_list = []
    for i, ele in enumerate(mask):
        if ele == 1:
            idx_1_list.append(i)
    for id_i in pock_f_id:
        p_f_id = [prot_f_id[id_j] for id_j in idx_1_list]
        if id_i in p_f_id:
            new_mask = mask
        else:
            print(sequence, 'unmatched')
    return mask

def extract_pos(prot_fea, pock_pos):
    indices = torch.nonzero(torch.tensor(pock_pos), as_tuple=True)
    prot_fea_tensor = torch.tensor(prot_fea)
    pock_fea = prot_fea_tensor[indices[0]].numpy()
    return pock_fea

def generate_pock_esm(datasets_path, esm_dir, distance):
    pock_fea_dir = os.path.join(esm_dir, f'pock_{distance}A_fea')
    os.makedirs(pock_fea_dir, exist_ok=True)

    set_list = os.listdir(datasets_path)
    for prot_name in tqdm(set_list, ncols=50, desc="生成口袋ESM特征"):
        prot_file = os.path.join(datasets_path, prot_name, f'{prot_name}_protein.pdb')
        pock_file = os.path.join(datasets_path, prot_name, f'Pocket_{distance}A.pdb')

        prot_fea_path = os.path.join(datasets_path, prot_name, f'{prot_name}.pt')

        if not os.path.exists(prot_fea_path):
            print(f"全蛋白特征不存在：{prot_fea_path}，跳过")
            continue

        pock_fea_path = os.path.join(pock_fea_dir, f'{prot_name}_pock.pt')
        if os.path.exists(pock_fea_path):
            continue

        try:
            parser = PDBParser()
            protein = parser.get_structure(prot_name, prot_file)
            pocket = parser.get_structure(prot_name, pock_file)

            prot_res, prot_seq, prot_coords = get_sequence(protein)
            pock_res, pock_seq, pock_coords = get_sequence(pocket)
            pock_pos = generate_mask(prot_res, prot_seq, pock_res)

            with open(prot_fea_path, 'rb') as f:
                prot_fea = torch.load(f)['lm_prot_fea']
            pock_fea = extract_pos(prot_fea, pock_pos)
            torch.save({'lm_pock_fea': pock_fea}, pock_fea_path)
        except Exception as e:
            print(f'{prot_name} 错误: {str(e)}')

def extract_protein_sequences(data_dir):
    parser = PDBParser()
    protein_sequences = {}

    for cid in tqdm(os.listdir(data_dir), desc="提取蛋白质序列"):
        cid_dir = os.path.join(data_dir, cid)
        pdb_path = os.path.join(cid_dir, f"{cid}_protein.pdb")

        if not os.path.exists(pdb_path):
            continue

        try:
            structure = parser.get_structure(cid, pdb_path)

            residues = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.get_id()[0] == ' ' and residue.get_resname() != 'HOH':
                            residues.append(residue)

            sequence = ''.join([seq1(res.get_resname()) for res in residues])
            protein_sequences[cid] = sequence

        except Exception as e:
            print(f"处理 {cid} 时出错: {str(e)}")

    return protein_sequences


def esm2_pretraining(data_dir, esm_dir, model_name="esm2_t33_650M_UR50D",
                     embedding_dim=1280, max_seq_len=1024, device="cuda" if torch.cuda.is_available() else "cpu"):

    prot_fea_dir = os.path.join(esm_dir, "prot_fea")
    os.makedirs(prot_fea_dir, exist_ok=True)

    processed_files = set(f.replace(".pt", "") for f in os.listdir(prot_fea_dir) if f.endswith(".pt"))

    protein_sequences = extract_protein_sequences(data_dir)

    unprocessed = {cid: seq for cid, seq in protein_sequences.items() if cid not in processed_files}

    if not unprocessed:
        print("所有蛋白质序列已处理完毕")
        return

    print(f"加载ESM-2模型: {model_name}")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device)
    model.eval()

    batch_converter = alphabet.get_batch_converter()
    batch_size = 2

    cids = list(unprocessed.keys())
    sequences = list(unprocessed.values())

    for i in tqdm(range(0, len(cids), batch_size), desc="生成ESM-2嵌入"):
        batch_cids = cids[i:i + batch_size]
        batch_seqs = sequences[i:i + batch_size]

        data = [(cid, seq[:max_seq_len]) for cid, seq in zip(batch_cids, batch_seqs)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)

        token_representations = results["representations"][33]

        for j, cid in enumerate(batch_cids):
            seq_len = len(batch_strs[j])
            embedding = token_representations[j, 1:seq_len + 1].cpu().numpy()

            save_path = os.path.join(prot_fea_dir, f"{cid}.pt")
            torch.save({"lm_prot_fea": embedding}, save_path)


if __name__ == '__main__':
    data_root = "data"
    distance = 5
    input_ligand_format = 'mol2'
    data_dir=os.path.join(data_root, 'toy_set')
    esm_dir =os.path.join(data_root, 'toy_set')
    data_df =pd.read_csv(os.path.join(data_root, "toy_examples.csv"))
    generate_pocket(data_dir=data_dir, distance=distance)
    generate_complex(data_dir, data_df, distance=distance, input_ligand_format=input_ligand_format)
    generate_pock_esm(data_dir, esm_dir, distance)
