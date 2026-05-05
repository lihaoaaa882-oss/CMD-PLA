import os
import torch
import pandas as pd
from tqdm import tqdm
import esm
import requests
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

LETTER_TO_NUM = {
    'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
    'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
    'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
    'N': 2, 'Y': 18, 'M': 12, 'X': 20
}

base_dir = "toy_set"

train_df = pd.read_csv("toy_examples.csv")

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

batch_converter = alphabet.get_batch_converter()
model.eval()


def extract_sequence_from_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", pdb_file)

        residues_list = [res for res in structure.get_residues() if res.get_id()[0] == ' ']

        residues = []
        for res in residues_list:
            child_keys = res.child_dict.keys()
            if all(atom in child_keys for atom in ('N', 'CA', 'C', 'O')):
                residues.append(res)

        seq = ''.join(seq1(res.get_resname()) for res in residues)

        return seq if seq else None

    except Exception as e:
        print(f"[错误] PDB 解析失败：{pdb_file}，原因：{e}")
        return None

def download_pdb(pdb_id, save_path):
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'w') as f:
                f.write(response.text)
            print(f"[下载] 成功下载 PDB 文件：{pdb_id}")
            return True
        else:
            print(f"[跳过] 下载失败（{response.status_code}）：{pdb_id}")
            return False
    except Exception as e:
        print(f"[跳过] 下载异常 {pdb_id}：{e}")
        return False

for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
    pdb_id = str(row['id']).lower()

    output_dir = os.path.join(base_dir, pdb_id)
    os.makedirs(output_dir, exist_ok=True)
    pdb_path = os.path.join(output_dir, f"{pdb_id}_protein.pdb")

    if not os.path.exists(pdb_path):
        # 自动从RCSB下载
        download_success = download_pdb(pdb_id, pdb_path)
        if not download_success:
            print(f"[跳过] 未能获取 {pdb_id} 的PDB文件（本地+下载都失败）")
            continue

    seq = extract_sequence_from_pdb(pdb_path)
    if seq:
        print(f"[提取] 从 PDB 文件获取了 {pdb_id} 的序列")
    else:
        print(f"[跳过] PDB 文件存在但未提取到有效序列：{pdb_id}")
        continue

    output_path = os.path.join(base_dir, pdb_id, f"{pdb_id}.pt")


    if os.path.exists(output_path):
        print(f"[跳过] 目标文件 {output_path} 已存在，跳过当前处理")
        continue

    try:
        data = [(pdb_id, seq)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]
            embedding = token_representations[0, 1:len(seq)+1]

        torch.save({'lm_prot_fea': embedding.cpu()}, output_path)

    except Exception as e:
        print(f"[错误] 处理 {pdb_id} 时出错：{e}")
        continue


