# gen_lig_mol2vec.py
import os
import warnings
from tqdm import tqdm
import torch
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, MolSentence
import numpy as np

warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

BASE_DIR = "toy_set"
CSV_PATH = "toy_examples.csv"


MOL2VEC_MODEL_PATH = "model_300dim.pkl"

LIGAND_SUFFIX = "_ligand.mol2"


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"[INFO] 加载 {CSV_PATH}, 共 {len(df)} 条样本")

    print(f"[INFO] 加载 Mol2Vec 模型: {MOL2VEC_MODEL_PATH}")
    model = word2vec.Word2Vec.load(MOL2VEC_MODEL_PATH)

    if hasattr(model.wv, "key_to_index"):
        keys = set(model.wv.key_to_index.keys())
    else:
        keys = set(model.wv.vocab.keys())

    unseen = "UNK"
    unseen_vec = model.wv.word_vec(unseen)

    for _, row in tqdm(df.iterrows(), total=len(df), ncols=80):
        cid = str(row["id"]).lower()

        complex_dir = os.path.join(BASE_DIR, cid)
        os.makedirs(complex_dir, exist_ok=True)

        lig_path = os.path.join(complex_dir, f"{cid}{LIGAND_SUFFIX}")
        if not os.path.exists(lig_path):
            print(f"[跳过] 未找到配体文件: {lig_path}")
            continue

        out_path = os.path.join(complex_dir, f"{cid}_lig.pt")

        try:

            mol = Chem.MolFromMol2File(lig_path, sanitize=True, removeHs=True)
            if mol is None:
                print(f"[跳过] RDKit 无法读取配体: {lig_path}")
                continue

            sentence = MolSentence(mol2alt_sentence(mol, radius=1))

            toks = [tok for tok in sentence if tok in keys]

            if len(toks) == 0:
                print(f"[提示] {cid} 所有 token 都不在词表中，使用 UNK 向量")
                vector = unseen_vec
            else:
                vecs = np.stack([model.wv.word_vec(tok) for tok in toks], axis=0)  # (num_tok, 300)
                vector = vecs.mean(axis=0)

            norm = np.linalg.norm(vector) + 1e-8
            vector = vector / norm

            lig_vec = torch.as_tensor(vector, dtype=torch.float32)

            torch.save({"lig_mol2vec": lig_vec}, out_path)
        except Exception as e:
            print(f"[错误] 处理 {cid} 时出错: {e}")
            continue

    print("[DONE] 全部配体 Mol2Vec 特征提取完成。")


if __name__ == "__main__":
    main()
