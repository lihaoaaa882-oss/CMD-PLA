# %%
import os
import shutil
import pandas as pd
import torch
from HG import HG
from dataset import GraphDataset, PLIDataLoader
import numpy as np
from utils_hg import *
from sklearn.metrics import mean_squared_error
import metrics


# %%

def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    for data in dataloader:
        drug, pock, comp, esm_fea = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
        with torch.no_grad():
            pred = model(drug, pock, comp, esm_fea)
            label = data[0].y

            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))
    evaluation = {
        'RMSE': metrics.RMSE(label, pred),
        'MAE': metrics.MAE(label, pred),
        'SD': metrics.SD(label, pred),
        'CORR': metrics.CORR(label, pred),
    }
    model.train()

    return rmse, coff, evaluation


# %%
data_root = "data"
graph_type = 'Graph_HG'
batch_size = 128
num_workers = 0
valid_dir = os.path.join(data_root, 'PDBbind')
test_dir = os.path.join(data_root, 'PDBbind')
test2013_dir = os.path.join(data_root, 'CASF_2013')
test2016_dir = os.path.join(data_root, 'CASF_2016')

valid_df = pd.read_csv(os.path.join(data_root + 'valid.csv'))
test_df = pd.read_csv(os.path.join(data_root + 'test.csv'))
test2013_df = pd.read_csv(os.path.join(data_root + 'test_2013.csv'))
test2016_df = pd.read_csv(os.path.join(data_root + 'test_2016.csv'))

valid_set = GraphDataset(valid_dir, valid_df, graph_type=graph_type, create=False,dis_threshold=5)
test_set = GraphDataset(test_dir, test_df, graph_type=graph_type, create=False,dis_threshold=5)
test2013_set = GraphDataset(test2013_dir, test2013_df, graph_type=graph_type, create=False,dis_threshold=5)
test2016_set = GraphDataset(test2016_dir, test2016_df, graph_type=graph_type, create=False,dis_threshold=5)

valid_loader = PLIDataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = PLIDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test2016_loader = PLIDataLoader(test2016_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test2013_loader = PLIDataLoader(test2013_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

device = torch.device('cuda:1')
model = HG(35, 256, 1, n_layers=4, normalize=True).to(device)
model = model.cuda()

model_list = [
    '/tmp/hybrid/hybrid_main/model/20251210_194114_HybridGeo_repeat0/model/epoch-606, train_loss-0.0690, train_rmse-0.2628, valid_rmse-1.1573, valid_pr-0.7940.pt',
    '/tmp/hybrid/hybrid_main/model/20251211_213404_HybridGeo_repeat0/model/epoch-750, train_loss-0.0701, train_rmse-0.2648, valid_rmse-1.1555, valid_pr-0.7935.pt',
    '/tmp/hybrid/hybrid_main/model/20251218_110538_HybridGeo_repeat0/model/epoch-461, train_loss-0.0965, train_rmse-0.3107, valid_rmse-1.1683, valid_pr-0.7875.pt'
    ]


def mean_std(RMSE, MAE, SD, CORR):
    RMSE_mean, RMSE_std = np.mean(RMSE, axis=0), np.std(RMSE, axis=0)
    MAE_mean, MAE_std = np.mean(MAE, axis=0), np.std(MAE, axis=0)
    SD_mean, SD_std = np.mean(SD, axis=0), np.std(SD, axis=0)
    CORR_mean, CORR_std = np.mean(CORR, axis=0), np.std(CORR, axis=0)
    print(f'{RMSE_mean[0]:.3f}' + f'({RMSE_std[0]:.3f})' + '\t'
          + f'{MAE_mean[0]:.3f}' + f'({MAE_std[0]:.3f})' + '\t'
          + f'{SD_mean[0]:.3f}' + f'({SD_std[0]:.3f})' + '\t'
          + f'{CORR_mean[0]:.3f}' + f'({CORR_std[0]:.3f})' + '\t')


repeats = 3
RMSE_val, MAE_val, SD_val, CORR_val = [], [], [], []
RMSE_tes, MAE_tes, SD_tes, CORR_tes = [], [], [], []
RMSE_2013, MAE_2013, SD_2013, CORR_2013 = [], [], [], []
RMSE_2016, MAE_2016, SD_2016, CORR_2016 = [], [], [], []
RMSE_hiq, MAE_hiq, SD_hiq, CORR_hiq = [], [], [], []
for repeat in range(repeats):
    load_model_dict(model, model_list[repeat])
    model = model.to(device)

    valid_rmse, valid_coff, val_evaluation = val(model, valid_loader, device)
    test_rmse, test_coff, test_evaluation = val(model, test_loader, device)
    test2013_rmse, test2013_coff, test2013_evaluation = val(model, test2013_loader, device)
    test2016_rmse, test2016_coff, test2016_evaluation = val(model, test2016_loader, device)

    RMSE_val.append(val_evaluation['RMSE'])
    MAE_val.append(val_evaluation['MAE'])
    SD_val.append(val_evaluation['SD'])
    CORR_val.append(val_evaluation['CORR'])

    RMSE_tes.append(test_evaluation['RMSE'])
    MAE_tes.append(test_evaluation['MAE'])
    SD_tes.append(test_evaluation['SD'])
    CORR_tes.append(test_evaluation['CORR'])

    RMSE_2013.append(test2013_evaluation['RMSE'])
    MAE_2013.append(test2013_evaluation['MAE'])
    SD_2013.append(test2013_evaluation['SD'])
    CORR_2013.append(test2013_evaluation['CORR'])

    RMSE_2016.append(test2016_evaluation['RMSE'])
    MAE_2016.append(test2016_evaluation['MAE'])
    SD_2016.append(test2016_evaluation['SD'])
    CORR_2016.append(test2016_evaluation['CORR'])

rmse_val, mae_val, sd_val, corr_val = np.vstack(RMSE_val), np.vstack(MAE_val), np.vstack(SD_val), np.vstack(CORR_val)
rmse_tes, mae_tes, sd_tes, corr_tes = np.vstack(RMSE_tes), np.vstack(MAE_tes), np.vstack(SD_tes), np.vstack(CORR_tes)
rmse_2013, mae_2013, sd_2013, corr_2013 = np.vstack(RMSE_2013), np.vstack(
    MAE_2013), np.vstack(SD_2013), np.vstack(CORR_2013)
rmse_2016, mae_2016, sd_2016, corr_2016 = np.vstack(RMSE_2016), np.vstack(
    MAE_2016), np.vstack(SD_2016), np.vstack(CORR_2016)


mean_std(rmse_val, mae_val, sd_val, corr_val)
mean_std(rmse_tes, mae_tes, sd_tes, corr_tes)
mean_std(rmse_2013, mae_2013, sd_2013, corr_2013)
mean_std(rmse_2016, mae_2016, sd_2016, corr_2016)