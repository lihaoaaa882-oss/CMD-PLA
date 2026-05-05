import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils_hg import AverageMeter, BestMeter, save_model_dict
from HG import HG
from dataset import GraphDataset, PLIDataLoader
from config_hg.config_dict import Config
from log.train_logger import TrainLogger
from sklearn.metrics import mean_squared_error


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    pred_list, label_list = [], []
    for data in dataloader:
        drug, pock, comp, esm_fea = (
            data[0].to(device, non_blocking=True),
            data[1].to(device, non_blocking=True),
            data[2].to(device, non_blocking=True),
            data[3].to(device, non_blocking=True),
        )
        pred = model(drug, pock, comp, esm_fea).view(-1)
        label = data[0].y.to(device, non_blocking=True).view(-1)

        # 转到 CPU 并做有限性检查
        p0 = pred.detach().cpu()
        l0 = label.detach().cpu()
        mask = torch.isfinite(p0) & torch.isfinite(l0)
        if not mask.any():
            print("[eval] skipped a batch: all non-finite preds/labels")
            continue

        p = torch.nan_to_num(p0[mask])
        l = torch.nan_to_num(l0[mask])

        pred_list.append(p.numpy())
        label_list.append(l.numpy())

    if len(pred_list) == 0:
        # 没有有效样本，返回安全值
        model.train()
        return float("nan"), 0.0

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    rmse = np.sqrt(mean_squared_error(label, pred))
    pr = np.corrcoef(pred, label)[0, 1] if len(label) > 1 else 0.0

    model.train()
    return rmse, pr


if __name__ == '__main__':
    # ---- 基础配置 ----
    cfg = 'TrainConfig'
    config = Config(cfg)
    args = config.get_config()

    gpu = args.get("gpu", 1)
    graph_type = args.get("graph_type")
    save_model = args.get("save_model", True)
    num_works = args.get("num_works", 4)
    batch_size = args.get("batch_size", 128)
    data_root = args.get('data_root')
    epochs = args.get('epochs', 800)
    repeats = args.get('repeat', 3)
    early_stop_epoch = args.get("early_stop_epoch", 800)

    # ---- 数据路径 ----
    data_path = os.path.join(data_root, 'toy_set')
    test2013_dir = os.path.join(data_root, 'toy_set')
    test2016_dir = os.path.join(data_root, 'toy_set')

    # ---- 划分 ----
    train_df = pd.read_csv(os.path.join(data_root, "toy_examples.csv")).sample(frac=1., random_state=123)
    valid_df = pd.read_csv(os.path.join(data_root, "toy_examples.csv")).sample(frac=1., random_state=123)
    test_df = pd.read_csv(os.path.join(data_root, "toy_examples.csv"))
    test2013_df = pd.read_csv(os.path.join(data_root, 'toy_examples.csv'))
    test2016_df = pd.read_csv(os.path.join(data_root, 'toy_examples.csv'))

    # ---- 数据集/加载器（离线：GraphDataset 里已经用的是 offline CMD 坐标）----
    train_set = GraphDataset(data_path, train_df, graph_type=graph_type, create=False, dis_threshold=5)
    valid_set = GraphDataset(data_path, valid_df, graph_type=graph_type, create=False, dis_threshold=5)
    test_set = GraphDataset(data_path, test_df, graph_type=graph_type, create=False, dis_threshold=5)
    test2013_set = GraphDataset(test2013_dir, test2013_df, graph_type=graph_type, create=False, dis_threshold=5)
    test2016_set = GraphDataset(test2016_dir, test2016_df, graph_type=graph_type, create=False, dis_threshold=5)

    train_loader = PLIDataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_works, pin_memory=True
    )
    valid_loader = PLIDataLoader(
        valid_set, batch_size=batch_size, shuffle=False,
        num_workers=num_works, pin_memory=True
    )
    test_loader = PLIDataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_works, pin_memory=True
    )
    test2013_loader = PLIDataLoader(
        test2013_set, batch_size=batch_size, shuffle=False,
        num_workers=num_works, pin_memory=True
    )
    test2016_loader = PLIDataLoader(
        test2016_set, batch_size=batch_size, shuffle=False,
        num_workers=num_works, pin_memory=True
    )

    device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True

    for repeat in range(repeats):
        args['repeat'] = repeat
        logger = TrainLogger(args, cfg, create=True)
        logger.info(__file__)
        logger.info(
            f"train data: {len(train_set)}\n"
            f"valid data: {len(valid_set)}\n"
            f"test data: {len(test_set)}\n"
            f"test_2013 data: {len(test2013_set)}\n"
            f"test_2016 data: {len(test2016_set)}\n"
        )

        model = HG(
            in_node_nf=35,
            hidden_nf=256,
            out_node_nf=1,
            n_layers=args.get("n_layers", 4),
            normalize=True,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-6)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',          # 指标越小越好
            factor=0.5,          # 触发后把 LR 乘 0.5
            patience=180,        # 连续若干 epoch 没明显改善才降
            threshold=1e-3,      # “明显改善”的相对阈值
            threshold_mode='rel',
            cooldown=0,
            min_lr=2e-5,
            verbose=True,
        )

        criterion = nn.MSELoss()

        running_loss = AverageMeter()
        running_best_mse = BestMeter("min")

        model.train()
        break_flag = False

        best_model_path = None
        best_epoch = -1

        for epoch in range(epochs):
            # ====== 训练 ======
            for data in train_loader:
                drug, pock, comp, esm_fea = (
                    data[0].to(device, non_blocking=True),
                    data[1].to(device, non_blocking=True),
                    data[2].to(device, non_blocking=True),
                    data[3].to(device, non_blocking=True),
                )

                pred = model(drug, pock, comp, esm_fea).view(-1)
                label = data[0].y.to(device, non_blocking=True).view(-1)

                # 1) 只保留有限样本
                mask = torch.isfinite(pred) & torch.isfinite(label)
                if not mask.any():
                    print("[train] skipped batch: all non-finite preds/labels")
                    continue
                pred = pred[mask]
                label = label[mask]

                # 2) 计算损失并兜底
                loss = criterion(pred, label)
                if not torch.isfinite(loss):
                    print("[train] non-finite loss after masking; skip this batch")
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss.update(loss.item(), label.size(0))

            epoch_loss = running_loss.get_average()
            epoch_rmse = np.sqrt(epoch_loss)
            running_loss.reset()

            # ====== 验证 ======
            valid_rmse, valid_pr = evaluate(model, valid_loader, device)

            # ====== 按验证集调整 LR ======
            if np.isfinite(valid_rmse):
                scheduler.step(valid_rmse)
            else:
                logger.info("valid_rmse is NaN/Inf; skip scheduler.step this epoch")

            curr_lr = optimizer.param_groups[0]['lr']
            logger.info(f"[lr] now={curr_lr:.2e}")

            msg = (
                "epoch-%d, train_loss-%.4f, train_rmse-%.4f, "
                "valid_rmse-%.4f, valid_pr-%.4f"
            ) % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
            logger.info(msg)

            # ====== 早停 & 保存 best-valid 模型 ======
            if valid_rmse < running_best_mse.get_best():
                running_best_mse.update(valid_rmse)
                best_epoch = epoch

                if save_model:
                    save_msg = (
                        "epoch-%d, train_loss-%.4f, train_rmse-%.4f, "
                        "valid_rmse-%.4f, valid_pr-%.4f"
                    ) % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
                    save_model_dict(model, logger.get_model_dir(), save_msg)
                    best_model_path = os.path.join(logger.get_model_dir(), save_msg + '.pt')
            else:
                if running_best_mse.counter() > early_stop_epoch:
                    best_mse = running_best_mse.get_best()
                    logger.info(f"early stop in epoch {epoch}")
                    logger.info("best_rmse: %.4f" % best_mse)
                    break_flag = True
                    break

        # ====== 每个 repeat 结束后：加载 best-valid checkpoint，做最终测试 ======
        logger.info("Final evaluation on test sets using best-valid checkpoint.")
        if best_model_path and os.path.exists(best_model_path):
            ck = torch.load(best_model_path, map_location=device)
            loaded_ok = False
            if isinstance(ck, dict):
                for k in ('model_state', 'state_dict'):
                    if k in ck:
                        try:
                            model.load_state_dict(ck[k], strict=False)
                            loaded_ok = True
                            break
                        except Exception:
                            pass
            if not loaded_ok:
                try:
                    model.load_state_dict(ck, strict=False)
                except Exception:
                    pass
            model.to(device)
            model.eval()
            logger.info(f"Loaded best checkpoint from epoch {best_epoch}: {best_model_path}")
        else:
            logger.info("Best checkpoint not found; using current in-memory model state.")

        # 只评一次 test
        test_rmse, test_pr = evaluate(model, test_loader, device)
        test2013_rmse, test2013_pr = evaluate(model, test2013_loader, device)
        test2016_rmse, test2016_pr = evaluate(model, test2016_loader, device)
        best_msg = (
            "FINAL => test_rmse-%.4f, test_pr-%.4f, "
            "test_2013_rmse-%.4f, test_2013_pr-%.4f, "
            "test_2016_rmse-%.4f, test_2016_pr-%.4f"
        ) % (test_rmse, test_pr, test2013_rmse, test2013_pr, test2016_rmse, test2016_pr)
        logger.info(best_msg)

        if break_flag:
            continue