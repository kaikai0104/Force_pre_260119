"""
PI-GRU混合神经网络训练脚本
结合Prandtl-Ishlinskii算子和GRU网络进行夹紧力预测
"""
import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 环境配置
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ==================== 数据处理工具函数 ====================

def physical_force_constraint(force_series, dt=0.01, max_rate=5000.0, min_force=0.0):
    """应用物理约束：非负性 + 变化率限制"""
    force_constrained = force_series.copy()
    max_change_per_step = max_rate * dt
    
    for i in range(1, len(force_constrained)):
        delta = force_constrained[i] - force_constrained[i-1]
        if delta > max_change_per_step:
            force_constrained[i] = force_constrained[i-1] + max_change_per_step
        elif delta < -max_change_per_step:
            force_constrained[i] = force_constrained[i-1] - max_change_per_step
    
    force_constrained = np.maximum(force_constrained, min_force)
    return force_constrained

def moving_average(a, w):
    """移动平均滤波"""
    if w <= 1:
        return a.copy()
    cumsum = np.cumsum(np.insert(a, 0, 0.0))
    res = (cumsum[w:] - cumsum[:-w]) / float(w)
    pad_left = np.full(w//2, res[0])
    pad_right = np.full(len(a) - len(res) - len(pad_left), res[-1])
    return np.concatenate([pad_left, res, pad_right])

def best_lag(a, b, max_lag=100):
    """互相关分析找最佳时间延迟"""
    a0, b0 = a - np.mean(a), b - np.mean(b)
    lags = np.arange(-max_lag, max_lag+1)
    corr = []
    for lag in lags:
        if lag < 0:
            corr.append(np.dot(a0[:lag], b0[-lag:]))
        elif lag > 0:
            corr.append(np.dot(a0[lag:], b0[:-lag]))
        else:
            corr.append(np.dot(a0, b0))
    return int(lags[int(np.argmax(corr))])

def shift(a, lag):
    """时间序列移位"""
    if lag == 0:
        return a.copy()
    if lag > 0:
        return np.concatenate([np.full(lag, a[0]), a[:-lag]])
    lag = -lag
    return np.concatenate([a[lag:], np.full(lag, a[-1])])

def play_operator_series(x_series, r):
    """PI模型的Play算子"""
    y = np.zeros_like(x_series)
    y[0] = x_series[0]
    for t in range(1, len(x_series)):
        delta = x_series[t] - y[t-1]
        if delta > r:
            y[t] = x_series[t] - r
        elif delta < -r:
            y[t] = x_series[t] + r
        else:
            y[t] = y[t-1]
    return y

# ==================== 神经网络模型 ====================

class ResidualGRU(nn.Module):
    """GRU残差网络"""
    def __init__(self, in_dim=5, hidden=8):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

class SeqDS(Dataset):
    """序列数据集"""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]

# ==================== 主训练流程 ====================

def main():
    # 参数解析
    ap = argparse.ArgumentParser(description="PI-GRU混合模型训练")
    
    # 数据参数
    ap.add_argument("--inputs", type=str, default="*.xlsx", help="输入Excel文件匹配模式")
    ap.add_argument("--dt", type=float, default=0.01, help="采样周期(秒)")
    ap.add_argument("--ma", type=int, default=5, help="移动平均窗口大小")
    
    # PI模型参数
    ap.add_argument("--M", type=int, default=16, help="Play算子数量")
    ap.add_argument("--rmin", type=float, default=0.005, help="最小阈值r(相对位移范围)")
    ap.add_argument("--rmax", type=float, default=0.2, help="最大阈值r(相对位移范围)")
    ap.add_argument("--ridge", type=float, default=1e-6, help="PI拟合的岭回归系数")
    ap.add_argument("--use_diq", action="store_true", help="是否包含电流导数项")
    
    # GRU参数
    ap.add_argument("--seq_len", type=int, default=32, help="序列长度")
    ap.add_argument("--stride", type=int, default=4, help="窗口滑动步长")
    ap.add_argument("--hidden", type=int, default=16, help="GRU隐藏层大小")
    
    # 训练参数
    ap.add_argument("--epochs", type=int, default=100, help="训练轮数")
    ap.add_argument("--batch_train", type=int, default=256, help="训练批次大小")
    ap.add_argument("--batch_eval", type=int, default=512, help="验证批次大小")
    ap.add_argument("--lr", type=float, default=1e-3, help="学习率")
    ap.add_argument("--clip", type=float, default=1.0, help="梯度裁剪")
    
    # 其他参数
    ap.add_argument("--mode_dx_thr", type=float, default=1e-5, help="模式判断dx阈值")
    ap.add_argument("--mode_iq_ratio", type=float, default=0.02, help="模式判断电流比例")
    ap.add_argument("--loss_w_mode", type=float, default=2.0, help="模式切换点损失权重")
    ap.add_argument("--downsample", type=int, default=1, help="下采样因子")
    ap.add_argument("--loss", type=str, default="smoothl1", choices=["l1","mse","smoothl1"], help="损失函数类型")
    ap.add_argument("--iq_scale", type=float, default=1.0, help="电流特征缩放系数")
    ap.add_argument("--force_max_rate", type=float, default=5000.0, help="夹紧力最大变化率(N/s)")
    ap.add_argument("--force_min", type=float, default=0.0, help="夹紧力最小值(N)")
    ap.add_argument("--apply_physical_constraint", action="store_true", help="应用物理约束")
    ap.add_argument("--artifacts", type=str, default="./artifacts", help="输出目录")
    
    args = ap.parse_args()
    os.makedirs(args.artifacts, exist_ok=True)

    # ========== 1. 数据加载 ==========
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_pattern = os.path.join(script_dir, args.inputs)
    paths = sorted(glob.glob(excel_pattern))
    
    if not paths:
        raise FileNotFoundError(f"未找到Excel文件: {excel_pattern}")
    print(f"找到 {len(paths)} 个Excel文件")
    
    dfs = []
    for fp in paths:
        try:
            df = pd.read_excel(fp, header=0, engine="openpyxl")
            df = df.iloc[:, :3].copy()
            df.columns = ["x", "iq", "F"]
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            file_num = int(os.path.basename(fp).split('.')[0])
            dfs.extend([df.reset_index(drop=True)] * 10)  # 10倍数据增强
            print(f"  [{file_num}] 样本数: {len(df)} → {len(df)*10}")
        except Exception as e:
            print(f"  [错误] {fp}: {str(e)}")
            continue
            
    data = pd.concat(dfs, axis=0, ignore_index=True)
    print(f"总样本数: {len(data)}\n")

    # ========== 2. 数据预处理 ==========
    dt = args.dt
    x = pd.to_numeric(data["x"], errors="coerce").values.astype(np.float64)
    iq = pd.to_numeric(data["iq"], errors="coerce").values.astype(np.float64)
    F = pd.to_numeric(data["F"], errors="coerce").values.astype(np.float64)
    mask = np.isfinite(x) & np.isfinite(iq) & np.isfinite(F)
    x, iq, F = x[mask], iq[mask], F[mask]

    x_s = moving_average(x, args.ma)
    iq_s = moving_average(iq, args.ma)
    F_s = moving_average(F, args.ma)
    dx = moving_average(np.gradient(x_s, dt), args.ma)
    diq = moving_average(np.gradient(iq_s, dt), args.ma)

    iq_s_scaled = iq_s * args.iq_scale
    diq_scaled = diq * args.iq_scale

    # ========== 3. PI基线模型拟合 ==========
    lag = best_lag(x_s, F_s, max_lag=100)
    F_aligned = shift(F_s, lag)
    print(f"时间延迟: {lag} 样本 ({lag*dt:.3f} 秒)")

    x_range = float(np.max(x_s) - np.min(x_s))
    r_list = np.linspace(args.rmin*x_range, args.rmax*x_range, args.M)
    P = np.zeros((len(x_s), args.M), dtype=np.float64)
    for k, r in enumerate(r_list):
        P[:,k] = play_operator_series(x_s, r)

    components = [P, dx.reshape(-1,1), iq_s_scaled.reshape(-1,1)]
    if args.use_diq:
        components.append(diq_scaled.reshape(-1,1))
    components.append(np.ones((len(x_s),1)))
    
    X = np.hstack(components)
    y = F_aligned
    XtX = X.T @ X + args.ridge * np.eye(X.shape[1])
    Xty = X.T @ y
    w = np.linalg.solve(XtX, Xty)
    w_P = w[:args.M]
    beta_dx = w[args.M]
    alpha_iq = w[args.M+1]
    offset = args.M + 2
    if args.use_diq:
        beta_diq = w[offset - 1]
        bias = w[offset]
    else:
        beta_diq = 0.0
        bias = w[args.M+2]
    
    F0 = X @ w
    
    if args.apply_physical_constraint:
        F0 = physical_force_constraint(F0, dt=dt, max_rate=args.force_max_rate, min_force=args.force_min)
        print(f"物理约束: [{np.min(F0):.1f}, {np.max(F0):.1f}] N")

    pi_params = {
        "dt": dt, "lag_samples_x_to_F": int(lag), "M": int(args.M),
        "r_list": r_list.tolist(), "w_P": w_P.tolist(),
        "beta_dx": float(beta_dx), "alpha_iq": float(alpha_iq),
        "beta_diq": float(beta_diq), "bias": float(bias),
        "use_diq": bool(args.use_diq), "iq_scale": float(args.iq_scale),
        "apply_physical_constraint": bool(args.apply_physical_constraint),
        "force_max_rate": float(args.force_max_rate),
        "force_min": float(args.force_min)
    }
    with open(os.path.join(args.artifacts, "pi_params.json"), "w", encoding="utf-8") as f:
        json.dump(pi_params, f, indent=2)

    # ========== 4. 特征工程 ==========
    residual = (F_aligned - F0).astype(np.float32)
    iq_abs_max = np.max(np.abs(iq_s_scaled)) + 1e-12
    mode_flag = ((dx < -args.mode_dx_thr) & (np.abs(iq_s_scaled) > args.mode_iq_ratio * iq_abs_max)).astype(np.float32)
    feats_list = [x_s, dx, iq_s_scaled, diq_scaled, F0, mode_flag]
    feats = np.stack(feats_list, axis=1).astype(np.float32)
    ds = max(1, int(args.downsample))
    feats = feats[::ds]
    residual = residual[::ds]
    mode_flag_ds = mode_flag[::ds]

    # ========== 5. 数据集划分 ==========
    N = len(feats)
    i_tr = int(0.8*N)
    i_va = int(0.9*N)
    Xtr, Xva, Xte = feats[:i_tr], feats[i_tr:i_va], feats[i_va:]
    Ytr_raw, Yva_raw, Yte_raw = residual[:i_tr], residual[i_tr:i_va], residual[i_va:]
    mode_tr, mode_va, mode_te = mode_flag_ds[:i_tr], mode_flag_ds[i_tr:i_va], mode_flag_ds[i_va:]

    r_mean = float(Ytr_raw.mean())
    r_std = float(Ytr_raw.std() + 1e-8)
    Ytr = (Ytr_raw - r_mean) / r_std
    Yva = (Yva_raw - r_mean) / r_std
    Yte = (Yte_raw - r_mean) / r_std

    f_min = Xtr.min(axis=0)
    f_max = Xtr.max(axis=0)
    denom = np.where((f_max - f_min)==0, 1.0, (f_max - f_min))
    
    def norm(Z):
        return 2.0*(Z - f_min)/denom - 1.0
    
    Xtr_n, Xva_n, Xte_n = norm(Xtr), norm(Xva), norm(Xte)

    # ========== 6. 序列窗口生成 ==========
    L = int(args.seq_len)
    stride = int(args.stride)
    
    def make_windows(X, y, L, stride):
        xs, ys, i = [], [], 0
        while i + L <= len(X):
            xs.append(X[i:i+L])
            ys.append(y[i+L-1])
            i += stride
        return np.stack(xs), np.array(ys)
    
    Xtr_w, Ytr_w = make_windows(Xtr_n, Ytr, L, stride)
    Xva_w, Yva_w = make_windows(Xva_n, Yva, L, stride)
    Xte_w, Yte_w = make_windows(Xte_n, Yte, L, stride)

    print(f"训练集: {len(Xtr_w)} 窗口")
    print(f"验证集: {len(Xva_w)} 窗口")
    print(f"测试集: {len(Xte_w)} 窗口\n")

    # ========== 7. 模型初始化 ==========
    device = torch.device("cpu")
    model = ResidualGRU(in_dim=feats.shape[1], hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if args.loss == "l1":
        criterion = nn.L1Loss(reduction='none')
    elif args.loss == "mse":
        criterion = nn.MSELoss(reduction='none')
    else:
        criterion = nn.SmoothL1Loss(beta=1.0, reduction='none')
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)
    
    train_loader = DataLoader(SeqDS(Xtr_w, Ytr_w), batch_size=args.batch_train, shuffle=True)
    val_loader = DataLoader(SeqDS(Xva_w, Yva_w), batch_size=args.batch_eval, shuffle=False)
    
    best_val = float("inf")
    best_path = os.path.join(args.artifacts, "model_residual.pth")
    best_state = None

    # ========== 8. 训练循环 ==========
    print("开始训练...\n")
    for ep in range(1, args.epochs+1):
        model.train()
        tr_loss, count = 0.0, 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb).mean()
            loss.backward()
            
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            count += xb.size(0)
        
        tr_loss = tr_loss / max(1, count)
        
        model.eval()
        va_loss, vcount = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                va_loss += criterion(pred, yb).mean().item() * xb.size(0)
                vcount += xb.size(0)
        
        va_loss = va_loss / max(1, vcount)
        scheduler.step(va_loss)
        
        print(f"Epoch {ep}/{args.epochs} | Train: {tr_loss:.6f} | Val: {va_loss:.6f} | LR: {opt.param_groups[0]['lr']:.2e}")
        
        if va_loss < best_val - 1e-5:
            best_val = va_loss
            best_state = model.state_dict()
            torch.save({
                "state_dict": best_state,
                "f_min": f_min, "f_max": f_max,
                "seq_len": L, "r_mean": r_mean, "r_std": r_std
            }, best_path)

    # ========== 9. 测试评估 ==========
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n训练完成！最佳验证损失: {best_val:.6f}\n")

    test_starts = list(range(0, len(Xte_n)-L+1, stride))
    
    def make_test_windows(X, y, starts, L):
        xs, ys = [], []
        for s in starts:
            xs.append(X[s:s+L])
            ys.append(y[s+L-1])
        return torch.from_numpy(np.stack(xs)).float(), torch.from_numpy(np.array(ys)).float()
    
    Xte_w_t, Yte_w_t = make_test_windows(Xte_n, Yte, test_starts, L)
    
    model.eval()
    preds_norm = []
    with torch.no_grad():
        for i in range(Xte_w_t.size(0)):
            p = model(Xte_w_t[i:i+1]).item()
            preds_norm.append(p)
    
    preds_norm = np.array(preds_norm, dtype=np.float32)
    Yte_aligned = Yte[np.array(test_starts) + L - 1]
    res_pred = preds_norm * r_std + r_mean
    res_true = Yte_aligned * r_std + r_mean
    
    if args.apply_physical_constraint:
        test_F0_slice = F0[np.array(test_starts) + L - 1]
        full_pred = test_F0_slice + res_pred
        full_pred = physical_force_constraint(full_pred, dt=dt, max_rate=args.force_max_rate, min_force=args.force_min)
        res_pred = full_pred - test_F0_slice
    
    mae_res = float(np.mean(np.abs(res_pred - res_true)))
    rmse_res = float(np.sqrt(np.mean((res_pred - res_true)**2)))
    print(f"测试集残差: MAE={mae_res:.3f} | RMSE={rmse_res:.3f}")

    # ========== 10. 保存归一化参数 ==========
    feature_norm = {
        "f_min": [float(v) for v in f_min],
        "f_max": [float(v) for v in f_max],
        "seq_len": int(L),
        "r_mean": r_mean,
        "r_std": r_std,
        "apply_physical_constraint": bool(args.apply_physical_constraint),
        "force_max_rate": float(args.force_max_rate),
        "force_min": float(args.force_min),
        "feature_order": ["x", "dx", "iq", "diq", "F0", "mode"]
    }
    
    with open(os.path.join(args.artifacts, "feature_norm.json"), "w", encoding="utf-8") as f:
        json.dump(feature_norm, f, indent=2)
    
    print(f"\n训练artifacts已保存到: {args.artifacts}")

if __name__ == "__main__":
    main()