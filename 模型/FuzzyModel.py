# 完整模型_模糊集.py
"""
完整模型：简单基线 + 模糊集隶属度模块
目标：验证模糊集对骨龄预测的提升效果
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import EfficientNet_B4_Weights
import warnings
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import cv2

warnings.filterwarnings('ignore')


# ======================
# 配置
# ======================
class Config:
    def __init__(self):
        self.data_dir = r"D:/boneage/1/preprocessed_data/preprocessed_data"
        self.output_dir = r"D:/boneage/1/fuzzy_model_results"
        self.image_root = r"D:/boneage/1/boneage-training-dataset" 
        os.makedirs(self.output_dir, exist_ok=True)

        self.train_csv = os.path.join(self.data_dir, "train_fold1.csv")
        self.val_csv = os.path.join(self.data_dir, "val_fold1.csv")

        self.batch_size = 16
        self.epochs = 50                     # 修改：从 30 改为 50

        self.lr = 5e-4
        self.lr_backbone = 5e-5
        self.weight_decay = 1e-5

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_workers = 0

        # 模糊集配置
        self.num_fuzzy_stages = 9
        self.fuzzy_centers = [0, 12, 24, 36, 60, 78, 96, 144, 192]
        self.fuzzy_widths = [8, 10, 10, 12, 15, 15, 18, 20, 24]

        # 损失权重
        self.lambda_age = 1.0
        self.lambda_fuzzy = 0.5             # 修改：从 0.1 改为 0.5

        print("=" * 60)
        print("完整模型：简单基线 + 模糊集模块")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"模糊阶段数: {self.num_fuzzy_stages}")
        print(f"模糊损失权重: {self.lambda_fuzzy}")
        print("=" * 60)


cfg = Config()


# ======================
# 辅助函数：智能获取图片路径
# ======================
def get_image_path(raw_path):
    """尝试多种方式找到正确的图片路径"""
    # 1. 如果是绝对路径且文件存在，直接返回
    if os.path.isabs(raw_path) and os.path.exists(raw_path):
        return raw_path
    
    # 2. 尝试直接使用 raw_path（可能是相对路径）
    if os.path.exists(raw_path):
        return raw_path
    
    # 3. 尝试拼接 image_root
    candidate1 = os.path.join(cfg.image_root, raw_path)
    if os.path.exists(candidate1):
        return candidate1
    
    # 4. 尝试只取文件名（去除可能的上级目录）
    basename = os.path.basename(raw_path)
    candidate2 = os.path.join(cfg.image_root, basename)
    if os.path.exists(candidate2):
        return candidate2
    
    # 5. 如果 raw_path 已经包含 image_root 的一部分，尝试去除重复
    # 例如 raw_path = "boneage-training-dataset/3316.png"，image_root = ".../boneage-training-dataset"
    # 拼接会重复，所以尝试直接使用 raw_path 作为相对路径
    if raw_path.startswith("boneage-training-dataset"):
        rel = raw_path.replace("boneage-training-dataset", "", 1).lstrip("/\\")
        candidate3 = os.path.join(cfg.image_root, rel)
        if os.path.exists(candidate3):
            return candidate3
    
    # 6. 都不行，打印警告并返回 None
    print(f"警告: 无法定位图片 {raw_path}")
    return None


# ======================
# 数据集类
# ======================
class PreprocessedDataset(Dataset):
    def __init__(self, csv_path, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.is_train = is_train

        # 预计算模糊标签
        self.fuzzy_labels = []
        for _, row in self.df.iterrows():
            fuzzy = self._create_fuzzy_membership(row['age'])
            self.fuzzy_labels.append(fuzzy)

        print(f"{'训练' if is_train else '验证'}集: {len(self.df)} 样本")

    def _create_fuzzy_membership(self, age):
        """创建模糊集隶属度标签"""
        memberships = []
        for c, w in zip(cfg.fuzzy_centers, cfg.fuzzy_widths):
            mu = np.exp(-((age - c) ** 2) / (2 * w ** 2))
            memberships.append(mu)
        memberships = np.array(memberships, dtype=np.float32)
        if memberships.sum() > 0:
            memberships = memberships / memberships.sum()
        return memberships

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 智能获取图片路径
        img_path = get_image_path(row['image_path'])
        img = cv2.imread(img_path) if img_path is not None else None
        
        if img is None:
            # 如果还是读取失败，使用空白图片并打印一次警告
            if idx == 0:
                print(f"警告: 无法读取图片 {img_path or row['image_path']}，使用空白图片代替")
            img = np.zeros((380, 380, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (380, 380))

        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        if self.is_train and np.random.random() > 0.5:
            img = np.fliplr(img).copy()

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        age = torch.tensor(row['age'], dtype=torch.float32)
        gender = torch.tensor([row['gender']], dtype=torch.float32)
        fuzzy = torch.tensor(self.fuzzy_labels[idx], dtype=torch.float32)

        return img, age, gender, fuzzy


# ======================
# 模糊集隶属度层
# ======================
class FuzzyMembershipLayer(nn.Module):
    """模糊集隶属度预测层"""

    def __init__(self, num_stages=9, feature_dim=256):
        super().__init__()
        self.num_stages = num_stages

        # 固定的高斯中心（可学习的偏移）
        centers = torch.tensor(cfg.fuzzy_centers, dtype=torch.float32)
        widths = torch.tensor(cfg.fuzzy_widths, dtype=torch.float32)
        self.register_buffer('centers', centers)
        self.register_buffer('widths', widths)

        # 可学习的偏移
        self.center_shift = nn.Parameter(torch.zeros(num_stages))
        self.width_scale = nn.Parameter(torch.ones(num_stages))

        # 从特征预测模糊隶属度
        self.fuzzy_predictor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_stages)
        )

    def forward(self, features, age=None):
        """预测模糊隶属度"""
        fuzzy_pred = torch.sigmoid(self.fuzzy_predictor(features))
        fuzzy_pred = fuzzy_pred / (fuzzy_pred.sum(dim=1, keepdim=True) + 1e-8)

        ideal_fuzzy = None
        if age is not None:
            # 计算理想的模糊隶属度（基于真实年龄）
            centers_adj = self.centers + self.center_shift
            widths_adj = self.widths * torch.abs(self.width_scale)
            ideal_list = []
            for c, w in zip(centers_adj, widths_adj):
                mu = torch.exp(-((age - c) ** 2) / (2 * w ** 2))
                ideal_list.append(mu)
            ideal_fuzzy = torch.stack(ideal_list, dim=1)
            ideal_fuzzy = ideal_fuzzy / (ideal_fuzzy.sum(dim=1, keepdim=True) + 1e-8)

        return fuzzy_pred, ideal_fuzzy


# ======================
# 完整模型（B4 + 性别 + 模糊集）
# ======================
class FuzzyBoneAgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载B4特征提取器
        backbone = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # 冻结前2层
        features_list = list(self.features.children())
        for i, layer in enumerate(features_list[:2]):
            for param in layer.parameters():
                param.requires_grad = False

        # B4特征维度
        self.feature_dim = 1792

        # 特征投影（降维到256，用于模糊集）
        self.feature_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 模糊集层
        self.fuzzy_layer = FuzzyMembershipLayer(
            num_stages=cfg.num_fuzzy_stages,
            feature_dim=256
        )

        # 性别嵌入
        self.gender_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 回归头（融合图像特征 + 模糊特征 + 性别）
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim + cfg.num_fuzzy_stages + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 1)
        )

    def forward(self, x, gender, age=None):
        # 特征提取
        x = self.features(x)
        x = self.avgpool(x)          # [b, 1792, 1, 1]
        x_flat = torch.flatten(x, 1) # [b, 1792]

        # 投影到256维用于模糊集（需要4维输入）
        projected_feat = self.feature_projector(x)  # x 已经是 [b,1792,1,1]，可以直接用

        # 模糊集预测
        fuzzy_pred, ideal_fuzzy = self.fuzzy_layer(projected_feat, age)

        # 性别嵌入
        gender_feat = self.gender_embed(gender)

        # 融合所有特征
        combined = torch.cat([x_flat, fuzzy_pred, gender_feat], dim=1)

        # 回归
        age_pred = self.regressor(combined).squeeze(-1)

        return {
            'age_pred': age_pred,
            'fuzzy_pred': fuzzy_pred,
            'ideal_fuzzy': ideal_fuzzy
        }


# ======================
# 组合损失函数
# ======================
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.age_loss_fn = nn.HuberLoss(delta=1.0)
        self.fuzzy_loss_fn = nn.MSELoss()

    def forward(self, outputs, targets):
        age_true = targets['age']
        fuzzy_true = targets['fuzzy']

        # 年龄回归损失
        age_loss = self.age_loss_fn(outputs['age_pred'], age_true)

        # 模糊集损失
        fuzzy_loss = 0
        if fuzzy_true is not None and outputs['fuzzy_pred'] is not None:
            fuzzy_loss = self.fuzzy_loss_fn(outputs['fuzzy_pred'], fuzzy_true)

        total = cfg.lambda_age * age_loss + cfg.lambda_fuzzy * fuzzy_loss

        return {
            'total': total,
            'age': age_loss.item(),
            'fuzzy': fuzzy_loss.item() if isinstance(fuzzy_loss, torch.Tensor) else 0
        }


# ======================
# 训练函数
# ======================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    age_loss_sum = 0
    fuzzy_loss_sum = 0
    all_preds = []
    all_targets = []

    for imgs, ages, genders, fuzzy in tqdm(loader, desc="训练"):
        imgs = imgs.to(cfg.device)
        ages = ages.to(cfg.device)
        genders = genders.to(cfg.device)
        fuzzy = fuzzy.to(cfg.device)

        optimizer.zero_grad()
        outputs = model(imgs, genders, age=ages)
        targets = {'age': ages, 'fuzzy': fuzzy}
        losses = criterion(outputs, targets)

        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += losses['total'].item()
        age_loss_sum += losses['age']
        fuzzy_loss_sum += losses['fuzzy']
        all_preds.extend(outputs['age_pred'].detach().cpu().numpy())
        all_targets.extend(ages.cpu().numpy())

    n_batches = len(loader)
    return {
        'loss': total_loss / n_batches,
        'age_loss': age_loss_sum / n_batches,
        'fuzzy_loss': fuzzy_loss_sum / n_batches,
        'mae': mean_absolute_error(all_targets, all_preds)
    }


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    age_loss_sum = 0
    fuzzy_loss_sum = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, ages, genders, fuzzy in tqdm(loader, desc="验证"):
            imgs = imgs.to(cfg.device)
            ages = ages.to(cfg.device)
            genders = genders.to(cfg.device)
            fuzzy = fuzzy.to(cfg.device)

            outputs = model(imgs, genders, age=ages)
            targets = {'age': ages, 'fuzzy': fuzzy}
            losses = criterion(outputs, targets)

            total_loss += losses['total'].item()
            age_loss_sum += losses['age']
            fuzzy_loss_sum += losses['fuzzy']
            all_preds.extend(outputs['age_pred'].cpu().numpy())
            all_targets.extend(ages.cpu().numpy())

    n_batches = len(loader)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))

    return {
        'loss': total_loss / n_batches,
        'age_loss': age_loss_sum / n_batches,
        'fuzzy_loss': fuzzy_loss_sum / n_batches,
        'mae': mae,
        'rmse': rmse,
        'preds': np.array(all_preds),
        'targets': np.array(all_targets)
    }


# ======================
# 主函数
# ======================
def main():
    print("=" * 60)
    print("完整模型训练（B4 + 性别 + 模糊集）")
    print("=" * 60)

    # 加载数据
    print("\n[1/4] 加载数据...")
    train_dataset = PreprocessedDataset(cfg.train_csv, is_train=True)
    val_dataset = PreprocessedDataset(cfg.val_csv, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers)

    # 创建模型
    print("\n[2/4] 创建模型...")
    model = FuzzyBoneAgeModel().to(cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params / 1e6:.2f}M")
    print(f"可训练: {trainable_params / 1e6:.2f}M")

    # 优化器
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'features' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': cfg.lr_backbone},
        {'params': other_params, 'lr': cfg.lr}
    ], weight_decay=cfg.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = CombinedLoss()

    # 训练
    print("\n[3/4] 开始训练...")
    best_mae = float('inf')
    history = {'train_mae': [], 'val_mae': [], 'train_loss': [], 'val_loss': []}

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, criterion)
        val_metrics = validate(model, val_loader, criterion)
        scheduler.step()

        history['train_mae'].append(train_metrics['mae'])
        history['val_mae'].append(val_metrics['mae'])
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])

        print(f"  训练 - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.2f}月")
        print(
            f"  验证 - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.2f}月, RMSE: {val_metrics['rmse']:.2f}月")
        print(f"  年龄损失: {val_metrics['age_loss']:.4f}, 模糊损失: {val_metrics['fuzzy_loss']:.4f}")

        if val_metrics['mae'] < best_mae:
            best_mae = val_metrics['mae']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_mae': val_metrics['mae'],
                'val_rmse': val_metrics['rmse'],
                'history': history
            }, os.path.join(cfg.output_dir, 'fuzzy_model_best.pth'))
            print(f"  ✓ 保存最佳模型 (MAE: {val_metrics['mae']:.2f}月)")

    # 可视化
    print("\n[4/4] 生成报告...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 训练曲线
    ax1 = axes[0, 0]
    ax1.plot(history['train_mae'], label='训练MAE', marker='o')
    ax1.plot(history['val_mae'], label='验证MAE', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MAE (月)')
    ax1.set_title(f'模糊集模型 (最佳MAE: {best_mae:.2f}月)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 损失曲线
    ax2 = axes[0, 1]
    ax2.plot(history['train_loss'], label='训练损失', marker='o')
    ax2.plot(history['val_loss'], label='验证损失', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('损失曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 预测 vs 真实
    ax3 = axes[1, 0]
    ax3.scatter(val_metrics['targets'], val_metrics['preds'], alpha=0.5, s=5)
    ax3.plot([0, 228], [0, 228], 'r--', label='理想预测')
    ax3.set_xlabel('真实年龄 (月)')
    ax3.set_ylabel('预测年龄 (月)')
    ax3.set_title(f'预测 vs 真实 (MAE: {best_mae:.2f}月)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 误差分布
    ax4 = axes[1, 1]
    errors = val_metrics['preds'] - val_metrics['targets']
    ax4.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax4.axvline(x=0, color='r', linestyle='--')
    ax4.axvline(x=np.mean(errors), color='g', linestyle='--', label=f'均值: {np.mean(errors):.2f}')
    ax4.set_xlabel('预测误差 (月)')
    ax4.set_ylabel('频次')
    ax4.set_title(f'误差分布 (标准差: {np.std(errors):.2f}月)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, 'fuzzy_model_results.png'), dpi=150)
    plt.show()

    # 保存结果
    results = {
        'model': 'Fuzzy Model (B4 + Gender + Fuzzy)',
        'best_mae': float(best_mae),
        'baseline_mae': 8.32,  # 从简单基线得到
        'improvement': 8.32 - best_mae,
        'config': {
            'batch_size': cfg.batch_size,
            'epochs': cfg.epochs,
            'lambda_fuzzy': cfg.lambda_fuzzy
        }
    }

    import json
    with open(os.path.join(cfg.output_dir, 'fuzzy_model_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ 模糊集模型训练完成！")
    print("=" * 60)
    print(f"简单基线 MAE: 8.32 个月")
    print(f"模糊集模型 MAE: {best_mae:.2f} 个月")
    print(f"提升: {8.32 - best_mae:.2f} 个月")
    print("=" * 60)

    return best_mae


if __name__ == '__main__':
    main()