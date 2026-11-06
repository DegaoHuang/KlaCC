# -*- coding: utf-8 -*-
# Filename: train_cv_test_final_with_vis.py
# Function: 5-fold cross-validation + full training + independent test + t-SNE/pLOGO
# Modification: Classifier input = backbone features(512) + projection features(128)
#               + Fully reproducible seed + per-fold metrics export
#               + ★ FIX: Step2 val 用 stratify 划分；compute_separation 去对角线；Stage2 warmup

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score, matthews_corrcoef,
    average_precision_score, roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, train_test_split   # ★ FIX: 新增 train_test_split
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import logomaker
import argparse
import os
import json
import random
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ====================================================================
#                    ★ Reproducibility helpers ★
# ====================================================================

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(dataset, batch_size, shuffle, seed, num_workers=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=g,
        worker_init_fn=lambda wid: np.random.seed(seed + wid),
    )


# ====================================================================
#                    ★ Hyperparameter Management ★
# ====================================================================

class HyperParams:
    def __init__(self):
        self.TRAIN_DATA_PATH = './train.xlsx'
        self.TEST_DATA_PATH = './test.xlsx'
        self.AA_EMB_PATH = './aa_embeddings.csv'
        self.SEQ_LENGTH = 51
        self.NUM_CLASSES = 2

        self.SEQ_COLUMN = 'protein_sequence'
        self.LABEL_COLUMN = 'label'
        self.POS_COLUMN = 'position'

        self.ADAPTER_DIM = 512

        self.TRANSFORMER_D_MODEL = 256
        self.TRANSFORMER_NHEAD = 8
        self.TRANSFORMER_LAYERS = 4
        self.DROPOUT_CONV = 0.1
        self.POOLING_MODE = 'max'
        self.UNIFIED_DIM = 512

        self.CLASSIFIER_HEAD_DIMS = [128, 64]
        self.CLASSIFIER_HEAD_DROPOUT = 0.1

        self.PROJECTION_HEAD_DIMS = [256, 128]
        self.PROJECTION_OUT_DIM = 128
        self.PROJECTION_HEAD_DROPOUT = 0.1
        self.PROJECTION_USE_BN = True

        self.USE_BCE_LOSS = True
        self.NOISE_STD = 0.05
        self.POS_MARGIN = 0.95
        self.NEG_MARGIN = 0.05
        self.POS_PENALTY = 1.0
        self.NEG_PENALTY = 1.0
        self.SUPCON_TEMPERATURE = 0.02

        self.USE_STAGE1 = True
        self.USE_STAGE2 = True
        self.USE_STAGE3 = True

        self.STAGE1_EPOCHS = 1000
        self.STAGE1_BATCH_SIZE = 32
        self.STAGE1_EARLY_STOP = 20
        self.STAGE1_LR = 1e-4
        self.STAGE1_TARGET_AUC = 1

        self.STAGE2_EPOCHS = 1000
        self.STAGE2_BATCH_SIZE = 2048
        self.STAGE2_EARLY_STOP = 20
        self.STAGE2_LR = 1e-4
        self.STAGE2_MIN_EPOCHS = 5     # ★ FIX: Stage2 warmup，前 N 轮不参与 best 追踪

        self.STAGE3_EPOCHS = 1000
        self.STAGE3_BATCH_SIZE = 32
        self.STAGE3_EARLY_STOP = 20
        self.STAGE3_LR = 1e-4
        self.STAGE3_FREEZE_ENCODER = True

        self.N_FOLDS = 5
        self.SEED = 123

        self.VAL_BATCH_SIZE = 128
        self.TEST_BATCH_SIZE = 128
        self.NUM_WORKERS = 4
        self.USE_AMP = True

        self.OUTPUT_DIR = './experiments/cv_test_final'
        self.SAVE_PLOTS = True
        self.SAVE_MODELS = True
        self.VERBOSE = True

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def print_summary(self):
        print("\n" + "=" * 70)
        print("Experiment configuration (CV + full training + independent test)")
        print("=" * 70)
        print(f"[Training data] {self.TRAIN_DATA_PATH}")
        print(f"[Test data] {self.TEST_DATA_PATH}")
        print(f"[Transformer] d_model={self.TRANSFORMER_D_MODEL}, layers={self.TRANSFORMER_LAYERS}")
        print(f"[Classifier input] {self.UNIFIED_DIM} + {self.PROJECTION_OUT_DIM} = {self.UNIFIED_DIM + self.PROJECTION_OUT_DIM}")
        print(f"[Ablation] Stage1={self.USE_STAGE1}, Stage2={self.USE_STAGE2}, Stage3={self.USE_STAGE3}")
        print(f"[CV] Folds={self.N_FOLDS}, Seed={self.SEED}")
        print(f"[Stage2 warmup] {self.STAGE2_MIN_EPOCHS} epochs")
        print(f"[Output] {self.OUTPUT_DIR}")
        print("=" * 70)


# ====================================================================
#                    ★ Helper Functions ★
# ====================================================================

def build_mlp(input_dim, hidden_dims, output_dim, dropout=0.0, use_bn=False):
    layers = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        if use_bn:
            layers.append(nn.LayerNorm(h_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def compute_all_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
    auprc = average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {'acc': acc, 'recall': recall, 'precision': precision, 'f1': f1,
            'mcc': mcc, 'auc': auc, 'auprc': auprc, 'specificity': specificity}


# ====================================================================
#                    ★ Dataset ★
# ====================================================================

class AAIndexDataset(Dataset):
    def __init__(self, aa_sequences, labels, aa_to_idx, seq_length, return_str=True):
        self.aa_sequences = aa_sequences
        self.labels = labels
        self.aa_to_idx = aa_to_idx
        self.seq_length = seq_length
        self.return_str = return_str

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.aa_sequences[idx]
        indices = np.zeros(self.seq_length, dtype=np.int64)
        for i in range(self.seq_length):
            if i < len(seq):
                indices[i] = self.aa_to_idx.get(seq[i], 20)
            else:
                indices[i] = 20
        if self.return_str:
            return (torch.tensor(indices, dtype=torch.long),
                    seq,
                    torch.tensor(self.labels[idx], dtype=torch.long))
        else:
            return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# ====================================================================
#                    ★ Model ★
# ====================================================================

class TransformerEncoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.adapter_dim = hp.ADAPTER_DIM
        self.d_model = hp.TRANSFORMER_D_MODEL
        self.nhead = hp.TRANSFORMER_NHEAD
        self.num_layers = hp.TRANSFORMER_LAYERS
        self.max_seq_len = hp.SEQ_LENGTH
        self.unified_dim = hp.UNIFIED_DIM
        self.dropout = hp.DROPOUT_CONV

        self.aa_embedding = nn.Embedding(21, self.adapter_dim)
        nn.init.trunc_normal_(self.aa_embedding.weight, std=0.02)

        self.input_proj = nn.Linear(self.adapter_dim, self.d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_seq_len, self.d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.fc = nn.Linear(self.d_model, self.unified_dim)

        self.projection = build_mlp(
            input_dim=self.unified_dim,
            hidden_dims=hp.PROJECTION_HEAD_DIMS,
            output_dim=hp.PROJECTION_OUT_DIM,
            dropout=hp.PROJECTION_HEAD_DROPOUT,
            use_bn=hp.PROJECTION_USE_BN
        )

    def forward(self, x, seq_strs=None, return_projection=False):
        emb = self.aa_embedding(x)
        if self.training and hasattr(self, 'noise_std') and self.noise_std > 0:
            emb = emb + torch.randn_like(emb) * self.noise_std
        local = self.input_proj(emb)
        local = local + self.pos_embed[:, :local.size(1), :]
        local = self.transformer(local)
        feat = local.mean(dim=1)
        feat = F.relu(self.fc(feat))
        projection = self.projection(feat)
        if return_projection:
            return feat, projection
        return projection


class Classifier(nn.Module):
    def __init__(self, encoder, hp):
        super().__init__()
        self.encoder = encoder
        self.use_bce = hp.USE_BCE_LOSS
        output_dim = 1 if self.use_bce else hp.NUM_CLASSES
        input_dim = hp.UNIFIED_DIM + hp.PROJECTION_OUT_DIM
        self.classifier = build_mlp(
            input_dim=input_dim,
            hidden_dims=hp.CLASSIFIER_HEAD_DIMS,
            output_dim=output_dim,
            dropout=hp.CLASSIFIER_HEAD_DROPOUT,
            use_bn=False
        )

    def forward(self, x, seq_strs=None, return_features=False):
        feat, projection = self.encoder(x, seq_strs=seq_strs, return_projection=True)
        combined = torch.cat([feat, projection], dim=1)
        logits = self.classifier(combined)
        if return_features:
            return logits, combined
        return logits


# ====================================================================
# ★ Loss function (SupCon) ★
# ====================================================================

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, pos_margin=0.5, neg_margin=0.4,
                 pos_penalty=0.1, neg_penalty=0.1):
        super().__init__()
        self.temperature = temperature
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_penalty = pos_penalty
        self.neg_penalty = neg_penalty

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T)
        sim_temp = sim / self.temperature

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask - torch.eye(mask.size(0), device=device)

        logits = sim_temp - torch.max(sim_temp, dim=1, keepdim=True)[0].detach()
        exp_logits = torch.exp(logits)
        sum_exp = exp_logits.sum(1, keepdim=True) - exp_logits.diag().view(-1, 1)
        pos_exp = (mask * exp_logits).sum(1)
        supcon_loss = -torch.log(pos_exp / (sum_exp + 1e-12) + 1e-12).mean()

        pos_sim = sim * mask
        pos_count = mask.sum(1, keepdim=True).clamp(min=1)
        pos_mean = (pos_sim.sum(1, keepdim=True)) / pos_count
        pos_loss = torch.relu(self.pos_margin - pos_mean).mean() * self.pos_penalty

        neg_mask = 1 - mask - torch.eye(mask.size(0), device=device)
        neg_sim = sim * neg_mask
        neg_count = neg_mask.sum(1, keepdim=True).clamp(min=1)
        neg_mean = (neg_sim.sum(1, keepdim=True)) / neg_count
        neg_loss = torch.relu(neg_mean - self.neg_margin).mean() * self.neg_penalty

        return supcon_loss + pos_loss + neg_loss


# ====================================================================
# ★ Training functions ★
# ====================================================================

def train_stage1(model, train_loader, val_loader, hp, device):
    criterion = nn.BCEWithLogitsLoss() if hp.USE_BCE_LOSS else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.STAGE1_LR, weight_decay=1e-5)
    best_auc = 0.0
    patience = 0
    best_state = None
    for epoch in range(hp.STAGE1_EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            if len(batch) == 3:
                x, seq_strs, labels = batch
            else:
                x, labels = batch
                seq_strs = None
            x = x.to(device)
            if hp.USE_BCE_LOSS:
                labels = labels.float().unsqueeze(1).to(device)
            else:
                labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(x, seq_strs=seq_strs)
            loss = criterion(logits.squeeze(1), labels.squeeze(1)) if hp.USE_BCE_LOSS else criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        val_metrics, _, _, _, _ = evaluate(model, val_loader, criterion, device, hp)
        print(f"  Stage1 Epoch {epoch + 1}: Loss={total_loss / len(train_loader):.4f}, Val AUC={val_metrics['auc']:.4f}")
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= hp.STAGE1_EARLY_STOP or val_metrics['auc'] >= hp.STAGE1_TARGET_AUC:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return best_auc


def train_stage2(encoder, train_loader, val_loader, hp, device):
    """Stage 2: SupCon pre-training with warmup (skip first N epochs for best tracking)."""
    supcon_loss = SupConLoss(
        temperature=hp.SUPCON_TEMPERATURE,
        pos_margin=hp.POS_MARGIN, neg_margin=hp.NEG_MARGIN,
        pos_penalty=hp.POS_PENALTY, neg_penalty=hp.NEG_PENALTY
    )
    optimizer = torch.optim.Adam(encoder.parameters(), lr=hp.STAGE2_LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    best_sep = -float('inf')
    patience = 0
    best_state = None

    # ★ FIX: warmup，前 STAGE2_MIN_EPOCHS 轮不参与 best 追踪
    warmup_epochs = getattr(hp, 'STAGE2_MIN_EPOCHS', 5)

    for epoch in range(hp.STAGE2_EPOCHS):
        encoder.train()
        total_loss = 0
        for batch in train_loader:
            if len(batch) == 3:
                x, seq_strs, labels = batch
            else:
                x, labels = batch
                seq_strs = None
            x = x.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            proj = encoder(x, seq_strs=seq_strs, return_projection=False)
            loss = supcon_loss(proj, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        sep, pos_sim, neg_sim = compute_separation(encoder, val_loader, device)

        # ★ FIX: warmup 期间只打印，不更新 best / patience
        if epoch + 1 < warmup_epochs:
            print(f"  Stage2 Epoch {epoch + 1}: Loss={avg_loss:.4f}, Separation={sep:.4f} "
                  f"(warmup, pos={pos_sim:.4f}, neg={neg_sim:.4f})")
            continue

        print(f"  Stage2 Epoch {epoch + 1}: Loss={avg_loss:.4f}, Separation={sep:.4f} "
              f"(pos={pos_sim:.4f}, neg={neg_sim:.4f})")
        if sep > best_sep:
            best_sep = sep
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
        else:
            patience += 1
        if patience >= hp.STAGE2_EARLY_STOP:
            print(f"Early stopping at epoch {epoch + 1} with best separation = {best_sep:.4f}")
            break
        scheduler.step(sep)
    if best_state is not None:
        encoder.load_state_dict(best_state)
    encoder.to(device)
    return best_sep


def train_stage3(model, train_loader, val_loader, hp, device):
    criterion = nn.BCEWithLogitsLoss() if hp.USE_BCE_LOSS else nn.CrossEntropyLoss()
    if hp.STAGE3_FREEZE_ENCODER:
        for param in model.encoder.parameters():
            param.requires_grad = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.STAGE3_LR, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    best_auc = 0.0
    patience = 0
    best_state = None
    for epoch in range(hp.STAGE3_EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            if len(batch) == 3:
                x, seq_strs, labels = batch
            else:
                x, labels = batch
                seq_strs = None
            x = x.to(device)
            if hp.USE_BCE_LOSS:
                labels = labels.float().unsqueeze(1).to(device)
            else:
                labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(x, seq_strs=seq_strs)
            loss = criterion(logits.squeeze(1), labels.squeeze(1)) if hp.USE_BCE_LOSS else criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        val_metrics, _, _, _, _ = evaluate(model, val_loader, criterion, device, hp)
        print(f"  Stage3 Epoch {epoch + 1}: Loss={total_loss / len(train_loader):.4f}, Val AUC={val_metrics['auc']:.4f}")
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= hp.STAGE3_EARLY_STOP:
            break
        scheduler.step(val_metrics['auc'])
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return best_auc


def evaluate(model, dataloader, criterion, device, hp):
    model.eval()
    total_loss = 0
    all_preds, all_probs, all_labels = [], [], []
    all_features = []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x, seq_strs, labels = batch
            else:
                x, labels = batch
                seq_strs = None
            x = x.to(device)
            labels_orig = labels.cpu().numpy()
            labels_f = labels.float().unsqueeze(1).to(device) if hp.USE_BCE_LOSS else labels.to(device)
            logits, features = model(x, seq_strs=seq_strs, return_features=True)
            if hp.USE_BCE_LOSS:
                loss = criterion(logits.squeeze(1), labels_f.squeeze(1))
                probs = torch.sigmoid(logits.squeeze(1))
                preds = (probs > 0.5).int()
                probs_np = probs.cpu().numpy()
            else:
                loss = criterion(logits, labels_f)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                probs_np = probs.cpu().numpy()[:, 1]
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs_np)
            all_labels.extend(labels_orig)
            all_features.append(features.cpu().float().numpy())
    features = np.vstack(all_features) if all_features else np.array([])
    metrics = compute_all_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(dataloader)
    return metrics, features, all_labels, all_preds, all_probs


def compute_separation(encoder, dataloader, device, num_samples=2000):
    """★ FIX: 排除对角线（自己与自己相似度=1），避免 separation 虚高。"""
    encoder.eval()
    all_proj, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x, seq_strs, labels = batch
            else:
                x, labels = batch
                seq_strs = None
            x = x.to(device)
            proj = encoder(x, seq_strs=seq_strs, return_projection=False)
            all_proj.append(proj.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if len(np.vstack(all_proj)) >= num_samples:
                break
    if not all_proj:
        return 0.0, 0.0, 0.0
    all_proj = np.vstack(all_proj)
    all_labels = np.array(all_labels)
    mask = (all_labels == 0) | (all_labels == 1)
    if mask.sum() < 2:
        return 0.0, 0.0, 0.0
    proj = all_proj[mask]
    lab = all_labels[mask]

    # ★ FIX 核心：如果验证集只有单一类别，separation 无意义，直接返回 0
    unique_labels = np.unique(lab)
    if len(unique_labels) < 2:
        return 0.0, 0.0, 0.0

    proj_t = torch.tensor(proj, dtype=torch.float32)
    lab_t = torch.tensor(lab, dtype=torch.long)
    sim = F.cosine_similarity(proj_t.unsqueeze(1), proj_t.unsqueeze(0), dim=2)

    # ★ FIX: 排除对角线，pos_mask 不再包含自己与自己
    eye = torch.eye(len(lab), dtype=torch.bool, device=sim.device)
    same_label = torch.eq(lab_t.unsqueeze(1), lab_t.unsqueeze(0))
    pos_mask = same_label & (~eye)
    neg_mask = (~same_label) & (~eye)

    pos_sim = sim[pos_mask].mean().item() if pos_mask.sum() > 0 else 0.0
    neg_sim = sim[neg_mask].mean().item() if neg_mask.sum() > 0 else 0.0
    return pos_sim - neg_sim, pos_sim, neg_sim


# ====================================================================
# ★ Data loading ★
# ====================================================================

def load_aa_embeddings(csv_path):
    aa_order = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    aa_to_idx = {aa: i for i, aa in enumerate(aa_order)}
    pretrained = np.zeros((21, 64), dtype=np.float32)
    for i in range(20):
        pretrained[i] = np.random.randn(64).astype(np.float32) * 0.01
    pretrained[20] = np.zeros(64, dtype=np.float32)
    return aa_to_idx, pretrained


def load_data_from_xlsx(data_path, seq_col, label_col, pos_col, seq_length,
                        pad_n='*', pad_c='#', position_is_one_based=False):
    df = pd.read_excel(data_path, engine='openpyxl')
    sequences = df[seq_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    positions = df[pos_col].astype(int).tolist()
    half = seq_length // 2
    window_sequences = []
    for seq, pos in zip(sequences, positions):
        if position_is_one_based:
            pos = pos - 1
        start = pos - half
        end = pos + half + 1
        window_chars = []
        for i in range(start, end):
            if i < 0 or i >= len(seq):
                window_chars.append(pad_n if i < 0 else pad_c)
            else:
                window_chars.append(seq[i])
        window_sequences.append(''.join(window_chars))
    return window_sequences, labels


# ====================================================================
# ★ t-SNE and pLOGO helper functions ★
# ====================================================================

def perform_tsne_and_clustering(encoder, dataloader, device, hp, n_clusters=4, random_state=42):
    encoder.eval()
    all_feats, all_labels, all_seqs = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x, seq_strs, labels = batch
            else:
                x, labels = batch
                seq_strs = None
            x = x.to(device)
            proj = encoder(x, seq_strs=seq_strs, return_projection=False)
            all_feats.append(proj.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if seq_strs is not None:
                all_seqs.extend(seq_strs)
            else:
                all_seqs.extend([''] * len(labels))
    feats = np.vstack(all_feats)
    labels = np.array(all_labels)

    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30, n_iter=1000, init='pca')
    coords = tsne.fit_transform(feats)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(feats)

    return feats, labels, coords, cluster_labels, all_seqs


def draw_sequence_logo(seqs, output_path, title='Sequence Logo', max_len=None):
    if max_len is None:
        max_len = len(seqs[0])
    aa_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    counts = np.zeros((max_len, len(aa_list)))
    for seq in seqs:
        seq = seq[:max_len]
        for pos, aa in enumerate(seq):
            if aa in aa_list:
                counts[pos, aa_list.index(aa)] += 1
    df = pd.DataFrame(counts, columns=aa_list)
    df = df.div(df.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    logo = logomaker.Logo(df, ax=ax)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    logo.style_spines(spines=['top', 'right'], visible=False)
    ax.set_title(title)
    ax.set_xlabel('Position')
    ax.set_ylabel('Probability')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ====================================================================
# ★ Main program ★
# ====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no_stage1', action='store_true')
    parser.add_argument('--no_stage2', action='store_true')
    parser.add_argument('--no_stage3', action='store_true')
    args = parser.parse_args()

    hp = HyperParams()
    if args.train_path: hp.TRAIN_DATA_PATH = args.train_path
    if args.test_path: hp.TEST_DATA_PATH = args.test_path
    if args.output_dir: hp.OUTPUT_DIR = args.output_dir
    if args.n_folds: hp.N_FOLDS = args.n_folds
    if args.seed is not None: hp.SEED = args.seed
    if args.no_stage1: hp.USE_STAGE1 = False
    if args.no_stage2: hp.USE_STAGE2 = False
    if args.no_stage3: hp.USE_STAGE3 = False

    hp.print_summary()
    seed_everything(hp.SEED)

    os.makedirs(hp.OUTPUT_DIR, exist_ok=True)
    plots_dir = os.path.join(hp.OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    with open(os.path.join(hp.OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(hp.to_dict(), f, indent=2)

    # ---- Load data ----
    print("\n🔍 Loading data...")
    aa_to_idx, pretrained_embeddings = load_aa_embeddings(hp.AA_EMB_PATH)
    train_sequences, train_labels = load_data_from_xlsx(
        hp.TRAIN_DATA_PATH, hp.SEQ_COLUMN, hp.LABEL_COLUMN, hp.POS_COLUMN, hp.SEQ_LENGTH
    )
    test_sequences, test_labels = load_data_from_xlsx(
        hp.TEST_DATA_PATH, hp.SEQ_COLUMN, hp.LABEL_COLUMN, hp.POS_COLUMN, hp.SEQ_LENGTH
    )
    print(f"Training set: {len(train_sequences)} samples")
    print(f"Test set: {len(test_sequences)} samples")

    # ★ FIX: 打印标签分布，方便诊断
    print(f"[DIAG] train label distribution: {np.bincount(np.array(train_labels))}")
    print(f"[DIAG] test  label distribution: {np.bincount(np.array(test_labels))}")

    # ---- Step 1: 5-fold cross-validation ----
    print("\n" + "=" * 70)
    print("Step 1: 5-fold cross-validation")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=hp.N_FOLDS, shuffle=True, random_state=hp.SEED)
    fold_aucs = []
    fold_metrics_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_sequences, train_labels)):
        seed_everything(hp.SEED + fold)
        print(f"\nFold {fold + 1}/{hp.N_FOLDS}")
        train_seq_fold = [train_sequences[i] for i in train_idx]
        train_lab_fold = [train_labels[i] for i in train_idx]
        val_seq_fold = [train_sequences[i] for i in val_idx]
        val_lab_fold = [train_labels[i] for i in val_idx]

        train_dataset = AAIndexDataset(train_seq_fold, train_lab_fold, aa_to_idx, hp.SEQ_LENGTH, return_str=True)
        val_dataset = AAIndexDataset(val_seq_fold, val_lab_fold, aa_to_idx, hp.SEQ_LENGTH, return_str=True)
        train_loader = make_loader(train_dataset, hp.STAGE1_BATCH_SIZE, True, hp.SEED + fold, hp.NUM_WORKERS)
        val_loader = make_loader(val_dataset, hp.VAL_BATCH_SIZE, False, hp.SEED + fold, hp.NUM_WORKERS)

        encoder = TransformerEncoder(hp).to(DEVICE)
        model = Classifier(encoder, hp).to(DEVICE)

        if hp.USE_STAGE1:
            print("  Stage 1: BCE warm-up")
            train_stage1(model, train_loader, val_loader, hp, DEVICE)
        if hp.USE_STAGE2:
            print("  Stage 2: SupCon")
            classifier_state = model.classifier.state_dict()
            train_stage2(encoder, train_loader, val_loader, hp, DEVICE)
            model = Classifier(encoder, hp).to(DEVICE)
            model.classifier.load_state_dict(classifier_state)
        if hp.USE_STAGE3:
            print("  Stage 3: End-to-end fine-tuning")
            train_stage3(model, train_loader, val_loader, hp, DEVICE)

        val_metrics, _, val_labels_np, val_preds, val_probs = evaluate(
            model, val_loader, nn.BCEWithLogitsLoss(), DEVICE, hp
        )
        fold_aucs.append(val_metrics['auc'])
        fold_metrics_list.append({
            'Fold': fold + 1,
            'Acc': val_metrics['acc'], 'Rec': val_metrics['recall'],
            'Pre': val_metrics['precision'], 'MCC': val_metrics['mcc'],
            'F1': val_metrics['f1'], 'AUC': val_metrics['auc'],
            'AUCPR': val_metrics['auprc'], 'Sp': val_metrics['specificity'],
        })
        print(f"  Fold {fold + 1}: AUC={val_metrics['auc']:.4f}  Acc={val_metrics['acc']:.4f}  "
              f"Rec={val_metrics['recall']:.4f}  Pre={val_metrics['precision']:.4f}  "
              f"F1={val_metrics['f1']:.4f}  MCC={val_metrics['mcc']:.4f}  "
              f"AUCPR={val_metrics['auprc']:.4f}  Sp={val_metrics['specificity']:.4f}")

        if hp.SAVE_PLOTS:
            fold_plot_dir = os.path.join(plots_dir, f'fold_{fold + 1}')
            os.makedirs(fold_plot_dir, exist_ok=True)
            fpr, tpr, _ = roc_curve(val_labels_np, val_probs)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {val_metrics["auc"]:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=1)
            plt.xlim([0, 1]); plt.ylim([0, 1.05])
            plt.xlabel('FPR'); plt.ylabel('TPR')
            plt.title(f'ROC - Fold {fold + 1}')
            plt.legend(loc='lower right'); plt.grid(alpha=0.3)
            plt.tight_layout(); plt.savefig(os.path.join(fold_plot_dir, 'roc_curve.png'), dpi=300); plt.close()

            precision, recall, _ = precision_recall_curve(val_labels_np, val_probs)
            plt.figure(figsize=(6, 5))
            plt.plot(recall, precision, color='green', lw=2, label=f'PR (AUPRC = {val_metrics["auprc"]:.4f})')
            plt.xlim([0, 1]); plt.ylim([0, 1.05])
            plt.xlabel('Recall'); plt.ylabel('Precision')
            plt.title(f'Precision-Recall - Fold {fold + 1}')
            plt.legend(loc='lower left'); plt.grid(alpha=0.3)
            plt.tight_layout(); plt.savefig(os.path.join(fold_plot_dir, 'pr_curve.png'), dpi=300); plt.close()

    # ---- CV summary ----
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    fold_df = pd.DataFrame(fold_metrics_list)
    metric_cols = ['Acc', 'Rec', 'Pre', 'MCC', 'F1', 'AUC', 'AUCPR', 'Sp']

    print("\n" + "=" * 100)
    print("Per-fold CV metrics")
    print("=" * 100)
    print(fold_df[['Fold'] + metric_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    summary_rows = []
    for col in metric_cols:
        vals = fold_df[col].values
        summary_rows.append({
            'Metric': col, 'Mean': float(np.mean(vals)), 'Std': float(np.std(vals)),
            'Mean±Std': f"{np.mean(vals):.4f} ± {np.std(vals):.4f}",
        })
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 100)
    print("CV summary (mean ± std over folds)")
    print("=" * 100)
    print(summary_df[['Metric', 'Mean±Std']].to_string(index=False))

    fold_df.to_csv(os.path.join(hp.OUTPUT_DIR, 'cv_fold_metrics.csv'), index=False)
    summary_df.to_csv(os.path.join(hp.OUTPUT_DIR, 'cv_summary_metrics.csv'), index=False)
    cv_results = {
        'fold_aucs': fold_aucs, 'mean_auc': mean_auc, 'std_auc': std_auc,
        'fold_metrics': fold_metrics_list, 'summary_metrics': summary_rows,
    }
    with open(os.path.join(hp.OUTPUT_DIR, 'cv_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)

    # ====================================================================
    # ---- Step 2: Train final model on full training set ----
    # ★★★ FIX 核心：用 train_test_split(stratify=...) 替代 [-val_size:] ★★★
    # ====================================================================
    print("\n" + "=" * 70)
    print("Step 2: Train final model on full training set")
    print("=" * 70)

    seed_everything(hp.SEED)

    # ★ FIX: 分层划分，保证 val 集正负样本都存在
    train_seq_full, val_seq_full, train_lab_full, val_lab_full = train_test_split(
        train_sequences, train_labels,
        test_size=0.1,
        stratify=train_labels,
        random_state=hp.SEED,
    )
    print(f"[DIAG] Step2 train label distribution: {np.bincount(np.array(train_lab_full))}")
    print(f"[DIAG] Step2 val   label distribution: {np.bincount(np.array(val_lab_full))}")

    if len(np.unique(val_lab_full)) < 2:
        raise RuntimeError(
            "Step2 val 集只有单一类别！请检查 train.xlsx 的 label 分布，"
            "或调整 stratify / test_size 参数。"
        )

    train_dataset_full = AAIndexDataset(train_seq_full, train_lab_full, aa_to_idx, hp.SEQ_LENGTH, return_str=True)
    val_dataset_full = AAIndexDataset(val_seq_full, val_lab_full, aa_to_idx, hp.SEQ_LENGTH, return_str=True)
    train_loader_full = make_loader(train_dataset_full, hp.STAGE1_BATCH_SIZE, True, hp.SEED, hp.NUM_WORKERS)
    val_loader_full = make_loader(val_dataset_full, hp.VAL_BATCH_SIZE, False, hp.SEED, hp.NUM_WORKERS)

    encoder_final = TransformerEncoder(hp).to(DEVICE)
    model_final = Classifier(encoder_final, hp).to(DEVICE)

    if hp.USE_STAGE1:
        print("Stage 1: BCE warm-up")
        train_stage1(model_final, train_loader_full, val_loader_full, hp, DEVICE)
    if hp.USE_STAGE2:
        print("Stage 2: SupCon")
        classifier_state_final = model_final.classifier.state_dict()
        train_stage2(encoder_final, train_loader_full, val_loader_full, hp, DEVICE)
        model_final = Classifier(encoder_final, hp).to(DEVICE)
        model_final.classifier.load_state_dict(classifier_state_final)
    if hp.USE_STAGE3:
        print("Stage 3: End-to-end fine-tuning")
        train_stage3(model_final, train_loader_full, val_loader_full, hp, DEVICE)

    if hp.SAVE_MODELS:
        torch.save(model_final.state_dict(), os.path.join(hp.OUTPUT_DIR, 'final_model.pth'))
        torch.save(encoder_final.state_dict(), os.path.join(hp.OUTPUT_DIR, 'final_encoder.pth'))
        print("✅ Final model saved")

    # ---- Step 3: Independent test ----
    print("\n" + "=" * 70)
    print("Step 3: Evaluate final model on independent test set")
    print("=" * 70)

    test_dataset = AAIndexDataset(test_sequences, test_labels, aa_to_idx, hp.SEQ_LENGTH, return_str=True)
    test_loader = make_loader(test_dataset, hp.TEST_BATCH_SIZE, False, hp.SEED, hp.NUM_WORKERS)

    test_metrics, test_features, test_labels_np, test_preds, test_probs = evaluate(
        model_final, test_loader, nn.BCEWithLogitsLoss(), DEVICE, hp
    )

    print("\nTest set metrics:")
    for k, v in test_metrics.items():
        if k != 'loss':
            print(f"  {k.upper()}: {v:.4f}")

    test_results = {
        'test_metrics': test_metrics,
        'cv_mean_auc': mean_auc, 'cv_std_auc': std_auc,
        'config': hp.to_dict(),
        'timestamp': datetime.now().isoformat()
    }
    with open(os.path.join(hp.OUTPUT_DIR, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)

    # ---- Visualization ----
    if hp.SAVE_PLOTS:
        plt.figure(figsize=(8, 5))
        plt.bar(range(1, hp.N_FOLDS + 1), fold_aucs, color='skyblue', edgecolor='navy')
        plt.axhline(y=mean_auc, color='r', linestyle='--', label=f'Mean={mean_auc:.4f}')
        plt.xlabel('Fold'); plt.ylabel('Validation AUC'); plt.title('Cross-Validation AUC')
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(plots_dir, 'cv_auc_bars.png'), dpi=300); plt.close()

        cm = confusion_matrix(test_labels_np, test_preds)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap='Blues', interpolation='nearest'); plt.colorbar()
        plt.xticks([0, 1], ['Negative', 'Positive']); plt.yticks([0, 1], ['Negative', 'Positive'])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center')
        plt.title('Confusion Matrix (Test)'); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'confusion_matrix_test.png'), dpi=300); plt.close()

        fpr, tpr, _ = roc_curve(test_labels_np, test_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {test_metrics["auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0, 1]); plt.ylim([0, 1.05])
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC - Test Set')
        plt.legend(loc='lower right'); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'roc_curve_test.png'), dpi=300); plt.close()

        precision, recall, _ = precision_recall_curve(test_labels_np, test_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, color='green', lw=2, label=f'PR (AUPRC = {test_metrics["auprc"]:.4f})')
        plt.xlim([0, 1]); plt.ylim([0, 1.05])
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR - Test Set')
        plt.legend(loc='lower left'); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'pr_curve_test.png'), dpi=300); plt.close()

        print("\n🔬 Generating t-SNE and sequence logos...")
        all_seqs_for_tsne = train_sequences + test_sequences
        all_labels_for_tsne = train_labels + test_labels
        combined_dataset = AAIndexDataset(all_seqs_for_tsne, all_labels_for_tsne, aa_to_idx, hp.SEQ_LENGTH, return_str=True)
        combined_loader = make_loader(combined_dataset, hp.TEST_BATCH_SIZE, False, hp.SEED, hp.NUM_WORKERS)

        if len(all_seqs_for_tsne) > 5000:
            indices = np.random.choice(len(all_seqs_for_tsne), 5000, replace=False)
            sampled_seqs = [all_seqs_for_tsne[i] for i in indices]
            sampled_labels = [all_labels_for_tsne[i] for i in indices]
            sampled_dataset = AAIndexDataset(sampled_seqs, sampled_labels, aa_to_idx, hp.SEQ_LENGTH, return_str=True)
            combined_loader = make_loader(sampled_dataset, hp.TEST_BATCH_SIZE, False, hp.SEED, hp.NUM_WORKERS)

        n_clusters = 4
        feats, true_labels, tsne_coords, cluster_labels, seqs = perform_tsne_and_clustering(
            encoder_final, combined_loader, DEVICE, hp, n_clusters=n_clusters
        )

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=tsne_coords[:, 0], y=tsne_coords[:, 1], hue=true_labels, palette='Set1', alpha=0.7)
        plt.title('t-SNE by True Label'); plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2')
        plt.legend(title='Label'); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'tsne_by_true_label.png'), dpi=300); plt.close()

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=tsne_coords[:, 0], y=tsne_coords[:, 1], hue=cluster_labels, palette='tab10', alpha=0.7)
        plt.title('t-SNE by KMeans Cluster'); plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2')
        plt.legend(title='Cluster'); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'tsne_by_cluster.png'), dpi=300); plt.close()

        cluster_seqs = {i: [] for i in range(n_clusters)}
        for seq, c in zip(seqs, cluster_labels):
            cluster_seqs[c].append(seq)
        logo_dir = os.path.join(plots_dir, 'sequence_logos')
        os.makedirs(logo_dir, exist_ok=True)
        for c, seq_list in cluster_seqs.items():
            if len(seq_list) < 3:
                print(f"  Cluster {c} has too few samples ({len(seq_list)}), skipping logo")
                continue
            draw_sequence_logo(seq_list, os.path.join(logo_dir, f'cluster_{c}_logo.png'),
                               title=f'Cluster {c} Logo (n={len(seq_list)})', max_len=hp.SEQ_LENGTH)

        for label in [0, 1]:
            label_seqs = [seq for seq, lab in zip(seqs, true_labels) if lab == label]
            if len(label_seqs) > 3:
                draw_sequence_logo(label_seqs, os.path.join(logo_dir, f'label_{label}_logo.png'),
                                   title=f'Label {label} Logo (n={len(label_seqs)})', max_len=hp.SEQ_LENGTH)

    print("\n" + "=" * 70)
    print(f"✅ All done! Results saved to: {hp.OUTPUT_DIR}")
    print(f"Ablation: Stage1={hp.USE_STAGE1}, Stage2={hp.USE_STAGE2}, Stage3={hp.USE_STAGE3}")
    print(f"CV mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Test set AUC: {test_metrics['auc']:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
