import argparse
import gc
import os
import glob
import pickle
import random
import re
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

from transformers import AutoTokenizer, AutoModel

def parse_args():
    parser = argparse.ArgumentParser(description="Predict using Translation-Prophet model")

    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use for training (default: 0)")
    parser.add_argument("--protT5_dir", type=str, required=True, help="Path to ProtT5 model directory")
    parser.add_argument("--syncodonlm_dir", type=str, required=True, help="Path to SynCodonLM model directory")
    parser.add_argument("--model_file", type=str, required=True, help="Trained Translation-Prophet model checkpoint")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save prediction results")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for prediction (default: 128)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (default: 0)")

    return parser.parse_args()

def numerical_sort_key(path):
    match = re.search(r"(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else -1

def load_all_embeddings(embedding_dir):
    embedding_files = sorted(
        glob.glob(os.path.join(embedding_dir, "embeddings_batch_*.pkl")),
        key=numerical_sort_key
    )
    all_embeddings, all_labels, all_ids = [], [], []
    for f in embedding_files:
        with open(f, 'rb') as fin:
            data = pickle.load(fin)
            emb = data['embeddings']
            all_embeddings.append(emb)
            all_labels.append(data['labels'])
            all_ids.append(data['ids'])
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    ids = np.concatenate(all_ids)
    print(f"{embedding_dir} loaded: {embeddings.shape}")
    return embeddings, labels, ids

class Linear_DimReducer(nn.Module):
    def __init__(self, input_dim, hidden_dim1=512):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim1)
        self.ln = nn.LayerNorm(hidden_dim1)

    def forward(self, x):
        with torch.no_grad():
            out = self.proj(x)
            out = self.ln(out)
        return out
    
def load_reducer(checkpoint_path, name, input_dim, hidden_dim1, device="cuda"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    reducer = Linear_DimReducer(input_dim, hidden_dim1).to(device)
    reducer.load_state_dict(ckpt[name])
    reducer.eval()
    return reducer

def reduce_with_dimreducer(emb_np, name, input_dim, hidden_dim1,
                           checkpoint_path, device="cuda", batch_size=8):
    reducer = load_reducer(checkpoint_path, name, input_dim, hidden_dim1, device)
    X = torch.from_numpy(emb_np).float()
    reduced_batches = []
    reducer.eval()
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            batch = X[i:i+batch_size].to(device, non_blocking=True)
            reduced = reducer(batch).cpu()
            reduced_batches.append(reduced)
            del batch
            torch.cuda.empty_cache()
    del X, reducer
    torch.cuda.empty_cache()

    return torch.cat(reduced_batches, dim=0)

class Trainable_Encoder(nn.Module):
    def __init__(self, in_dim, lstm_hidden_dim2=128, cnn_num_filters=128, dropout=0.5):
        super().__init__()
        self.lstm2 = nn.LSTM(
            input_size=in_dim,
            hidden_size=lstm_hidden_dim2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        out_dim_lstm2 = lstm_hidden_dim2 * 2
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=out_dim_lstm2, out_channels=cnn_num_filters, kernel_size=k)
            for k in [3, 6, 9]
        ])
        self.dropout = nn.Dropout(dropout)
        self.out_dim = cnn_num_filters * len(self.convs)

    def forward(self, x):
        y, _ = self.lstm2(x)
        y = y.transpose(1, 2)
        feats = [F.relu(conv(y)) for conv in self.convs]
        pooled = [F.max_pool1d(f, kernel_size=f.size(2)).squeeze(2) for f in feats]
        out = torch.cat(pooled, dim=1)
        out = self.dropout(out)
        return out

class MultiPathReducedDataset(Dataset):
    def __init__(self, reduced_list_np, labels_np):
        self.reduced_list = reduced_list_np
        self.labels = labels_np.astype(np.int64)
        self.n = self.labels.shape[0]
        for arr in self.reduced_list:
            assert arr.shape[0] == self.n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        xs = [torch.from_numpy(arr[idx]).float() for arr in self.reduced_list]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return xs, y

def collate_paths(batch):
    num_paths = len(batch[0][0])
    out_paths = []
    for p in range(num_paths):
        tensors = [item[0][p] for item in batch]
        out_paths.append(torch.stack(tensors, dim=0))
    labels = torch.stack([item[1] for item in batch], dim=0)
    return out_paths, labels

class GatedFusionNet(nn.Module):
    def __init__(self, in_dims_per_path, lstm_hidden_dim2=128, cnn_num_filters=128,
                 num_classes=2, dropout=0.5):
        super().__init__()
        self.num_paths = len(in_dims_per_path)
        self.encoders = nn.ModuleList([
            Trainable_Encoder(in_dim=d, lstm_hidden_dim2=lstm_hidden_dim2,
                              cnn_num_filters=cnn_num_filters, dropout=dropout)
            for d in in_dims_per_path
        ])
        self.gates = nn.Parameter(torch.ones(self.num_paths, dtype=torch.float32))
        fused_dim = sum(enc.out_dim for enc in self.encoders)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fused_dim, num_classes)

    def forward(self, x_list):
        outs = []
        for i, x in enumerate(x_list):
            feat = self.encoders[i](x)
            gate = torch.sigmoid(self.gates[i])
            outs.append(feat * gate)
        fused = torch.cat(outs, dim=1)          
        fused = self.dropout(fused)
        logits = self.fc(fused)
        return logits
 

def main():
    args = parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = args.batch_size
    
    protT5_dir = args.protT5_dir
    syncodonlm_dir = args.syncodonlm_dir
    model_file = args.model_file
    out_dir = args.out_dir
    

    os.makedirs(out_dir, exist_ok=True)

    hidden_dim1_per_path = {
        "protT5": 512,
        "syncodonlm": 512
    }

    emb1, labels1, ids1 = load_all_embeddings(protT5_dir)
    emb2, labels2, ids2 = load_all_embeddings(syncodonlm_dir)
    assert np.all(ids1 == ids2) , "IDs are inconsistent!"
    

    emb1_red = reduce_with_dimreducer(
        emb1,
        name="protT5",
        input_dim=emb1.shape[-1],
        hidden_dim1=hidden_dim1_per_path["protT5"],
        checkpoint_path=f"model/dim_reducers.pth",
        device=device,
        batch_size=8
    )
    del emb1; gc.collect()

    emb2_red = reduce_with_dimreducer(
        emb2,
        name="syncodonml",
        input_dim=emb2.shape[-1],
        hidden_dim1=hidden_dim1_per_path["syncodonlm"],
        checkpoint_path=f"model/dim_reducers.pth",
        device=device,
        batch_size=8
    )
    del emb2; gc.collect()

    labels = labels1
    
    dataset = MultiPathReducedDataset(
        reduced_list_np=[emb1_red.numpy(), emb2_red.numpy()],
        labels_np=labels
    )
    
    dataset_loader  = DataLoader(dataset , batch_size=batch_size, shuffle=False)
    
    in_dims_per_path = [
        hidden_dim1_per_path["protT5"],
        hidden_dim1_per_path["syncodonlm"],
    ]

    model = GatedFusionNet(
        in_dims_per_path,
        lstm_hidden_dim2=128,
        cnn_num_filters=128,
        num_classes=2,
        dropout=0.5
    ).to(device)

    state_dict = torch.load(model_file, map_location=device)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in dataset_loader:
            batch_x = [x.to(device) for x in batch_x]
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)

    TP = np.sum((all_labels == 1) & (all_preds == 1))
    TN = np.sum((all_labels == 0) & (all_preds == 0))
    FP = np.sum((all_labels == 0) & (all_preds == 1))
    FN = np.sum((all_labels == 1) & (all_preds == 0))

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    print(f"AUC: {auc:.4f}, ACC: {acc:.4f}, "
        f"TPR: {TPR:.4f}, TNR: {TNR:.4f}, FPR: {FPR:.4f}, FNR: {FNR:.4f}")

    results = {
        "AUC": auc,
        "ACC": acc,
        "TPR": TPR,
        "TNR": TNR,
        "FPR": FPR,
        "FNR": FNR,
    }
    
    def summarize(metric):
        values = [r[metric] for r in all_results]
        return np.mean(values), np.std(values)

    pd.DataFrame([results]).to_csv(
        os.path.join(out_dir, "results.csv"),
        index=False
    )
    print(results)
        

if __name__ == "__main__":
    main()
