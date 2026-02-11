import argparse
import gc
import glob
import os
import pickle
import random
import re
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

from transformers import AutoTokenizer, AutoModel

from captum.attr import GradientShap

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Interpret Translation-Prophet with attribution methods")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use for computation (default: 0)")
    parser.add_argument("--protT5_dir", type=str, required=True, help="Directory of ProtT5 embeddings")
    parser.add_argument("--syncodonlm_dir", type=str, required=True, help="Directory of SynCodonLM embeddings")
    parser.add_argument("--model_file", type=str, required=True, help="Trained Translation-Prophet model checkpoint")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save interpretation results")
    
    parser.add_argument(
        "--baseline_protT5_dir",
        type=str,
        default=None,
        help="Directory of baseline ProtT5 embeddings for GradientShap attribution"
    )

    parser.add_argument(
        "--baseline_syncodonlm_dir",
        type=str,
        default=None,
        help="Directory of baseline SynCodonLM embeddings for GradientShap attribution"
    )

    parser.add_argument(
        "--baseline_batch_size",
        type=int,
        default=None,
        help="Batch size for computing baseline embeddings in GradientShap"
    )

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for interpretation computation (default: 1)")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length for tokenization (default: 1024)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (default: 0)")

    return parser.parse_args()



def numerical_sort_key(path):
    match = re.search(r"(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else -1

def load_all_embeddings(embedding_dir, target_dim=None):
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

def build_reduced_dataset(protT5_dir, syncodonlm_dir, dim_reducer_ckpt, device, hidden_dim1=512, batch_size=8):
    emb1, labels1, ids1 = load_all_embeddings(protT5_dir)
    emb1 = np.ascontiguousarray(emb1, dtype=np.float32)

    _, labels2, ids2 = load_all_embeddings(syncodonlm_dir)
    assert np.all(ids1 == ids2), "IDs are inconsistent between ProtT5 and SyncodonLM!"

    emb1_red = reduce_with_dimreducer(
        emb1,
        name="protT5",
        input_dim=emb1.shape[-1],
        hidden_dim1=hidden_dim1,
        checkpoint_path=dim_reducer_ckpt,
        device=device,
        batch_size=batch_size
    )

    emb2, _, _ = load_all_embeddings(syncodonlm_dir)
    emb2 = np.ascontiguousarray(emb2, dtype=np.float32)

    emb2_red = reduce_with_dimreducer(
        emb2,
        name="syncodonml",
        input_dim=emb2.shape[-1],
        hidden_dim1=hidden_dim1,
        checkpoint_path=dim_reducer_ckpt,
        device=device,
        batch_size=batch_size
    )

    dataset = MultiPathReducedDataset(
        reduced_list_np=[emb1_red.numpy(), emb2_red.numpy()],
        labels_np=labels1,
        ids=ids1
    )

    return dataset

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

def reduce_with_dimreducer(emb_np, name, input_dim, hidden_dim1,
                           checkpoint_path, device="cuda", batch_size=8):
    reducer = load_reducer(checkpoint_path, name, input_dim, hidden_dim1, device)
    X = torch.from_numpy(emb_np).float()
    reduced_batches = []

    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            batch = X[i:i+batch_size].to(device, non_blocking=True)
            reduced = reducer(batch).cpu()
            reduced_batches.append(reduced)
            del batch
            torch.cuda.empty_cache()

    return torch.cat(reduced_batches, dim=0)


def load_reducer(checkpoint_path, name, input_dim, hidden_dim1, device="cuda"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    reducer = Linear_DimReducer(input_dim, hidden_dim1).to(device)
    reducer.load_state_dict(ckpt[name])
    reducer.eval()
    return reducer

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
            nn.Conv1d(in_channels=out_dim_lstm2,
                      out_channels=cnn_num_filters,
                      kernel_size=k)
            for k in [3, 6, 9]
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.out_dim = cnn_num_filters * len(self.convs)

    def forward(self, x, return_token_feat=False):
        y, _ = self.lstm2(x)
        token_feat = y    
        y = y.transpose(1, 2)
        feats = [F.relu(conv(y)) for conv in self.convs]
        pooled = [F.max_pool1d(f, kernel_size=f.size(2)).squeeze(2) for f in feats]
        out = torch.cat(pooled, dim=1)
        out = self.dropout(out)
        
        if return_token_feat:
            return out, token_feat
        return out
    
class MultiPathReducedDataset(Dataset):
    def __init__(self, reduced_list_np, labels_np, ids=None):
        self.reduced_list = reduced_list_np
        self.labels = labels_np.astype(np.int64)
        self.n = self.labels.shape[0]

        if ids is None:
            self.ids = np.arange(self.n)
        else:
            assert len(ids) == self.n
            self.ids = ids

        for arr in self.reduced_list:
            assert arr.shape[0] == self.n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        xs = [torch.from_numpy(arr[idx]).float() for arr in self.reduced_list]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        id_ = self.ids[idx]
        return xs, y, id_

def collate_paths(batch):
    num_paths = len(batch[0][0])
    out_paths = []
    for p in range(num_paths):
        tensors = [item[0][p] for item in batch]
        out_paths.append(torch.stack(tensors, dim=0))
    labels = torch.stack([item[1] for item in batch], dim=0)
    ids = [item[2] for item in batch]
    return out_paths, labels, ids

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

    def forward_with_token_feat(self, x_list):
        outs = []
        token_feats_weighted = []
        for i, x in enumerate(x_list):
            feat, token_feat = self.encoders[i](x, return_token_feat=True)
            gate = torch.sigmoid(self.gates[i])
            outs.append(feat * gate)
            token_feats_weighted.append(token_feat * gate)
        fused = torch.cat(outs, dim=1)
        fused_token_feat = torch.cat(token_feats_weighted, dim=2)
        fused = self.dropout(fused)
        logits = self.fc(fused)
        return logits, fused_token_feat
 
class TestDataset(Dataset):
    def __init__(self, original_dataset, indices):
        self.data = [original_dataset[i] for i in indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] 
    
import __main__
__main__.TestDataset = TestDataset
   
def compute_token_attribution(model, dataset, device="cuda", collate_fn=None, num_samples=5):
    
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=50, shuffle=True, collate_fn=collate_fn)
    bg_x_list, _, _ = next(iter(loader))
    bg_x_list = [x.to(device) for x in bg_x_list]
    
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *x_list):
            logits = self.model(list(x_list))
            probs = torch.softmax(logits, dim=-1)
            return probs

    wrapped_model = ModelWrapper(model)
    gradient_shap = GradientShap(wrapped_model)

    token_shap_dict = {}
    label0_token_shap_dict = {}
    label1_token_shap_dict = {}
    label_dict = {}
    label0_vals, label1_vals = [], []

    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for x_list, y, ids in test_loader:
        x_tensor_tuple = tuple(x.to(device) for x in x_list)
        bg_x_tuple = tuple(bg.to(device) for bg in bg_x_list)

        seq_id = ids[0]
        label = y.item()
        label_dict[seq_id] = label

        wrapped_model.train()  

        sampled_attrs = []
        for _ in range(num_samples):
            with torch.no_grad():
                attributions = gradient_shap.attribute(
                    inputs=x_tensor_tuple,
                    baselines=bg_x_tuple,
                    target=label
                )

            token_shap = torch.cat([attr.sum(dim=2) for attr in attributions], dim=1).squeeze(0).cpu().numpy()
            sampled_attrs.append(token_shap)

        token_shap = np.mean(sampled_attrs, axis=0)

        token_shap_dict[seq_id] = token_shap
        if label == 0:
            label0_vals.append(token_shap)
            label0_token_shap_dict[seq_id] = token_shap
        else:
            label1_vals.append(token_shap)
            label1_token_shap_dict[seq_id] = token_shap

    label0_mean = np.mean(label0_vals, axis=0) if label0_vals else None
    label1_mean = np.mean(label1_vals, axis=0) if label1_vals else None

    return (
        label0_mean,
        label1_mean,
        token_shap_dict,
        label0_token_shap_dict,
        label1_token_shap_dict,
        label_dict
    )

def plot_token_shap_signed_bar_by_seq(label0_mean, label1_mean,
                                      protT5_len=1024, syncodon_len=1024,
                                      out_file="token_shap_signed_bar.pdf"):
    
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    
    protT5_label0 = label0_mean[:protT5_len]
    protT5_label1 = label1_mean[:protT5_len]
    syncodon_label0 = label0_mean[protT5_len:]
    syncodon_label1 = label1_mean[protT5_len:]

    x_protT5 = np.arange(1, protT5_len + 1)
    x_syncodon = np.arange(1, syncodon_len + 1)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharey=False)

    # ===== Top panel: ProtT5 =====
    axes[0].bar(x_protT5 - 0.2, protT5_label0, width=0.4, label="Insoluable", color="#2572a9")
    axes[0].bar(x_protT5 + 0.2, protT5_label1, width=0.4, label="Soluable", color="#d5231e")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("ProtT5 Embedding")
    axes[0].set_xlabel("Token position")
    axes[0].set_ylabel("Mean SHAP value (signed)")
    axes[0].legend()

    # ===== Bottom panel: SynCodonLM =====
    axes[1].bar(x_syncodon - 0.2, syncodon_label0, width=0.4, label="Insoluable", color="#2572a9")
    axes[1].bar(x_syncodon + 0.2, syncodon_label1, width=0.4, label="Soluable", color="#d5231e")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("SynCodonLM Embedding")
    axes[1].set_xlabel("Token position")
    axes[1].set_ylabel("Mean SHAP value (signed)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def main():
    args = parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    seed = args.seed
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = args.batch_size
    protT5_dir = args.protT5_dir
    syncodonlm_dir = args.syncodonlm_dir
    model_file = args.model_file
    out_dir = args.out_dir
    max_length = args.max_length
    
    os.makedirs(out_dir, exist_ok=True)

    hidden_dim1_per_path = {
        "protT5": 512,
        "syncodonlm": 512,
    }

    dataset = build_reduced_dataset(
        protT5_dir=protT5_dir,
        syncodonlm_dir=syncodonlm_dir,
        dim_reducer_ckpt="model/dim_reducers.pth",
        device=device,
        batch_size=batch_size
    )

    all_results = []
    
    if args.baseline_protT5_dir is not None and args.baseline_syncodonlm_dir is not None:
        print("[INFO] Building GradientShap baseline dataset from embedding directories...")

        dataset_baseline = build_reduced_dataset(
            protT5_dir=args.baseline_protT5_dir,
            syncodonlm_dir=args.baseline_syncodonlm_dir,
            dim_reducer_ckpt="model/dim_reducers.pth",
            device=device,
            batch_size=args.baseline_batch_size
        )

    else:
        print("[INFO] Loading precomputed GradientShap baseline dataset...")

        loaded = torch.load(
            # "dataset/GradientShap_baseline/test_dataset.pt",
            "/home/wuzhiyong/ecoli/bioseqbridge/model/two_m/linear_aa_codon/test_dataset.pt",
            map_location=device,
            weights_only=False
        )
        dataset_baseline = loaded["test_dataset"]
    
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
    
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    
    label0_mean, label1_mean, token_shap_dict, label0_token_shap_dict, label1_token_shap_dict, label_dict = compute_token_attribution(model, dataset_baseline, device="cuda")
    
    with open(os.path.join(out_dir, "token_shap_dict.pkl"), "wb") as f:
        pickle.dump(token_shap_dict, f)

    with open(os.path.join(out_dir, "label0_token_shap_dict.pkl"), "wb") as f:
        pickle.dump(label0_token_shap_dict, f)

    with open(os.path.join(out_dir, "label1_token_shap_dict.pkl"), "wb") as f:
        pickle.dump(label1_token_shap_dict, f)

    np.save(os.path.join(out_dir, "label0_mean.npy"), label0_mean)
    np.save(os.path.join(out_dir, "label1_mean.npy"), label1_mean)
    

    plot_token_shap_signed_bar_by_seq(label0_mean, label1_mean, protT5_len=max_length, syncodon_len=max_length, out_file=f"{out_dir}/shap_bar.pdf")

        
if __name__ == "__main__":
    main()
