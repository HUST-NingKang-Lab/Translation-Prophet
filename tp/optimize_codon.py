import os
import time
import argparse
import random
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel

from Bio import SeqIO
from Bio.Seq import Seq


def parse_args():
    parser = argparse.ArgumentParser(
        description="Codon optimization using Translation-Prophet"
    )

    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use (default: 0)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference (default: 64)")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length (default: 1024)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (default: 0)")
    parser.add_argument("--window_size", type=int, default=12, help="Sliding window size in nucleotides (default: 12)")
    parser.add_argument("--step_size", type=int, default=9, help="Step size for sliding window (default: 9)")
    parser.add_argument("--k", type=int, default=3, help="Beam size: top-k sequences retained per window (default: 3)")
    parser.add_argument("--raw_fasta", type=str, required=True, help="Input FASTA file containing nucleotide sequences")
    parser.add_argument("--model_file", type=str, required=True, help="Path to the trained Translation-Prophet model")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save optimized sequences and results")
    parser.add_argument(
        "--protT5_embedding_model_dir",
        type=str,
        default="embedding_model/ProtT5",
        help="Directory where ProtT5 checkpoints are stored (default: 'embedding_model/ProtT5')"
    )
    parser.add_argument(
        "--syncodonlm_embedding_model_dir",
        type=str,
        default="embedding_model/SynCodonLM",
        help="Directory where SynCodonLM checkpoints are stored (default: 'embedding_model/SynCodonLM')"
    )
    return parser.parse_args()


def read_fasta_sequences(raw_fasta):
    seqs, ids = [], []
    for record in SeqIO.parse(raw_fasta, "fasta"):
        seq = str(record.seq).upper().replace("U", "T") 
        seqs.append(seq)
        ids.append(record.id) 
    return seqs, ids


def fasta_to_sequences_and_labels_optimize(seqs):
    input_sequences = [seq for seq in seqs]
    labels = [1] * len(seqs)
    ids = list(range(len(seqs)))
    return input_sequences, labels, ids


def fasta_to_sequences_and_labels_aa(seqs):
    input_sequences, labels, ids = [], [], []
    for i, seq in enumerate(seqs):
        aa_seq = str(Seq(seq).translate(to_stop=False))
        aa_seq = aa_seq.replace("*", "").upper()
        aa_seq = " ".join(list(aa_seq))
        input_sequences.append(aa_seq)
        labels.append(1)
        ids.append(i)
    return input_sequences, labels, ids


def fasta_to_sequences_and_labels_codon(seqs):
    input_sequences, labels, ids = [], [], []
    for i, seq in enumerate(seqs):
        seq = ' '.join([seq[j:j+3] for j in range(0, len(seq), 3)])
        input_sequences.append(seq)
        labels.append(1)
        ids.append(i)
    return input_sequences, labels, ids


def align_indices_by_id(aa_ids, codon_ids):
    id2idx_codon = {id_: i for i, id_ in enumerate(codon_ids)}
    idx_aa, idx_codon, aligned_ids = [], [], []
    for i, id_ in enumerate(aa_ids):
        if id_ in id2idx_codon:
            idx_aa.append(i)
            idx_codon.append(id2idx_codon[id_])
            aligned_ids.append(id_)
    return idx_aa, idx_codon, aligned_ids

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


def load_reducer(checkpoint_path, name, input_dim, hidden_dim1, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    reducer = Linear_DimReducer(input_dim, hidden_dim1).to(device)
    reducer.load_state_dict(ckpt[name])
    reducer.eval()
    return reducer

def reduce_with_dimreducer(emb_np, name, input_dim, hidden_dim1,
                           checkpoint_path, device=None, batch_size=None):
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
    def __init__(self, reduced_list_np, labels_np, ids_np=None):
        self.reduced_list = reduced_list_np
        self.labels = labels_np.astype(int)
        self.ids = ids_np if ids_np is not None else np.arange(len(labels_np)) 
        self.n = self.labels.shape[0]

        assert len(self.ids) == self.n
        for arr in self.reduced_list:
            assert arr.shape[0] == self.n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        xs = [torch.from_numpy(arr[idx]).float() for arr in self.reduced_list]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        seq_id = self.ids[idx]
        return xs, y, seq_id

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

def optimize_sequence_by_window(
    raw_fasta, 
    model, 
    prot_tokenizer, prot_encoder, 
    codon_tokenizer, codon_encoder, 
    dim_reducer_ckpt,
    window_size,
    step_size,
    k,
    max_length,
    codon_table,
    batch_size=None,
    device=None,
    save_path="optimized_results.csv"
):
    raw_seqs, raw_names = read_fasta_sequences(raw_fasta)
    aa_seqs, _, _ = fasta_to_sequences_and_labels_aa(raw_seqs)
    codon_seqs, _, _ = fasta_to_sequences_and_labels_codon(raw_seqs)
    seqs, _, _ = fasta_to_sequences_and_labels_optimize(raw_seqs)

    def seqs_to_emb_batch(aa_batch, codon_batch):
        aa_inputs = prot_tokenizer(
            aa_batch,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = prot_encoder(**aa_inputs, output_hidden_states=True)
            emb_aa = outputs.hidden_states[-1].cpu().numpy()

        emb1_red = reduce_with_dimreducer(
            emb_aa, name="protT5",
            input_dim=emb_aa.shape[-1], hidden_dim1=512,
            checkpoint_path=f"model/dim_reducers.pth",
            device=device, batch_size=batch_size
        )

        token_type_id = 67  # E. coli
        codon_inputs = codon_tokenizer(
            codon_batch,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        codon_inputs["token_type_ids"] = torch.full_like(
            codon_inputs["input_ids"], token_type_id
        ).to(device)

        with torch.no_grad():
            outputs = codon_encoder(**codon_inputs, output_hidden_states=True)
            emb_codon = outputs.hidden_states[-1].cpu().numpy()

        emb2_red = reduce_with_dimreducer(
            emb_codon, name="syncodonml",
            input_dim=emb_codon.shape[-1], hidden_dim1=512,
            checkpoint_path=f"model/dim_reducers.pth",
            device=device, batch_size=batch_size
        )

        dataset = MultiPathReducedDataset(
            reduced_list_np=[emb1_red, emb2_red],
            labels_np=np.zeros(emb1_red.shape[0])
        )
        return dataset.reduced_list

    def evaluate_batch(aa_batch, codon_batch):
        model.eval()
        emb_list = seqs_to_emb_batch(aa_batch, codon_batch)
        emb_list = [e.to(device) for e in emb_list]
        with torch.no_grad():
            logits = model(emb_list).cpu().numpy()
        logits = np.array(logits)
        logit0 = logits[:, 0]
        logit1 = logits[:, 1]
        score = logit1 - logit0
        return logit0, logit1, score

    open(save_path, "w").close()
    write_header = True

    for name, aa_seq, codon_seq, seq in zip(raw_names, aa_seqs, codon_seqs, seqs):
        final_results = []
        print(f"\n=== Optimizing {name} ===")

        logit0, logit1, score = evaluate_batch([aa_seq], [codon_seq])
        beam = [(seq, score[0])]
        seq_len = len(seq)
        best_k_seqs = [seq]

        for start in range(0, seq_len, step_size):
            end = min(start + window_size, seq_len)
            print(f"\nWindow {start}-{end}")

            candidate_seqs = []

            for s in best_k_seqs:
                codons = [s[i:i+3] for i in range(start, end, 3)]
                synonym_options = []
                for codon in codons:
                    aa = codon_table.get(codon, None)
                    if aa is None:
                        synonym_options.append([codon])
                    else:
                        synonyms = [c for c in codon_table if codon_table[c] == aa]
                        synonym_options.append(synonyms)

                for codon_combo in itertools.product(*synonym_options):
                    new_window = "".join(codon_combo)
                    new_seq = s[:start] + new_window + s[end:]
                    candidate_seqs.append(new_seq)

            print(f"Generated {len(candidate_seqs)} candidates")

            if not candidate_seqs:
                continue

            aa_news, _, _ = fasta_to_sequences_and_labels_aa(candidate_seqs)
            codon_news, _, _ = fasta_to_sequences_and_labels_codon(candidate_seqs)

            logit0_all, logit1_all, score_all = [], [], []
            for i in range(0, len(candidate_seqs), batch_size):
                aa_batch = aa_news[i:i+batch_size]
                codon_batch = codon_news[i:i+batch_size]
                l0, l1, s = evaluate_batch(aa_batch, codon_batch)
                logit0_all.extend(l0)
                logit1_all.extend(l1)
                score_all.extend(s)

            logit0_all = np.array(logit0_all)
            logit1_all = np.array(logit1_all)
            score_all = np.array(score_all)

            topk_idx = np.argsort(score_all)[-k:]
            best_k_seqs = [candidate_seqs[i] for i in topk_idx]

            print(f"Top {k} sequences retained, best score={score_all[topk_idx[-1]]:.4f}")

        logit0_raw, logit1_raw, score_raw = evaluate_batch([aa_seq], [codon_seq])
        final_results.append({
            "seq_name": name,
            "optimized_seq": seq,
            "logit0": logit0_raw[0],
            "logit1": logit1_raw[0],
            "score": score_raw[0],
            "type": "raw"
        })

        aa_finals, _, _ = fasta_to_sequences_and_labels_aa(best_k_seqs)
        codon_finals, _, _ = fasta_to_sequences_and_labels_codon(best_k_seqs)
        l0, l1, s = evaluate_batch(aa_finals, codon_finals)
        for seq_final, lo0, lo1, sc in zip(best_k_seqs, l0, l1, s):
            final_results.append({
                "seq_name": name,
                "optimized_seq": seq_final,
                "logit0": float(lo0),
                "logit1": float(lo1),
                "score": float(sc),
                "type": "optimized"
            })
            print(f"[{name}] optimized score={sc:.4f}")

        

        df = pd.DataFrame(final_results)
        df.to_csv(save_path, mode="a", index=False, header=write_header)
        write_header = False 


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
    model_file = args.model_file
    out_dir = args.out_dir
    max_length = args.max_length
    window_size = args.window_size
    step_size = args.step_size
    k = args.k
    protT5_embedding_model_dir = args.protT5_embedding_model_dir
    syncodonlm_embedding_model_dir = args.syncodonlm_embedding_model_dir
    raw_fasta = args.raw_fasta
    
    os.makedirs(out_dir, exist_ok=True)
    
    codon_table = {
        'TTT': 'F', 'TTC': 'F',
        'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I',
        'ATG': 'M',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'TAT': 'Y', 'TAC': 'Y',
        'CAT': 'H', 'CAC': 'H',
        'CAA': 'Q', 'CAG': 'Q',
        'AAT': 'N', 'AAC': 'N',
        'AAA': 'K', 'AAG': 'K',
        'GAT': 'D', 'GAC': 'D',
        'GAA': 'E', 'GAG': 'E',
        'TGT': 'C', 'TGC': 'C',
        'TGG': 'W',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
        'TAA': '*', 'TAG': '*', 'TGA': '*'
    }
        
    aa_seqs, labels_aa, ids_aa = fasta_to_sequences_and_labels_aa(raw_fasta)
    labels_aa = np.array(labels_aa)

    prot_tokenizer = T5Tokenizer.from_pretrained(f'{protT5_embedding_model_dir}')
    prot_encoder = T5EncoderModel.from_pretrained(f'{protT5_embedding_model_dir}').to(device).eval()

    codon_seqs, labels_codon, ids_codon = fasta_to_sequences_and_labels_codon(raw_fasta)
    labels_codon = np.array(labels_codon)

    codon_tokenizer = AutoTokenizer.from_pretrained(f'{syncodonlm_embedding_model_dir}')
    codon_encoder = AutoModel.from_pretrained(f'{syncodonlm_embedding_model_dir}').to(device).eval()

    hidden_dim1_per_path = {
        "ProtT5": 512,
        "SynCodonLM": 512,
        "lucaone": 512,
        }

    in_dims_per_path = [
            hidden_dim1_per_path["ProtT5"],
            hidden_dim1_per_path["SynCodonLM"],
        ]

    results = []

    model = GatedFusionNet(
        in_dims_per_path, 
        lstm_hidden_dim2=128,
        cnn_num_filters=128, 
        num_classes=2, 
        dropout=0.5
    ).to(device)

    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    start_time = time.time()

    optimize_sequence_by_window(raw_fasta, 
                                model, 
                                prot_tokenizer, 
                                prot_encoder, 
                                codon_tokenizer, 
                                codon_encoder, 
                                in_dims_per_path, 
                                window_size=window_size, 
                                step_size=step_size, 
                                k=k, 
                                max_length=max_length,
                                codon_table=codon_table,
                                device=device, 
                                batch_size=batch_size, 
                                save_path=f"{out_dir}/optimized_results.csv"
                                )            
    end_time = time.time()
    print(f"{end_time - start_time:.2f} s")

if __name__ == "__main__":
    main()

