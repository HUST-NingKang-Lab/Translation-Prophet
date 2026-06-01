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
    parser.add_argument("--parallel_seqs", type=int, default=1, help="Number of FASTA sequences optimized together in one pooled inference loop (default: 1)")
    parser.add_argument("--dim_reducer_file", type=str, default="model/dim_reducers.pth", help="Path to dim reducer checkpoint (default: model/dim_reducers.pth)")
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
    parallel_seqs=1,
    save_path="optimized_results.csv"
):
    raw_seqs, raw_names = read_fasta_sequences(raw_fasta)
    aa_seqs, _, _ = fasta_to_sequences_and_labels_aa(raw_seqs)
    codon_seqs, _, _ = fasta_to_sequences_and_labels_codon(raw_seqs)
    seqs, _, _ = fasta_to_sequences_and_labels_optimize(raw_seqs)

    parallel_seqs = max(1, int(parallel_seqs))
    aa_to_codons = {}
    for _codon, _aa in codon_table.items():
        aa_to_codons.setdefault(_aa, []).append(_codon)
    dim_reducer_state = None
    reducer_cache = {}
    prot_reduced_cache = {}

    def get_cached_reducer(name, input_dim, hidden_dim1):
        nonlocal dim_reducer_state
        key = (name, input_dim, hidden_dim1)
        if key in reducer_cache:
            return reducer_cache[key]

        if dim_reducer_state is None:
            dim_reducer_state = torch.load(dim_reducer_ckpt, map_location="cpu")

        reducer = Linear_DimReducer(input_dim, hidden_dim1).to(device)
        reducer.load_state_dict(dim_reducer_state[name])
        reducer.eval()
        reducer_cache[key] = reducer
        return reducer

    def reduce_with_cached_dimreducer(emb_np, name, input_dim, hidden_dim1, batch_size=None):
        reducer = get_cached_reducer(name, input_dim, hidden_dim1)
        X = torch.from_numpy(emb_np).float()
        reduced_batches = []

        with torch.inference_mode():
            for i in range(0, X.shape[0], batch_size):
                batch = X[i:i+batch_size].to(device, non_blocking=True)
                reduced = reducer(batch)
                reduced_batches.append(reduced)
                del batch

        return torch.cat(reduced_batches, dim=0)

    def reduce_tensor_with_cached_dimreducer(emb_tensor, name, input_dim, hidden_dim1, batch_size=None):
        reducer = get_cached_reducer(name, input_dim, hidden_dim1)
        reduced_batches = []

        with torch.inference_mode():
            for i in range(0, emb_tensor.shape[0], batch_size):
                batch = emb_tensor[i:i+batch_size].to(device, non_blocking=True)
                reduced = reducer(batch)
                reduced_batches.append(reduced)
                del batch

        return torch.cat(reduced_batches, dim=0)

    def encode_prot_reduced_batch(aa_batch):
        aa_inputs = prot_tokenizer(
            aa_batch,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            outputs = prot_encoder(**aa_inputs, output_hidden_states=True)
            emb_aa = outputs.hidden_states[-1]

        return reduce_tensor_with_cached_dimreducer(
            emb_aa, name="protT5",
            input_dim=emb_aa.shape[-1], hidden_dim1=512,
            batch_size=batch_size
        )

    def encode_codon_reduced_batch(codon_batch):
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

        with torch.inference_mode():
            outputs = codon_encoder(**codon_inputs, output_hidden_states=True)
            emb_codon = outputs.hidden_states[-1]

        return reduce_tensor_with_cached_dimreducer(
            emb_codon, name="syncodonml",
            input_dim=emb_codon.shape[-1], hidden_dim1=512,
            batch_size=batch_size
        )

    def get_prot_reduced_for_item(item):
        cache_key = item["global_idx"]
        if cache_key not in prot_reduced_cache:
            prot_reduced_cache[cache_key] = encode_prot_reduced_batch([item["aa_seq"]])
        return prot_reduced_cache[cache_key]

    def evaluate_with_reduced_prot(prot_reduced_batch, codon_batch):
        model.eval()
        emb2_red = encode_codon_reduced_batch(codon_batch)
        emb_list = [
            prot_reduced_batch,
            emb2_red
        ]
        with torch.inference_mode():
            logits = model(emb_list).detach().cpu().numpy()
        logits = np.array(logits)
        logit0 = logits[:, 0]
        logit1 = logits[:, 1]
        score = logit1 - logit0
        return logit0, logit1, score

    def evaluate_codon_batch_for_item(item, codon_batch):
        prot_one = get_prot_reduced_for_item(item)
        prot_batch = prot_one.expand(len(codon_batch), -1, -1).contiguous()
        return evaluate_with_reduced_prot(prot_batch, codon_batch)

    def evaluate_codon_batch_for_active_indices(active_indices, codon_batch):
        prot_batch = torch.cat(
            [get_prot_reduced_for_item(active_items[idx]) for idx in active_indices],
            dim=0
        )
        return evaluate_with_reduced_prot(prot_batch, codon_batch)

    def seqs_to_emb_batch(aa_batch, codon_batch):
        emb1_red = encode_prot_reduced_batch(aa_batch)
        emb2_red = encode_codon_reduced_batch(codon_batch)
        return [emb1_red, emb2_red]

    def evaluate_batch(aa_batch, codon_batch):
        model.eval()
        emb_list = seqs_to_emb_batch(aa_batch, codon_batch)
        with torch.inference_mode():
            logits = model(emb_list).detach().cpu().numpy()
        logits = np.array(logits)
        logit0 = logits[:, 0]
        logit1 = logits[:, 1]
        score = logit1 - logit0
        return logit0, logit1, score

    def generate_candidates_for_window(best_k_seqs, start, end):
        candidate_seqs = []
        for s in best_k_seqs:
            codons = [s[i:i+3] for i in range(start, end, 3)]
            synonym_options = []
            for codon in codons:
                aa = codon_table.get(codon, None)
                if aa is None:
                    synonym_options.append([codon])
                else:
                    synonyms = aa_to_codons.get(aa, [codon])
                    synonym_options.append(synonyms)

            for codon_combo in itertools.product(*synonym_options):
                new_window = "".join(codon_combo)
                new_seq = s[:start] + new_window + s[end:]
                candidate_seqs.append(new_seq)

        return candidate_seqs

    open(save_path, "w").close()
    write_header = True

    def build_item(idx):
        seq = seqs[idx]
        item = {
            "global_idx": idx,
            "name": raw_names[idx],
            "aa_seq": aa_seqs[idx],
            "codon_seq": codon_seqs[idx],
            "raw_seq": seq,
            "seq_len": len(seq),
            "best_k_seqs": [seq],
            "window_starts": list(range(0, len(seq), step_size)),
            "next_window_idx": 0,
            "done_windows": 0,
        }

        logit0_raw, logit1_raw, score_raw = evaluate_codon_batch_for_item(item, [item["codon_seq"]])
        item["raw_logit0"] = float(logit0_raw[0])
        item["raw_logit1"] = float(logit1_raw[0])
        item["raw_score"] = float(score_raw[0])
        return item

    def write_final_results(item):
        nonlocal write_header

        final_results = []
        name = item["name"]
        aa_seq = item["aa_seq"]
        codon_seq = item["codon_seq"]
        seq = item["raw_seq"]
        best_k_seqs = item["best_k_seqs"]

        final_results.append({
            "seq_name": name,
            "optimized_seq": seq,
            "logit0": item["raw_logit0"],
            "logit1": item["raw_logit1"],
            "score": item["raw_score"],
            "type": "raw"
        })

        codon_finals, _, _ = fasta_to_sequences_and_labels_codon(best_k_seqs)
        l0, l1, s = evaluate_codon_batch_for_item(item, codon_finals)
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

        prot_reduced_cache.pop(item["global_idx"], None)

    next_seq_idx = 0
    active_items = []
    completed_count = 0
    total_count = len(seqs)

    def fill_active_slots():
        nonlocal next_seq_idx
        while len(active_items) < parallel_seqs and next_seq_idx < total_count:
            item = build_item(next_seq_idx)
            active_items.append(item)
            print(
                f"\n=== Activate {item['name']} "
                f"({next_seq_idx + 1}/{total_count}); "
                f"active={len(active_items)}/{parallel_seqs} ==="
            )
            next_seq_idx += 1

    fill_active_slots()
    round_idx = 0

    while active_items:
        round_idx += 1
        pooled_candidate_seqs = []
        pooled_blocks = []
        pooled_candidate_owner_indices = []

        print(f"\n=== Dynamic round {round_idx}; active sequences={len(active_items)} ===")

        for item_idx, item in enumerate(active_items):
            if item["next_window_idx"] >= len(item["window_starts"]):
                continue

            start = item["window_starts"][item["next_window_idx"]]
            end = min(start + window_size, item["seq_len"])
            print(f"\n[{item['name']}] Window {start}-{end}")

            candidate_seqs = generate_candidates_for_window(
                item["best_k_seqs"], start, end
            )
            print(f"[{item['name']}] Generated {len(candidate_seqs)} candidates")

            if not candidate_seqs:
                item["next_window_idx"] += 1
                item["done_windows"] += 1
                continue

            block_start = len(pooled_candidate_seqs)
            pooled_candidate_seqs.extend(candidate_seqs)
            pooled_candidate_owner_indices.extend([item_idx] * len(candidate_seqs))
            block_end = len(pooled_candidate_seqs)
            pooled_blocks.append((item_idx, block_start, block_end))

        if pooled_candidate_seqs:
            print(
                f"\nPooled candidates from {len(pooled_blocks)} active sequences: "
                f"{len(pooled_candidate_seqs)}"
            )

            codon_news, _, _ = fasta_to_sequences_and_labels_codon(pooled_candidate_seqs)

            logit0_all, logit1_all, score_all = [], [], []
            for i in range(0, len(pooled_candidate_seqs), batch_size):
                codon_batch = codon_news[i:i+batch_size]
                owner_batch = pooled_candidate_owner_indices[i:i+batch_size]
                l0, l1, s = evaluate_codon_batch_for_active_indices(owner_batch, codon_batch)
                logit0_all.extend(l0)
                logit1_all.extend(l1)
                score_all.extend(s)

            logit0_all = np.array(logit0_all)
            logit1_all = np.array(logit1_all)
            score_all = np.array(score_all)

            for item_idx, block_start, block_end in pooled_blocks:
                item = active_items[item_idx]
                candidate_seqs = pooled_candidate_seqs[block_start:block_end]
                score_block = score_all[block_start:block_end]

                k_eff = min(k, len(score_block))
                if len(score_block) > k_eff:
                    topk_idx = np.argpartition(score_block, -k_eff)[-k_eff:]
                    topk_idx = topk_idx[np.argsort(score_block[topk_idx])]
                else:
                    topk_idx = np.argsort(score_block)[-k_eff:]
                item["best_k_seqs"] = [candidate_seqs[i] for i in topk_idx]
                item["next_window_idx"] += 1
                item["done_windows"] += 1

                best_score = score_block[topk_idx[-1]]
                print(f"[{item['name']}] Top {k} sequences retained, best score={best_score:.4f}")

        finished_indices = [
            i for i, item in enumerate(active_items)
            if item["next_window_idx"] >= len(item["window_starts"])
        ]

        if finished_indices:
            for i in reversed(finished_indices):
                item = active_items.pop(i)
                completed_count += 1
                print(
                    f"\n=== Finished {item['name']} "
                    f"({completed_count}/{total_count}); "
                    f"windows={item['done_windows']} ==="
                )
                write_final_results(item)

            fill_active_slots()




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
    parallel_seqs = args.parallel_seqs
    dim_reducer_file = args.dim_reducer_file
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
                                dim_reducer_file, 
                                window_size=window_size, 
                                step_size=step_size, 
                                k=k, 
                                max_length=max_length,
                                codon_table=codon_table,
                                device=device, 
                                batch_size=batch_size, 
                                parallel_seqs=parallel_seqs,
                                save_path=f"{out_dir}/optimized_results.csv"
                                )            
    end_time = time.time()
    print(f"{end_time - start_time:.2f} s")

if __name__ == "__main__":
    main()

