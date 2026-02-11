import argparse
import os
import pickle
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from ProtT5 or SynCodonLM"
    )
    
    parser.add_argument(
        "--mode",
        choices=["protT5", "syncodonlm"],
        required=True,
        help="Select the embedding model to use: "
            "ProtT5 for protein-level embeddings, "
            "SyncodonLM for codon-level embeddings."
    )

    parser.add_argument("--pos_fasta", required=True, help="Path to the FASTA file containing positive sequences.")
    parser.add_argument("--neg_fasta", required=True, help="Path to the FASTA file containing negative sequences.")
    parser.add_argument("--model_dir", required=True, help="Directory where model checkpoints are stored.")
    parser.add_argument("--out_dir", required=True, help="Output directory to save embeddings and results.")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for embedding extraction (default: 64).")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length to process (default: 1024).")
    parser.add_argument("--token_type_id", type=int, default=67, help="Token type ID used for encoding (default: 67).")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use for computation (default: 0).")

    return parser.parse_args()


def load_sequences(mode, pos_fasta, neg_fasta):
    sequences, labels, ids = [], [], []

    for label, fasta in [(1, pos_fasta), (0, neg_fasta)]:
        for record in SeqIO.parse(fasta, "fasta"):
            if mode == "protT5":
                aa = str(Seq(str(record.seq)).translate(to_stop=False))
                aa = aa.replace("*", "").upper()
                seq = " ".join(list(aa))
            else:
                nt = str(record.seq).upper().replace("U", "T")
                seq = " ".join(nt[i:i+3] for i in range(0, len(nt), 3))

            sequences.append(seq)
            labels.append(label)
            ids.append(record.id)

    return sequences, labels, ids


def load_model_and_tokenizer(mode, model_dir, device):
    if mode == "protT5":
        tokenizer = T5Tokenizer.from_pretrained(model_dir)
        model = T5EncoderModel.from_pretrained(model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModel.from_pretrained(model_dir)

    return tokenizer, model.to(device).eval()


def main():
    args = parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Mode: {args.mode}, Device: {device}")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    sequences, labels, ids = load_sequences(
        args.mode, args.pos_fasta, args.neg_fasta
    )
    
    print(f"Total sequences: {len(sequences)}")

    tokenizer, model = load_model_and_tokenizer(
        args.mode, args.model_dir, device
    )

    num_batches = (len(sequences) + args.batch_size - 1) // args.batch_size

    for i in range(num_batches):
        start = i * args.batch_size
        end = min((i + 1) * args.batch_size, len(sequences))

        batch_seqs = sequences[start:end]
        batch_labels = labels[start:end]
        batch_ids = ids[start:end]

        inputs = tokenizer(
            batch_seqs,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors="pt"
        ).to(device)

        if args.mode == "syncodonlm":
            inputs["token_type_ids"] = torch.full_like(
                inputs["input_ids"], args.token_type_id
            )

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = (
                outputs.hidden_states[-1]
                .to(torch.float16)
                .cpu()
                .numpy()
            )

        cls_emb = last_hidden[:, 0, :]

        with open(os.path.join(out_dir, f"embeddings_batch_{i+1}.pkl"), "wb") as f:
            pickle.dump(
                {"embeddings": last_hidden, "labels": batch_labels, "ids": batch_ids}, f
            )

        print(
            f"[INFO] Batch {i+1}/{num_batches} "
            f"(full: {last_hidden.shape}, cls: {cls_emb.shape})"
        )


if __name__ == "__main__":
    main()