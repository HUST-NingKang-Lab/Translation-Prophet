import sys

from .train import main as train_main
from .embedding import main as embedding_main
from .predict import main as predict_main
from .interpret import main as interpret_main
from .optimize_codon import main as optimize_codon_main

def main():
    if len(sys.argv) < 2:
        print("Usage: tp <mode> [options]")
        print("Available modes: embedding, train, predict, interpret, optimize-codon")
        sys.exit(1)

    mode = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if mode == "train":
        train_main()
    elif mode == "embedding":
        embedding_main()
    elif mode == "predict":
        predict_main()
    elif mode == "interpret":
        interpret_main()
    elif mode == "optimize_codon":
        optimize_codon_main()
    else:
        raise ValueError(f"Unknown mode: {mode}")