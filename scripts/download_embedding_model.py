import os
from huggingface_hub import snapshot_download

DOWNLOAD_TASKS = [
    {
        "repo_id": "jheuschkel/SynCodonLM",
        "revision": "9b0db6a0a46ebf89cb2b18085c3de39fbe2a22c4",
        "out_dir": "embedding_model/SynCodonML",
    },
    {
        "repo_id": "Rostlab/prot_t5_xl_uniref50",
        "revision": "main",
        "out_dir": "embedding_model/ProtT5",
    }
]

for task in DOWNLOAD_TASKS:
    os.makedirs(task["out_dir"], exist_ok=True)
    snapshot_download(
        repo_id=task["repo_id"],
        revision=task["revision"],
        local_dir=task["out_dir"],
        local_dir_use_symlinks=False,
        resume_download=True,
    )

print("All models downloaded successfully.")