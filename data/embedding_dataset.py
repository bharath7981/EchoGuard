import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """
    Dataset for precomputed SSL embeddings stored on disk.
    Supports a single .pt file or comma-separated list of .pt files.
    """

    def __init__(self, pt_paths):
        if isinstance(pt_paths, str):
            paths = [p.strip() for p in pt_paths.split(",") if p.strip()]
        else:
            paths = list(pt_paths)

        all_embs = []
        all_lbls = []
        all_ids = []

        for p in paths:
            data = torch.load(p, weights_only=False)
            all_embs.append(data["embeddings"].float())
            all_lbls.append(data["labels"].long())
            all_ids.extend(data.get("utt_ids", []))

        self.embeddings = torch.cat(all_embs, dim=0)
        self.labels = torch.cat(all_lbls, dim=0)
        self.utt_ids = all_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ds = EmbeddingDataset(sys.argv[1])
        print(f"Loaded {len(ds)} samples, shape={ds[0][0].shape}, label={ds[0][1]}")
