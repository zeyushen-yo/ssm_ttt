"""
Data pipeline for training and evaluation.

Supports two modes:
1. Streaming from HuggingFace (login node / internet)
2. Loading from pre-tokenized .bin files (compute nodes / offline)

Spec constraints:
- Each sample = one contiguous document segment (no multi-document packing in Phase 1)
- Shared tokenizer across all three models (GPT-NeoX)
- Document-boundary resets for TTT
"""

import os
import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
from transformers import AutoTokenizer


def get_tokenizer(name="EleutherAI/gpt-neox-20b"):
    """Get the shared tokenizer (GPT-NeoX for The Pile compatibility)."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class PreTokenizedTrainDataset(Dataset):
    """
    Dataset loading from pre-tokenized .bin file (memory-mapped).

    Each item is a contiguous chunk of seq_len tokens from a single document.
    We split at EOS tokens to respect document boundaries (no packing).
    """

    def __init__(self, data_path, seq_len=2048, eos_token_id=0, seed=42):
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id

        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')

        meta_path = data_path.replace('.bin', '_meta.txt')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                for line in f:
                    if line.startswith("num_tokens="):
                        self.n_tokens = int(line.strip().split("=")[1])
                        break
                else:
                    self.n_tokens = len(self.data)
        else:
            self.n_tokens = self._find_valid_extent()

        print(f"PreTokenizedTrainDataset: using {self.n_tokens:,} / {len(self.data):,} tokens")

        self.rng = np.random.RandomState(seed)
        self._len = self.n_tokens // seq_len

    def _find_valid_extent(self):
        """Binary search for the boundary between written and unwritten data."""
        total = len(self.data)
        lo, hi = 0, total
        while lo < hi:
            mid = (lo + hi) // 2
            chunk = self.data[mid:min(mid + 10, total)]
            if any(chunk != 0):
                lo = mid + 1
            else:
                hi = mid
        return lo

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # Random start position (avoid last seq_len tokens)
        max_start = self.n_tokens - self.seq_len - 1
        if max_start <= 0:
            start = 0
        else:
            start = self.rng.randint(0, max_start)

        tokens = self.data[start:start + self.seq_len].astype(np.int64)

        # Check if there's an EOS within this chunk - if so, only use up to first EOS
        # This ensures no cross-document contamination
        eos_positions = np.where(tokens == self.eos_token_id)[0]
        if len(eos_positions) > 0:
            # Start from after the first EOS to get a clean document start
            first_eos = eos_positions[0]
            if first_eos < self.seq_len - 256:
                # Enough tokens after EOS
                remaining = tokens[first_eos + 1:]
                if len(remaining) >= 256:
                    tokens = remaining[:self.seq_len]
                    # Pad if needed
                    if len(tokens) < self.seq_len:
                        tokens = np.pad(tokens, (0, self.seq_len - len(tokens)),
                                       constant_values=self.eos_token_id)

        input_ids = torch.from_numpy(tokens.copy())
        return {"input_ids": input_ids, "labels": input_ids.clone()}


class PreTokenizedValDataset(Dataset):
    """
    Validation dataset from pre-tokenized .bin + offsets.
    Each item is one document (up to max_len tokens).
    """

    def __init__(self, data_path, offsets_path, max_len=32768):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.offsets = np.load(offsets_path)
        self.max_len = max_len
        self.num_docs = len(self.offsets) - 1

    def __len__(self):
        return self.num_docs

    def __getitem__(self, idx):
        start = self.offsets[idx]
        end = min(self.offsets[idx + 1], start + self.max_len)
        tokens = self.data[start:end].astype(np.int64)
        return {"input_ids": torch.from_numpy(tokens.copy())}


class StreamingTrainDataset(IterableDataset):
    """Fallback streaming dataset when pre-tokenized data not available."""

    def __init__(self, tokenizer, seq_len=2048, dataset_name="monology/pile-uncopyrighted",
                 seed=42, min_doc_len=256):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.seed = seed
        self.min_doc_len = min_doc_len

    def __iter__(self):
        from datasets import load_dataset
        dataset = load_dataset(
            self.dataset_name, split="train", streaming=True, trust_remote_code=True
        )
        dataset = dataset.shuffle(seed=self.seed, buffer_size=10000)

        for example in dataset:
            text = example.get("text", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) < self.min_doc_len:
                continue
            tokens = tokens[:self.seq_len]
            if len(tokens) < self.seq_len:
                tokens = tokens + [self.tokenizer.pad_token_id] * (self.seq_len - len(tokens))
            input_ids = torch.tensor(tokens, dtype=torch.long)
            yield {"input_ids": input_ids, "labels": input_ids.clone()}


def create_train_dataloader(
    tokenizer=None,
    seq_len=2048,
    batch_size=8,
    num_workers=4,
    data_dir=None,
    dataset_name="monology/pile-uncopyrighted",
    seed=42,
):
    """Create training dataloader. Uses pre-tokenized data if available."""
    if data_dir and os.path.exists(os.path.join(data_dir, "train.bin")):
        print(f"Loading pre-tokenized data from {data_dir}/train.bin")
        eos_id = tokenizer.eos_token_id if tokenizer else 0
        dataset = PreTokenizedTrainDataset(
            os.path.join(data_dir, "train.bin"),
            seq_len=seq_len,
            eos_token_id=eos_id,
            seed=seed,
        )
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
    else:
        print("Using streaming dataset (requires internet)")
        assert tokenizer is not None
        dataset = StreamingTrainDataset(
            tokenizer, seq_len=seq_len, dataset_name=dataset_name, seed=seed,
        )
        return DataLoader(
            dataset, batch_size=batch_size,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
