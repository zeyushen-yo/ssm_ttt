"""
Data pipeline for training and evaluation — v2.

Uses document-offset-based sampling to guarantee single-document sequences.
Worker RNG is properly seeded per (base_seed, worker_id, idx) to avoid
duplication across dataloader workers.

Spec constraints:
- Each sample = one contiguous document segment (no multi-document packing)
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


class DocOffsetTrainDataset(Dataset):
    """
    Training dataset using document offsets for guaranteed single-document samples.

    Each __getitem__ call:
    1. Derives a per-(worker, idx) RNG seed
    2. Samples a document from train_offsets.npy
    3. Samples a start position within that document
    4. Returns exactly seq_len tokens from that document (padded at doc end if needed)
    """

    def __init__(self, data_path, offsets_path, seq_len=2048, eos_token_id=0,
                 seed=42, min_doc_len=256):
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id
        self.seed = seed

        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.offsets = np.load(offsets_path)
        self.n_docs = len(self.offsets) - 1

        self.doc_lengths = self.offsets[1:] - self.offsets[:-1]
        self.valid_doc_mask = self.doc_lengths >= min_doc_len
        self.valid_doc_indices = np.where(self.valid_doc_mask)[0]
        self.n_valid_docs = len(self.valid_doc_indices)

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
            self.n_tokens = len(self.data)

        print(f"DocOffsetTrainDataset: {self.n_tokens:,} tokens, "
              f"{self.n_docs} total docs, {self.n_valid_docs} valid docs (>={min_doc_len})")

        self._len = self.n_tokens // seq_len

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        rng = np.random.default_rng(self.seed + 1000003 * worker_id + idx)

        doc_idx = self.valid_doc_indices[rng.integers(0, self.n_valid_docs)]
        doc_start = self.offsets[doc_idx]
        doc_end = self.offsets[doc_idx + 1]
        doc_len = doc_end - doc_start

        if doc_len <= self.seq_len:
            start = doc_start
            real_tokens = self.data[start:doc_end].astype(np.int64)
            real_len = len(real_tokens)
            if real_len < self.seq_len:
                tokens = np.pad(real_tokens, (0, self.seq_len - real_len),
                                constant_values=self.eos_token_id)
            else:
                tokens = real_tokens
                real_len = self.seq_len
        else:
            max_start_within_doc = doc_len - self.seq_len
            offset_in_doc = rng.integers(0, max_start_within_doc + 1)
            start = doc_start + offset_in_doc
            tokens = self.data[start:start + self.seq_len].astype(np.int64)
            real_len = self.seq_len

        input_ids = torch.from_numpy(tokens.copy())
        labels = input_ids.clone()
        if real_len < self.seq_len:
            labels[real_len:] = -100
        return {"input_ids": input_ids, "labels": labels}


class PackedTrainDataset(Dataset):
    """
    Packed document training dataset for long-context pre-training.
    Reads contiguous chunks of seq_len tokens from the full corpus,
    allowing sequences to span document boundaries. This gives access
    to the entire corpus regardless of individual document lengths.
    """

    def __init__(self, data_path, seq_len=32768, seed=42):
        self.seq_len = seq_len
        self.seed = seed
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
            self.n_tokens = len(self.data)

        print(f"PackedTrainDataset: {self.n_tokens:,} tokens, seq_len={seq_len}")
        self._len = self.n_tokens // seq_len

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        rng = np.random.default_rng(self.seed + 1000003 * worker_id + idx)

        max_start = self.n_tokens - self.seq_len - 1
        start = rng.integers(0, max(1, max_start))
        tokens = self.data[start:start + self.seq_len].astype(np.int64)

        input_ids = torch.from_numpy(tokens.copy())
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


class PreTokenizedTrainDataset(Dataset):
    """
    Fallback dataset for data_cache without train_offsets.npy.
    Uses the old random-start approach but with proper worker RNG.
    """

    def __init__(self, data_path, seq_len=2048, eos_token_id=0, seed=42):
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id
        self.seed = seed

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
            self.n_tokens = len(self.data)

        print(f"PreTokenizedTrainDataset (fallback): {self.n_tokens:,} tokens")
        self._len = self.n_tokens // seq_len

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        rng = np.random.default_rng(self.seed + 1000003 * worker_id + idx)

        max_start = self.n_tokens - self.seq_len - 1
        if max_start <= 0:
            start = 0
        else:
            start = rng.integers(0, max_start)

        tokens = self.data[start:start + self.seq_len].astype(np.int64)

        eos_positions = np.where(tokens == self.eos_token_id)[0]
        if len(eos_positions) > 0:
            first_eos = eos_positions[0]
            if first_eos < self.seq_len - 256:
                remaining = tokens[first_eos + 1:]
                if len(remaining) >= 256:
                    tokens = remaining[:self.seq_len]
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

    def __init__(self, data_path, offsets_path, max_len=65536):
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
    pack_documents=False,
):
    """Create training dataloader. Prefers doc-offset dataset, falls back to old format.

    Args:
        pack_documents: If True, use PackedTrainDataset which reads contiguous
            chunks from the full corpus (sequences may span document boundaries).
            Recommended for long seq_len training where few documents exceed seq_len.
    """
    if data_dir:
        train_bin = os.path.join(data_dir, "train.bin")
        offsets_file = os.path.join(data_dir, "train_offsets.npy")

        if pack_documents and os.path.exists(train_bin):
            print(f"Loading packed training data from {data_dir} (pack_documents=True)")
            dataset = PackedTrainDataset(
                train_bin, seq_len=seq_len, seed=seed,
            )
            return DataLoader(
                dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True, drop_last=True,
            )

        if os.path.exists(train_bin) and os.path.exists(offsets_file):
            print(f"Loading doc-offset training data from {data_dir}")
            eos_id = tokenizer.eos_token_id if tokenizer else 0
            dataset = DocOffsetTrainDataset(
                train_bin, offsets_file,
                seq_len=seq_len, eos_token_id=eos_id, seed=seed,
                min_doc_len=seq_len,
            )
            return DataLoader(
                dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True, drop_last=True,
            )
        elif os.path.exists(train_bin):
            print(f"Loading pre-tokenized data (no offsets) from {data_dir}/train.bin")
            eos_id = tokenizer.eos_token_id if tokenizer else 0
            dataset = PreTokenizedTrainDataset(
                train_bin, seq_len=seq_len, eos_token_id=eos_id, seed=seed,
            )
            return DataLoader(
                dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True, drop_last=True,
            )

    print("Using streaming dataset (requires internet)")
    assert tokenizer is not None
    dataset = StreamingTrainDataset(
        tokenizer, seq_len=seq_len, dataset_name=dataset_name, seed=seed,
    )
    return DataLoader(
        dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
