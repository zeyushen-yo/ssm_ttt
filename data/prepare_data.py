"""
Pre-tokenize and save training/validation data for offline use on compute nodes.

Downloads from HuggingFace on login node, tokenizes, and saves as memory-mapped
numpy arrays for fast loading during training.

Usage (run on login node with internet):
  python data/prepare_data.py --output_dir /scratch/gpfs/HENDERSON/zs7353/ssm_ttt/data_cache --num_tokens 2000000000
"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset


CHUNK_SIZE = 50_000_000


def get_tokenizer(name="EleutherAI/gpt-neox-20b"):
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_train_data(tokenizer, output_dir, num_tokens=2_000_000_000,
                       dataset_name="monology/pile-uncopyrighted"):
    """Download, tokenize, and save training data incrementally to memmap."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "train.bin")

    if os.path.exists(out_path):
        existing = np.memmap(out_path, dtype=np.uint16, mode='r')
        print(f"Train data already exists: {len(existing):,} tokens")
        if len(existing) >= num_tokens:
            print("Sufficient tokens already downloaded.")
            return out_path
        print(f"Need {num_tokens - len(existing):,} more tokens, re-downloading...")

    print(f"Downloading and tokenizing {num_tokens:,} tokens...")
    dataset = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)

    fp = np.memmap(out_path, dtype=np.uint16, mode='w+', shape=(num_tokens,))
    
    buf = np.empty(CHUNK_SIZE, dtype=np.uint16)
    buf_pos = 0
    written = 0
    eos_id = tokenizer.eos_token_id

    for example in tqdm(dataset, desc="Tokenizing"):
        text = example.get("text", "")
        if not text or len(text) < 100:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(eos_id)
        arr = np.array(tokens, dtype=np.uint16)

        remaining_space = num_tokens - written - buf_pos
        if len(arr) > remaining_space:
            arr = arr[:remaining_space]

        end = buf_pos + len(arr)
        if end <= CHUNK_SIZE:
            buf[buf_pos:end] = arr
            buf_pos = end
        else:
            space = CHUNK_SIZE - buf_pos
            buf[buf_pos:] = arr[:space]
            fp[written:written + CHUNK_SIZE] = buf
            fp.flush()
            written += CHUNK_SIZE
            print(f"  Written {written:,} / {num_tokens:,} tokens ({100*written/num_tokens:.1f}%)")

            leftover = arr[space:]
            buf_pos = len(leftover)
            buf[:buf_pos] = leftover

        if written + buf_pos >= num_tokens:
            break

    if buf_pos > 0:
        actual_end = min(written + buf_pos, num_tokens)
        fp[written:actual_end] = buf[:actual_end - written]
        fp.flush()
        written = actual_end

    actual_tokens = written
    del fp

    if actual_tokens < num_tokens:
        print(f"Dataset exhausted at {actual_tokens:,} tokens (requested {num_tokens:,})")
        final = np.memmap(out_path, dtype=np.uint16, mode='r+', shape=(actual_tokens,))
        final_copy = np.array(final[:actual_tokens])
        del final
        os.remove(out_path)
        fp2 = np.memmap(out_path, dtype=np.uint16, mode='w+', shape=(actual_tokens,))
        fp2[:] = final_copy
        fp2.flush()
        del fp2

    meta_path = os.path.join(output_dir, "train_meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"num_tokens={actual_tokens}\n")
        f.write(f"dtype=uint16\n")
        f.write(f"vocab_size={tokenizer.vocab_size}\n")
        f.write(f"eos_token_id={eos_id}\n")

    print(f"Done. Saved {actual_tokens:,} tokens to {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB)")
    return out_path


def prepare_val_data(tokenizer, output_dir, max_docs=200,
                     dataset_name="monology/pile-uncopyrighted"):
    """Download, tokenize, and save validation data."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "val.bin")
    offsets_path = os.path.join(output_dir, "val_offsets.npy")

    if os.path.exists(out_path) and os.path.exists(offsets_path):
        print(f"Validation data already exists at {out_path}")
        return out_path

    print(f"Downloading validation data (up to {max_docs} long docs)...")

    try:
        dataset = load_dataset(dataset_name, split="validation", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"No validation split found ({e}), using test split...")
        try:
            dataset = load_dataset(dataset_name, split="test", streaming=True, trust_remote_code=True)
        except Exception:
            print("No val/test split. Will use first 1% of train as validation.")
            dataset = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)

    all_tokens = []
    offsets = [0]
    count = 0
    eos_id = tokenizer.eos_token_id

    for example in tqdm(dataset, desc="Tokenizing val"):
        text = example.get("text", "")
        if not text:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) >= 4096:
            tokens = tokens[:32768]
            all_tokens.extend(tokens)
            offsets.append(len(all_tokens))
            count += 1

        if count >= max_docs:
            break

    all_tokens = np.array(all_tokens, dtype=np.uint16)
    offsets = np.array(offsets, dtype=np.int64)

    print(f"Saving {count} validation documents ({len(all_tokens):,} tokens)")

    fp = np.memmap(out_path, dtype=np.uint16, mode='w+', shape=all_tokens.shape)
    fp[:] = all_tokens[:]
    fp.flush()
    del fp

    np.save(offsets_path, offsets)
    print(f"Done. Val data: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/data_cache")
    parser.add_argument("--num_tokens", type=int, default=2_000_000_000,
                        help="Number of training tokens to prepare")
    parser.add_argument("--max_val_docs", type=int, default=200)
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.tokenizer)
    print(f"Tokenizer: {args.tokenizer}, vocab_size: {tokenizer.vocab_size}")

    prepare_train_data(tokenizer, args.output_dir, num_tokens=args.num_tokens)
    prepare_val_data(tokenizer, args.output_dir, max_docs=args.max_val_docs)
