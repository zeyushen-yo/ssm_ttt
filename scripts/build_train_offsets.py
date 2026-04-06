"""
Build train_offsets.npy from existing train.bin by scanning for EOS tokens.
This avoids re-downloading the full training set.
Also symlinks train.bin and train_meta.txt into data_cache_v2.
"""
import os
import sys
import numpy as np

SRC_DIR = "/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/data_cache"
DST_DIR = "/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/data_cache_v2"
EOS_ID = 0

os.makedirs(DST_DIR, exist_ok=True)

train_bin = os.path.join(SRC_DIR, "train.bin")
meta_file = os.path.join(SRC_DIR, "train_meta.txt")

num_tokens = None
with open(meta_file) as f:
    for line in f:
        if line.startswith("num_tokens="):
            num_tokens = int(line.strip().split("=")[1])
assert num_tokens is not None, "Could not read num_tokens from meta"
print(f"Scanning {num_tokens:,} tokens for EOS boundaries...")

data = np.memmap(train_bin, dtype=np.uint16, mode='r')

CHUNK = 100_000_000
offsets = [0]
n_eos = 0

for start in range(0, num_tokens, CHUNK):
    end = min(start + CHUNK, num_tokens)
    chunk = np.array(data[start:end])
    eos_pos = np.where(chunk == EOS_ID)[0]
    for p in eos_pos:
        abs_pos = start + int(p) + 1
        if abs_pos <= num_tokens:
            offsets.append(abs_pos)
            n_eos += 1
    pct = 100 * end / num_tokens
    print(f"  Scanned {end:,} / {num_tokens:,} ({pct:.1f}%), found {n_eos:,} documents so far")

if offsets[-1] != num_tokens:
    offsets.append(num_tokens)

offsets = np.array(offsets, dtype=np.int64)

doc_lens = offsets[1:] - offsets[:-1]
print(f"\nTotal documents: {len(offsets)-1:,}")
print(f"Mean doc length: {doc_lens.mean():.0f} tokens")
print(f"Median doc length: {np.median(doc_lens):.0f} tokens")
print(f"Min doc length: {doc_lens.min()} tokens")
print(f"Max doc length: {doc_lens.max()} tokens")
print(f"Docs >= 256 tokens: {(doc_lens >= 256).sum():,}")
print(f"Docs >= 2048 tokens: {(doc_lens >= 2048).sum():,}")

offsets_path = os.path.join(DST_DIR, "train_offsets.npy")
np.save(offsets_path, offsets)
print(f"\nSaved {offsets_path}")

dst_bin = os.path.join(DST_DIR, "train.bin")
dst_meta = os.path.join(DST_DIR, "train_meta.txt")
if not os.path.exists(dst_bin):
    os.symlink(train_bin, dst_bin)
    print(f"Symlinked {dst_bin} -> {train_bin}")
if not os.path.exists(dst_meta):
    with open(dst_meta, "w") as f:
        f.write(f"num_tokens={num_tokens}\n")
        f.write(f"num_documents={len(offsets)-1}\n")
        f.write(f"dtype=uint16\n")
        f.write(f"vocab_size=50254\n")
        f.write(f"eos_token_id={EOS_ID}\n")
    print(f"Wrote {dst_meta}")

print("\nDone!")
