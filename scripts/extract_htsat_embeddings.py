"""
Extract HTSAT embeddings for all unique Clotho validation clips
and visualize with t-SNE and UMAP.

Usage (on a GPU node):
  python scripts/extract_htsat_embeddings.py

Outputs:
  outputs/htsat_embeddings/embeddings.npz   -- latent_output + clipwise_output + filenames
  outputs/htsat_embeddings/tsne.png
  outputs/htsat_embeddings/umap.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import torchaudio
import numpy as np
import random
from tqdm import tqdm

# ── model setup ──────────────────────────────────────────────────────────────
from models.htsat import HTSATWrapper

HTSAT_CKPT = "/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/htsat/HTSAT_AudioSet_Saved_1.ckpt"
DATA_ROOT  = "/home/yandex/APDL2526a/idantarshish/mellow_apdl_project/data"
VAL_JSON   = "/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/datafiles/val_clotho.json"
OUT_DIR    = "/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/outputs/htsat_embeddings"
SAMPLE_RATE = 32000
MAX_SAMPLES = SAMPLE_RATE * 10   # 10 s

os.makedirs(OUT_DIR, exist_ok=True)

# ── load model ───────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

wrapper = HTSATWrapper().to(device)

ckpt = torch.load(HTSAT_CKPT, map_location="cpu")
new_ckpt = {k[10:]: v for k, v in ckpt["state_dict"].items()}
wrapper.htsat.load_state_dict(new_ckpt)
wrapper.eval()
print("HTSAT loaded.")

# ── collect unique clips ──────────────────────────────────────────────────────
with open(VAL_JSON) as f:
    data = json.load(f)

unique_clips = sorted(set(
    fp.replace('\\', '/') for item in data
    for fp in (item["filepath1"], item["filepath2"])
    if fp != ""
))
print(f"Unique clips: {len(unique_clips)}")

# ── audio helper ──────────────────────────────────────────────────────────────
def load_clip(rel_path):
    full = os.path.join(DATA_ROOT, rel_path.replace('\\', '/'))
    wav, sr = torchaudio.load(full, channels_first=True)
    # mix to mono
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    # resample
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    # pad / trim to 10s
    if wav.shape[1] >= MAX_SAMPLES:
        wav = wav[:, :MAX_SAMPLES]
    else:
        pad = torch.zeros(1, MAX_SAMPLES)
        pad[:, :wav.shape[1]] = wav
        wav = pad
    return wav  # [1, MAX_SAMPLES]

# ── extract embeddings ────────────────────────────────────────────────────────
latents   = []   # 768-dim pre-head representation
clipwise  = []   # 527-dim AudioSet sigmoid scores
filenames = []

BATCH = 16

with torch.no_grad():
    for i in tqdm(range(0, len(unique_clips), BATCH), desc="Extracting"):
        batch_paths = unique_clips[i : i + BATCH]
        wavs = []
        valid_paths = []
        for rp in batch_paths:
            try:
                w = load_clip(rp)
                # guard: skip clips with out-of-range amplitude
                if not torch.isfinite(w).all() or w.abs().max() >= 1.5:
                    continue
                wavs.append(w)
                valid_paths.append(rp)
            except Exception as e:
                print(f"  skip {rp}: {e}")

        if not wavs:
            continue

        x = torch.stack(wavs).squeeze(1).to(device)  # [B, T]
        out = wrapper(x)                           # calls HTSATWrapper.forward
        lat = out["latent_output"]                 # [B, 768]
        cw  = out["clipwise_output"]               # [B, 527]

        latents.extend(lat.cpu().float().numpy())
        clipwise.extend(cw.cpu().float().numpy())
        filenames.extend(valid_paths)

latents  = np.array(latents,  dtype=np.float32)
clipwise = np.array(clipwise, dtype=np.float32)
print(f"Extracted {len(filenames)} clips — latent shape: {latents.shape}")

np.savez(
    os.path.join(OUT_DIR, "embeddings.npz"),
    latents=latents,
    clipwise=clipwise,
    filenames=np.array(filenames),
)
print(f"Saved embeddings to {OUT_DIR}/embeddings.npz")

# ── AudioSet top-level category labels (simplified) ──────────────────────────
# Map top-1 AudioSet class → broad category for colouring.
# Indices based on AudioSet ontology order used by HTSAT (527 classes).
CATEGORY_RANGES = {
    "Human voice":   (0,  72),
    "Animal":        (72, 132),
    "Music":         (132, 200),
    "Mechanical":    (200, 290),
    "Vehicle":       (290, 340),
    "Domestic":      (340, 380),
    "Nature":        (380, 430),
    "Other":         (430, 527),
}

def top1_category(cw_vec):
    idx = int(np.argmax(cw_vec))
    for cat, (lo, hi) in CATEGORY_RANGES.items():
        if lo <= idx < hi:
            return cat
    return "Other"

categories = [top1_category(cw) for cw in clipwise]

# ── dimensionality reduction + plots ─────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

CAT_LIST = list(CATEGORY_RANGES.keys())
COLORS   = plt.cm.tab10(np.linspace(0, 1, len(CAT_LIST)))
color_map = {cat: COLORS[i] for i, cat in enumerate(CAT_LIST)}
point_colors = [color_map[c] for c in categories]

def plot_2d(coords, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    for cat in CAT_LIST:
        mask = [c == cat for c in categories]
        if not any(mask):
            continue
        xs = coords[mask, 0]
        ys = coords[mask, 1]
        ax.scatter(xs, ys, s=12, alpha=0.65, label=cat,
                   color=color_map[cat])
    ax.legend(markerscale=2, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

# t-SNE
print("Running t-SNE …")
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, n_jobs=4)
tsne_coords = tsne.fit_transform(latents)
plot_2d(tsne_coords, "HTSAT latent embeddings — t-SNE (Clotho val, ~1390 clips)",
        os.path.join(OUT_DIR, "tsne.png"))

# UMAP
print("Running UMAP …")
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
umap_coords = reducer.fit_transform(latents)
plot_2d(umap_coords, "HTSAT latent embeddings — UMAP (Clotho val, ~1390 clips)",
        os.path.join(OUT_DIR, "umap.png"))

print("Done.")
