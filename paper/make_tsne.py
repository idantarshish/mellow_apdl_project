"""
Extract HTSAT embeddings for Clotho validation clips and visualize with t-SNE.
Answers: do clips with similar sounds cluster together?

Run from project root:
  /home/yandex/APDL2526a/sapirtalmi/mellow_env/bin/python paper/make_tsne.py

Output: paper/figures/fig10_tsne.png
"""

import os, sys, json, re, random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.append('')
os.makedirs('paper/figures', exist_ok=True)

DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_PATH   = 'outputs/2026-04-01_12-23-01/model--epo-10.ckpt'
DATA_ROOT   = '/home/yandex/APDL2526a/idantarshish/mellow_apdl_project/data'
DATAFILE    = 'datafiles/val_clotho.json'
SR          = 32000
DURATION    = 10
MAX_CLIPS   = 400   # use up to 400 unique clips (full val split has ~1046)
HF_HOME     = '/home/yandex/APDL2526a/sapirtalmi/hf_cache'

os.environ['HF_HOME'] = HF_HOME
os.environ['TRANSFORMERS_CACHE'] = HF_HOME

# ── Sound category definitions (keyword → category label) ─────────────────────
CATEGORIES = {
    'rain':      ['rain', 'drizzle', 'shower', 'downpour', 'raindrop'],
    'water':     ['water', 'stream', 'river', 'waterfall', 'ocean', 'wave',
                  'splash', 'flowing', 'brook', 'faucet', 'dripping'],
    'birds':     ['bird', 'chirp', 'tweet', 'songbird', 'crow', 'owl',
                  'chirping', 'singing'],
    'traffic':   ['traffic', 'car', 'vehicle', 'engine', 'truck', 'bus',
                  'motorcycle', 'driving', 'road', 'highway'],
    'wind':      ['wind', 'breeze', 'blowing', 'gust', 'rustling leaves'],
    'people':    ['people', 'crowd', 'voice', 'talking', 'speech', 'children',
                  'laughter', 'conversation', 'shouting'],
    'machinery': ['machine', 'machinery', 'motor', 'engine', 'fan', 'drill',
                  'construction', 'industrial', 'generator'],
    'music':     ['music', 'singing', 'instrument', 'guitar', 'piano',
                  'drum', 'melody', 'song'],
}

COLORS = {
    'rain':      '#1f77b4',
    'water':     '#17becf',
    'birds':     '#2ca02c',
    'traffic':   '#d62728',
    'wind':      '#9467bd',
    'people':    '#e377c2',
    'machinery': '#ff7f0e',
    'music':     '#bcbd22',
    'other':     '#7f7f7f',
}

def infer_category(caption):
    caption = caption.lower()
    for cat, keywords in CATEGORIES.items():
        if any(kw in caption for kw in keywords):
            return cat
    return 'other'

# ── Step 1: Collect unique clips + infer category from captions ───────────────
print("Step 1: Loading validation clips...")

clip_to_captions = {}
with open(DATAFILE, 'r') as f:
    chunk = f.read(5_000_000)  # read first 5MB to get a sample

# Parse filepath1 and answer pairs
pairs = re.findall(r'"filepath1"\s*:\s*"([^"]+)".*?"answer"\s*:\s*"([^"]+)"', chunk)
for fp, ans in pairs:
    if fp not in clip_to_captions:
        clip_to_captions[fp] = []
    clip_to_captions[fp].append(ans)

print(f"  Found {len(clip_to_captions)} unique clips in sample")

# Assign category and filter to existing files
clips = []
for fp, captions in clip_to_captions.items():
    full_path = os.path.join(DATA_ROOT, fp) if not fp.startswith('/') else fp
    if os.path.exists(full_path):
        cat = infer_category(' '.join(captions))
        clips.append({'path': full_path, 'category': cat, 'caption': captions[0]})

print(f"  {len(clips)} clips found on disk")

# Balance categories and cap total
random.seed(42)
random.shuffle(clips)
clips = clips[:MAX_CLIPS]

cat_counts = {}
for c in clips:
    cat_counts[c['category']] = cat_counts.get(c['category'], 0) + 1
print(f"  Category distribution: {cat_counts}")

# ── Step 2: Load HTSAT encoder from checkpoint ────────────────────────────────
print("\nStep 2: Loading HTSAT encoder...")

from models.mellow import AudioEncoder

audio_encoder = AudioEncoder(
    audioenc_name='HTSAT',
    d_in=768,
    d_out=576,
    use_pretrained_audioencoder=True,
    freeze_audio_encoder_weights=True,
    pretrained_audioencoder_path='htsat',
).to(DEVICE)

# Load finetuned weights from checkpoint
ckpt = torch.load(CKPT_PATH, map_location='cpu')
state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))

# Extract only audio_encoder keys
ae_state = {k.replace('audio_encoder.', ''): v
            for k, v in state.items()
            if k.startswith('audio_encoder.')}
audio_encoder.load_state_dict(ae_state, strict=False)
audio_encoder.eval()
print(f"  Loaded checkpoint: {CKPT_PATH}")

# ── Step 3: Extract embeddings ────────────────────────────────────────────────
print(f"\nStep 3: Extracting embeddings for {len(clips)} clips...")

import librosa

def load_audio(path):
    y, _ = librosa.load(path, sr=SR, duration=DURATION, mono=True)
    target = SR * DURATION
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    return y[:target]

htsat_embeddings  = []  # raw HTSAT CLS embeddings (before projection)
proj_embeddings   = []  # after projection
categories        = []
captions_used     = []

with torch.no_grad():
    for i, clip in enumerate(clips):
        if i % 50 == 0:
            print(f"  [{i}/{len(clips)}]")
        try:
            y = load_audio(clip['path'])

            # Skip corrupted/OOD
            if np.isnan(y).any() or np.isinf(y).any() or np.abs(y).max() >= 1.5:
                continue

            wav = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Get raw HTSAT embedding (before projection)
            out_dict = audio_encoder.base(wav)
            htsat_emb = out_dict['embedding']  # (1, T+1, 768) or (1, 768)
            # Take mean over time dim if needed
            if htsat_emb.dim() == 3:
                htsat_emb = htsat_emb.mean(dim=1)  # (1, 768)
            htsat_embeddings.append(htsat_emb.squeeze(0).cpu().numpy())

            # Get projected embedding
            proj_emb, _, _ = audio_encoder(wav)
            if proj_emb.dim() == 3:
                proj_emb = proj_emb.mean(dim=1)
            proj_embeddings.append(proj_emb.squeeze(0).cpu().numpy())

            categories.append(clip['category'])
            captions_used.append(clip['caption'])

        except Exception as e:
            continue

print(f"  Successfully extracted {len(htsat_embeddings)} embeddings")

htsat_arr = np.array(htsat_embeddings)
proj_arr  = np.array(proj_embeddings)
cats      = np.array(categories)

# ── Step 4: t-SNE ─────────────────────────────────────────────────────────────
print("\nStep 4: Running t-SNE...")

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def run_tsne(X, perplexity=30):
    X_scaled = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                n_iter=1000, learning_rate='auto', init='pca')
    return tsne.fit_transform(X_scaled)

tsne_htsat = run_tsne(htsat_arr)
tsne_proj  = run_tsne(proj_arr)
print("  t-SNE done")

# ── Step 5: Plot ──────────────────────────────────────────────────────────────
print("\nStep 5: Plotting...")

unique_cats = sorted(set(categories))
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('t-SNE of HTSAT Audio Embeddings — Clotho Validation Set',
             fontsize=14, fontweight='bold')

for ax, emb2d, title in zip(
        axes,
        [tsne_htsat, tsne_proj],
        ['Raw HTSAT Embeddings\n(before projection)',
         'Projected Embeddings\n(after MLP projection → LM space)']):

    for cat in unique_cats:
        mask = cats == cat
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
                   c=COLORS.get(cat, '#7f7f7f'),
                   label=f'{cat} (n={mask.sum()})',
                   alpha=0.7, s=25, edgecolors='none')

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('t-SNE dim 1', fontsize=10)
    ax.set_ylabel('t-SNE dim 2', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)

# Shared legend
handles = [mpatches.Patch(color=COLORS.get(c, '#7f7f7f'), label=c)
           for c in unique_cats]
fig.legend(handles=handles, loc='lower center', ncol=len(unique_cats),
           fontsize=9, bbox_to_anchor=(0.5, -0.03))

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('paper/figures/fig10_tsne.pdf', bbox_inches='tight', dpi=150)
plt.savefig('paper/figures/fig10_tsne.png', bbox_inches='tight', dpi=150)
print("Saved paper/figures/fig10_tsne.png")

# ── Step 6: Cluster quality metric (silhouette score) ─────────────────────────
try:
    from sklearn.metrics import silhouette_score
    label_ids = [unique_cats.index(c) for c in categories]
    sil_htsat = silhouette_score(htsat_arr, label_ids)
    sil_proj  = silhouette_score(proj_arr,  label_ids)
    print(f"\nSilhouette scores (higher = better separation):")
    print(f"  HTSAT embeddings: {sil_htsat:.4f}")
    print(f"  Projected embeds: {sil_proj:.4f}")
    print("  (>0.1 = meaningful structure, >0.3 = clear clusters)")
except Exception as e:
    print(f"  Silhouette skipped: {e}")

print("\nDone.")
