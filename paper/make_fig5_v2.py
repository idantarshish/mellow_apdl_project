"""
Generate fig5_spectrogram_pairs.png using real epoch-10 model outputs.

Loads 200 random pairs from the Clotho validation set, runs inference,
scores visual distinctiveness + output quality, selects 3 representative
pairs (good / partial failure / grounding failure), and plots publication-
quality log-mel spectrograms with model output text boxes.

Run from project root:
  /home/yandex/APDL2526a/sapirtalmi/mellow_env/bin/python paper/make_fig5_v2.py

Or submit via SLURM (for GPU):
  sbatch paper/slurm_fig5.slurm
"""

import os
import sys
import json
import random
import re
import numpy as np

os.environ.setdefault('HF_HOME', '/home/yandex/APDL2526a/sapirtalmi/hf_cache')
os.environ.setdefault('TRANSFORMERS_CACHE', '/home/yandex/APDL2526a/sapirtalmi/hf_cache')

import torch
import torchaudio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, '/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project')
from models.mellow import Mellow
from models.generate import generate_greedy_batch

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT  = '/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/outputs/2026-04-01_12-23-01/model--epo-10.ckpt'
VAL_JSON    = '/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/datafiles/val_clotho.json'
DATA_ROOT   = '/home/yandex/APDL2526a/idantarshish/mellow_apdl_project/data'
HTSAT_PATH  = '/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/htsat'
TOKENIZER   = 'HuggingFaceTB/SmolLM2-135M'
OUT_DIR     = 'paper/figures'
SR          = 32000
DURATION    = 10
N_MELS      = 128
IP_TEXT_LEN = 129
N_SAMPLES   = SR * DURATION
NUM_PAIRS   = 200
RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

model = Mellow(
    audioenc_name='HTSAT',
    d_in=768,
    text_decoder=TOKENIZER,
    prefix_length=40,
    freeze_text_decoder_weights=False,
    d_out=576,
    use_pretrained_audioencoder=True,
    freeze_audio_encoder_weights=True,
    pretrained_audioencoder_path=HTSAT_PATH,
)

ckpt = torch.load(CHECKPOINT, map_location=device)
state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
model.load_state_dict(state, strict=True)
model.eval()
model = model.to(device)
print(f"  Checkpoint: {Path(CHECKPOINT).name}")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
tokenizer.add_special_tokens({'pad_token': '!'})
print("  Tokenizer ready")

# ── Audio / spectrogram helpers ───────────────────────────────────────────────
def load_audio_tensor(filepath):
    """Return mono float32 tensor [1, N_SAMPLES] at SR=32000."""
    audio, rate = torchaudio.load(filepath, channels_first=True)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if rate != SR:
        audio = torchaudio.functional.resample(audio, orig_freq=rate, new_freq=SR)
    if audio.shape[1] > N_SAMPLES:
        audio = audio[:, :N_SAMPLES]
    else:
        pad = torch.zeros(1, N_SAMPLES - audio.shape[1])
        audio = torch.cat([audio, pad], dim=1)
    return audio  # [1, N_SAMPLES]


def tokenize_input(text, max_len=IP_TEXT_LEN):
    text_with_eos = text + ' <|endoftext|>'
    return tokenizer.encode_plus(
        text=text_with_eos, add_special_tokens=True, truncation=True,
        max_length=max_len, pad_to_max_length=True, return_tensors="pt"
    )


def log_mel_spec(y_np, sr=SR, n_mels=N_MELS, hop_length=512):
    """Log-mel spectrogram [n_mels, T] in dB, referenced to max."""
    S = librosa.feature.melspectrogram(
        y=y_np, sr=sr, n_mels=n_mels, hop_length=hop_length, fmax=sr // 2
    )
    return librosa.power_to_db(S + 1e-10, ref=np.max)


def spectral_distinctiveness(y1, y2, sr=SR):
    """
    Score how visually distinct two audio clips are.
    Combines: spectral centroid diff, flatness diff (tonal vs noise), RMS ratio.
    Higher = more distinct.
    """
    sc1 = float(np.median(librosa.feature.spectral_centroid(y=y1, sr=sr)[0]))
    sc2 = float(np.median(librosa.feature.spectral_centroid(y=y2, sr=sr)[0]))
    centroid_diff = abs(sc1 - sc2) / (sr / 2)  # normalised 0-1

    sf1 = float(np.median(librosa.feature.spectral_flatness(y=y1)[0]))
    sf2 = float(np.median(librosa.feature.spectral_flatness(y=y2)[0]))
    flatness_diff = abs(sf1 - sf2)              # 0-1

    rms1 = float(np.sqrt(np.mean(y1 ** 2))) + 1e-10
    rms2 = float(np.sqrt(np.mean(y2 ** 2))) + 1e-10
    rms_ratio = max(rms1, rms2) / min(rms1, rms2)
    rms_score = min(np.log(rms_ratio) / np.log(100), 1.0)  # log-scale, cap at 1

    return centroid_diff + flatness_diff + 0.5 * rms_score


def score_output(text):
    """
    Heuristic output quality score.
    Returns (float 0-1, category str: 'good' | 'partial' | 'fail')
    """
    text = text.strip()
    words = text.split()

    if len(words) < 3:
        return 0.0, 'fail'

    # Strip MC-option prefix for content analysis
    clean = re.sub(r'^\s*[a-cA-C]\)\s*', '', text).strip()
    clean_words = clean.split()

    if len(clean_words) < 3:
        return 0.1, 'fail'

    # Repetition check (bigram uniqueness)
    if len(clean_words) > 10:
        bigrams = [f"{clean_words[i]} {clean_words[i+1]}" for i in range(len(clean_words)-1)]
        if len(set(bigrams)) / len(bigrams) < 0.5:
            return 0.1, 'fail'

    # Vocabulary check for meaningful content
    good_words = {
        'while', 'whereas', 'compared', 'versus', 'unlike', 'however',
        'contrast', 'difference', 'both', 'first', 'second', 'audio',
        'sound', 'features', 'contains', 'whereas', 'has', 'have',
        'hear', 'noise', 'frequency', 'rhythm', 'pattern', 'tone',
    }
    overlap = len(set(w.lower() for w in clean_words) & good_words)

    has_option_prefix = bool(re.match(r'^\s*[a-cA-C]\)', text))

    if len(clean_words) >= 8 and overlap >= 2:
        if has_option_prefix:
            return 0.55, 'partial'
        return 0.85, 'good'
    elif len(clean_words) >= 4:
        return 0.35, 'partial'
    else:
        return 0.15, 'fail'


# ── Load validation pairs ─────────────────────────────────────────────────────
print(f"\nLoading {VAL_JSON}...")
with open(VAL_JSON, 'r') as f:
    all_pairs = json.load(f)
print(f"  Total pairs: {len(all_pairs)}")

sample_pairs = random.sample(all_pairs, min(NUM_PAIRS, len(all_pairs)))
print(f"  Sampled: {len(sample_pairs)} pairs for inference")

# ── Run inference ─────────────────────────────────────────────────────────────
print(f"\nRunning inference...")
results = []

for i, pair in enumerate(sample_pairs):
    rel1 = pair.get('filepath1', '').replace('\\', '/')
    rel2 = pair.get('filepath2', '').replace('\\', '/')
    if not rel1 or not rel2:
        continue

    fp1 = os.path.join(DATA_ROOT, rel1)
    fp2 = os.path.join(DATA_ROOT, rel2)

    if not os.path.exists(fp1) or not os.path.exists(fp2):
        continue

    try:
        audio1 = load_audio_tensor(fp1)
        audio2 = load_audio_tensor(fp2)

        a1_np = audio1.squeeze(0).numpy()
        a2_np = audio2.squeeze(0).numpy()

        # Skip OOD audio (NaN / Inf / excessive amplitude)
        if (np.isnan(a1_np).any() or np.isinf(a1_np).any() or np.abs(a1_np).max() >= 1.5 or
                np.isnan(a2_np).any() or np.isinf(a2_np).any() or np.abs(a2_np).max() >= 1.5):
            continue

        dist_score = spectral_distinctiveness(a1_np, a2_np)

        # Tokenize the question prompt (same normalisation as dataset)
        raw_input = pair.get('input', 'explain the difference in few words')
        if raw_input == 'explain the difference in few words':
            input_text = 'Explain the difference between the two audios in few words.'
        elif raw_input == 'explain the difference in a sentence':
            input_text = 'Explain the difference between the two audios in one extended sentence.'
        elif raw_input == 'explain the difference in detail':
            input_text = 'Explain the difference between the two audios in detail.'
        else:
            input_text = raw_input
        tok_input = tokenize_input(input_text.lower())

        # Build input_dict — audio shape [1, N_SAMPLES], text shape [1, IP_TEXT_LEN]
        input_dict = {
            'audio1': audio1.to(device),
            'audio2': audio2.to(device),
            'input':  {k: v.to(device) for k, v in tok_input.items()},
            'answer': {k: v.to(device) for k, v in tok_input.items()},  # placeholder
        }

        with torch.no_grad():
            prefix, _, _ = model.generate_prefix_inference(input_dict)
            generated = generate_greedy_batch(
                model, tokenizer, embed=prefix,
                strip_option_prefix=True,
            )

        gen_text = generated[0] if generated else ''
        q_score, category = score_output(gen_text)

        results.append({
            'fp1': fp1, 'fp2': fp2,
            'a1_np': a1_np, 'a2_np': a2_np,
            'generated':  gen_text,
            'reference':  pair.get('answer', ''),
            'dist_score': dist_score,
            'q_score':    q_score,
            'category':   category,
            'name1': Path(fp1).stem,
            'name2': Path(fp2).stem,
        })

        if (i + 1) % 25 == 0 or (i + 1) == len(sample_pairs):
            cats = {r['category']: sum(1 for r2 in results if r2['category'] == r['category'])
                    for r in results}
            print(f"  [{i+1}/{len(sample_pairs)}] {len(results)} valid | {cats}")

    except Exception as e:
        print(f"  [{i+1}] Error on {Path(fp1).name}: {e}")
        continue

print(f"\nInference complete. {len(results)} valid pairs.")

# ── Select 3 representative pairs ─────────────────────────────────────────────
MIN_DIST = 0.04  # minimum distinctiveness to avoid boring pairs

def pick_best(category, min_dist=MIN_DIST):
    cands = [r for r in results if r['category'] == category and r['dist_score'] >= min_dist]
    if not cands:
        cands = [r for r in results if r['category'] == category]
    if not cands:
        return None
    # Prefer high distinctiveness and high quality (within category)
    cands.sort(key=lambda r: (r['dist_score'], r['q_score']), reverse=True)
    return cands[0]

selected = [
    ('Good output',       pick_best('good')),
    ('Partial failure',   pick_best('partial')),
    ('Grounding failure', pick_best('fail')),
]

# If any category is missing, fill from the highest-distinctiveness leftovers
used = {id(r) for _, r in selected if r is not None}
all_sorted = sorted(results, key=lambda r: r['dist_score'], reverse=True)
spare = [r for r in all_sorted if id(r) not in used]

for idx, (label, r) in enumerate(selected):
    if r is None and spare:
        selected[idx] = (label, spare.pop(0))

print("\nSelected pairs for figure:")
for label, r in selected:
    if r:
        print(f"  [{label}] dist={r['dist_score']:.3f} q={r['q_score']:.2f} "
              f"cat={r['category']} | \"{r['generated'][:70]}\"")

# ── Plot ──────────────────────────────────────────────────────────────────────
print("\nPlotting...")

ROW_COLORS = ['#2ca02c', '#ff7f0e', '#d62728']  # green / orange / red

fig = plt.figure(figsize=(10, 8))
fig.patch.set_facecolor('white')

outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.70, top=0.92, bottom=0.07,
                          left=0.12, right=0.97)

for row_idx, (row_label, r) in enumerate(selected):
    if r is None:
        continue

    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row_idx], wspace=0.25)

    S1 = log_mel_spec(r['a1_np'])
    S2 = log_mel_spec(r['a2_np'])
    row_color = ROW_COLORS[row_idx]
    name1 = r['name1'][:20]
    name2 = r['name2'][:20]

    # ── Left spectrogram ──
    ax1 = fig.add_subplot(inner[0])
    ax1.imshow(S1, aspect='auto', origin='lower', cmap='magma',
               extent=[0, DURATION, 0, N_MELS], vmin=S1.min(), vmax=0)
    ax1.set_xlabel('Time (s)', fontsize=8)
    ax1.set_ylabel('Mel frequency bin', fontsize=8)
    ax1.set_title(f'Audio 1: {name1}', fontsize=7.5, color='#333333',
                  style='italic', pad=3)
    ax1.tick_params(labelsize=7)

    # ── Right spectrogram ──
    ax2 = fig.add_subplot(inner[1])
    ax2.imshow(S2, aspect='auto', origin='lower', cmap='magma',
               extent=[0, DURATION, 0, N_MELS], vmin=S2.min(), vmax=0)
    ax2.set_xlabel('Time (s)', fontsize=8)
    ax2.set_ylabel('', fontsize=8)
    ax2.set_title(f'Audio 2: {name2}', fontsize=7.5, color='#333333',
                  style='italic', pad=3)
    ax2.tick_params(labelsize=7)
    ax2.set_yticks([])

    # ── Row label on the left margin ──
    ax1.annotate(
        row_label,
        xy=(-0.28, 0.5), xycoords='axes fraction',
        fontsize=8.5, fontweight='bold', color=row_color,
        rotation=90, va='center', ha='center',
    )

    # ── Model output text box below the row ──
    gen = r['generated'].strip()
    if len(gen) > 130:
        gen = gen[:127] + '...'

    row_pos = outer[row_idx].get_position(fig)
    fig.text(
        0.5, row_pos.y0 - 0.006,
        f'Model output: \u201c{gen}\u201d',
        ha='center', va='top',
        fontsize=8, style='italic', color='#1a3a5c',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#e8f4fd',
                  edgecolor=row_color, linewidth=1.2, alpha=0.92),
    )

fig.suptitle('Epoch-10 Spectrogram Pairs: Model Output Quality Range',
             fontsize=11, fontweight='bold', y=0.97)

out_png = os.path.join(OUT_DIR, 'fig5_spectrogram_pairs.png')
out_pdf = os.path.join(OUT_DIR, 'fig5_spectrogram_pairs.pdf')
plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig(out_pdf, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
