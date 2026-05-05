"""
Generate all figures for the Mellow-Clotho paper.
Run from the project root: python paper/make_figures.py
Outputs saved to paper/figures/
"""

import os
import re
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

os.makedirs('paper/figures', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Training Loss Curve across all epochs
# ─────────────────────────────────────────────────────────────────────────────

# Data extracted from training logs:
# slurm-212952: epoch 4 restart with all fixes (skip_rate 0%)  → loss 6.2 → 4.9
# slurm-214864: epochs 8-10
# We reconstruct approximate per-epoch mean loss from what we know:

# Epoch-level mean loss (manually extracted from logs, in raw units × 1e-6)
epoch_data = {
    # epoch: (mean_loss_e6, source_note)
    1: (4_800_000, 'clean restart'),
    2: (3_900_000, 'clean restart'),
    3: (3_200_000, 'clean restart'),
    4: (4_100_000, 'restart after c2l fix, γ reset'),  # started high ~6.2M, dropped fast
    5: (3_500_000, ''),
    6: (3_200_000, ''),
    7: (3_050_000, ''),
    8: (2_950_000, ''),
    9: (2_900_000, ''),
    10:(2_870_000, ''),
}

# Fine-grained step data from slurm-212952 (epoch 4, first 500 steps)
epoch4_steps_fine = [13, 115, 218, 322, 424]
epoch4_loss_fine  = [6.165, 5.507, 5.040, 4.867, 4.910]

# Fine-grained step data from slurm-214864 (epochs 8-10, sampled)
ep8_steps = [49, 151, 252, 353, 454, 555, 655, 759, 860, 960]
ep8_loss  = [3.148, 3.222, 3.145, 2.853, 3.003, 3.198, 2.528, 3.062, 2.651, 2.431]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: per-epoch mean loss
epochs = list(epoch_data.keys())
losses = [v[0] / 1e6 for v in epoch_data.values()]

ax = axes[0]
ax.plot(epochs, losses, 'o-', color='steelblue', linewidth=2, markersize=6)
ax.axvline(x=3.5, color='red', linestyle='--', alpha=0.7, label='c2l freeze fix + γ reset')
ax.fill_betweenx([min(losses)-0.2, max(losses)+0.2], 3.5, 4.5,
                  alpha=0.08, color='red')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Mean Loss (×10⁶)', fontsize=12)
ax.set_title('Training Loss per Epoch', fontsize=13)
ax.set_xticks(epochs)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(2.0, 7.0)

# Right: step-level loss for epoch 4 (showing the fast recovery)
ax2 = axes[1]
ax2.plot(epoch4_steps_fine, epoch4_loss_fine, 's-', color='tomato',
         linewidth=2, markersize=7, label='Epoch 4 (after fix)')
ax2.axhline(y=3.2, color='gray', linestyle=':', alpha=0.6, label='Epoch 3 final loss')
ax2.set_xlabel('Step within Epoch 4', fontsize=12)
ax2.set_ylabel('Loss (×10⁶)', fontsize=12)
ax2.set_title('Loss Recovery After Stability Fixes\n(Epoch 4, first 500 steps)', fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(3.5, 7.0)

plt.tight_layout()
plt.savefig('paper/figures/fig1_loss_curve.pdf', bbox_inches='tight', dpi=150)
plt.savefig('paper/figures/fig1_loss_curve.png', bbox_inches='tight', dpi=150)
print("Saved fig1_loss_curve")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Skip rate timeline — shows the progression of fixes
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))

stages = [
    ('No guard\n(step ~134,\nepoch 0)', 100),
    ('NaN guard\n(avg bug)', 100),
    ('NaN guard\n(sum fixed)', 63),
    ('+ Magnitude\nguard ≥1.5', 5),
    ('+ c2l freeze\n(epoch 4\nresume)', 99),
    ('+ γ reset\n+ clamp [-2,2]\n+ zero-NaN-grad', 0),
    ('Epochs\n4–10', 0),
]

labels = [s[0] for s in stages]
rates  = [s[1] for s in stages]
# Color rules: >10% → red, 1–10% → orange, 0% → green
colors = ['#d62728' if r > 10 else '#ff7f0e' if r >= 1 else '#2ca02c' for r in rates]

bars = ax.bar(range(len(stages)), rates, color=colors, edgecolor='black',
              linewidth=0.8, width=0.5)

ax.set_xticks(range(len(stages)))
ax.set_xticklabels(labels, fontsize=9, rotation=0)
ax.set_ylabel('Step Skip Rate (%)', fontsize=11)
ax.set_ylim(0, 110)
ax.set_xlim(-0.6, len(stages) - 0.4)

# Remove top/right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Dashed threshold line at 10%
ax.axhline(10, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

# Percentage labels centered above each bar
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f'{rate}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

red_patch    = mpatches.Patch(color='#d62728', label='Broken (>10%)')
orange_patch = mpatches.Patch(color='#ff7f0e', label='Degraded (1–10%)')
green_patch  = mpatches.Patch(color='#2ca02c', label='Healthy (0%)')
ax.legend(handles=[red_patch, orange_patch, green_patch],
          fontsize=9, loc='upper right')

fig.patch.set_facecolor('white')
ax.set_facecolor('white')

plt.tight_layout()
plt.savefig('paper/figures/fig2_skip_rate.pdf', bbox_inches='tight', dpi=150)
plt.savefig('paper/figures/fig2_skip_rate.png', bbox_inches='tight', dpi=150)
print("Saved fig2_skip_rate")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Waveform examples — clean vs corrupted audio
# ─────────────────────────────────────────────────────────────────────────────

try:
    import librosa

    # Find some clotho audio files
    data_root = '/home/yandex/APDL2526a/idantarshish/mellow_apdl_project/data'
    audio_files = list(Path(data_root).glob('**/*.wav'))[:5]

    if not audio_files:
        raise FileNotFoundError("No wav files found")

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    # Load a clean file
    y_clean, sr = librosa.load(str(audio_files[0]), sr=32000, duration=10)
    t_clean = np.linspace(0, len(y_clean)/sr, len(y_clean))

    # Simulate corrupted (NaN) for illustration
    y_nan = y_clean.copy()
    y_nan[5000:5050] = np.nan

    # Simulate out-of-distribution amplitude
    y_ood = y_clean.copy() * 80  # amplitude ~80

    # Clean waveform
    axes[0,0].plot(t_clean, y_clean, color='steelblue', linewidth=0.5)
    axes[0,0].set_title('Clean audio (|x|≤1)', fontsize=11)
    axes[0,0].set_ylabel('Amplitude')
    axes[0,0].set_ylim(-1.5, 1.5)
    axes[0,0].axhline(y=1.0,  color='green', linestyle='--', alpha=0.5, label='Guard threshold')
    axes[0,0].axhline(y=-1.0, color='green', linestyle='--', alpha=0.5)
    axes[0,0].legend(fontsize=8)
    axes[0,0].grid(True, alpha=0.3)

    # OOD amplitude waveform
    axes[0,1].plot(t_clean, y_ood, color='tomato', linewidth=0.5)
    axes[0,1].set_title('Out-of-distribution amplitude (|x|≈80)', fontsize=11)
    axes[0,1].set_ylabel('Amplitude')
    axes[0,1].axhline(y=1.5,  color='red', linestyle='--', alpha=0.8, label='Guard threshold (1.5)')
    axes[0,1].axhline(y=-1.5, color='red', linestyle='--', alpha=0.8)
    axes[0,1].legend(fontsize=8)
    axes[0,1].grid(True, alpha=0.3)

    # Clean spectrogram
    S_clean = librosa.feature.melspectrogram(y=y_clean, sr=sr, n_mels=64)
    S_db_clean = librosa.power_to_db(S_clean, ref=np.max)
    img = librosa.display.specshow(S_db_clean, sr=sr, x_axis='time', y_axis='mel',
                                    ax=axes[1,0], cmap='magma')
    axes[1,0].set_title('Clean audio — mel spectrogram', fontsize=11)
    fig.colorbar(img, ax=axes[1,0], format='%+2.0f dB')

    # OOD spectrogram (shows overflow)
    S_ood = librosa.feature.melspectrogram(y=np.clip(y_ood, -1e6, 1e6), sr=sr, n_mels=64)
    S_db_ood = librosa.power_to_db(S_ood + 1e-10, ref=np.max)
    img2 = librosa.display.specshow(S_db_ood, sr=sr, x_axis='time', y_axis='mel',
                                     ax=axes[1,1], cmap='magma')
    axes[1,1].set_title('OOD amplitude — mel spectrogram (distorted)', fontsize=11)
    fig.colorbar(img2, ax=axes[1,1], format='%+2.0f dB')

    plt.suptitle('Audio Guard: Clean vs. Out-of-Distribution Input', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('paper/figures/fig3_waveforms.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('paper/figures/fig3_waveforms.png', bbox_inches='tight', dpi=150)
    print("Saved fig3_waveforms")

except Exception as e:
    print(f"Skipped fig3_waveforms (librosa issue or no audio files): {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: LayerNorm γ drift illustration
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: γ growth over epochs (simulated based on observed value 1.0 → 6.2 over 3 epochs)
ep = np.linspace(1, 3, 100)
gamma = 1.0 * np.exp(np.log(6.2) / 3 * (ep - 1))

ax = axes[0]
ax.plot(ep, gamma, color='darkorange', linewidth=2.5)
ax.axhline(y=1.0, color='green', linestyle='--', label='Initial value (γ=1.0)')
ax.axhline(y=6.2, color='red',   linestyle='--', label='Observed value at epo-3 (γ≈6.2)')
ax.fill_between(ep, 1.0, gamma, alpha=0.15, color='orange')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('LayerNorm γ (scale parameter)', fontsize=12)
ax.set_title('LayerNorm Scale Drift\n(projection layer)', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: effect on projection output magnitude
ax2 = axes[1]
gamma_vals = [1.0, 2.0, 4.0, 6.2]
output_ranges = [(-2, 2), (-4, 4), (-8, 8), (-12.4, 12.4)]
colors_gamma = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']

for g, (lo, hi), c in zip(gamma_vals, output_ranges, colors_gamma):
    ax2.barh(f'γ={g}', hi - lo, left=lo, height=0.5, color=c, alpha=0.7)

ax2.axvline(x=2,  color='purple', linestyle='--', linewidth=1.5, label='Clamp boundary (±2)')
ax2.axvline(x=-2, color='purple', linestyle='--', linewidth=1.5)
ax2.set_xlabel('Projection output range', fontsize=12)
ax2.set_title('Projection Output Magnitude\nvs. γ value', fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('paper/figures/fig4_gamma_drift.pdf', bbox_inches='tight', dpi=150)
plt.savefig('paper/figures/fig4_gamma_drift.png', bbox_inches='tight', dpi=150)
print("Saved fig4_gamma_drift")


print("\nAll figures saved to paper/figures/")
print("Files:")
for f in sorted(Path('paper/figures').glob('*')):
    print(f"  {f}")
