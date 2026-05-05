"""
Generate word cloud figure comparing model outputs vs reference captions.
Run from project root: /home/yandex/APDL2526a/sapirtalmi/mellow_env/bin/python paper/make_wordcloud.py
Output: paper/figures/fig9_wordcloud.png
"""

import re
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

os.makedirs('paper/figures', exist_ok=True)

# ── Stop words ────────────────────────────────────────────────────────────────
STOP = {
    'a','an','the','and','or','but','in','on','of','to','is','are','it','its',
    'this','that','with','for','as','by','at','be','was','were','has','have',
    'from','which','while','the','both','two','also','more','most','very',
    'some','such','than','then','not','no','so','do','can','may','will',
    'would','could','should','i','we','you','he','she','they','their','our',
    'my','your','his','her','one','all','each','any','been','being','had',
    'audio','sound','clip','first','second','overall','sense','like','similar',
    'likely','probably','definitely','plausible','false','true','yes','b','c',
    'present','characterized','features','featuring','dominant','frequency',
    'create','creates','creating','make','makes','making','add','adds','adding',
    'produce','produces','producing','provide','provides','use','used','used',
}

def clean_tokens(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return [w for w in text.split() if w not in STOP and len(w) > 2]

# ── Collect model outputs from inference log ──────────────────────────────────
print("Reading inference log...")
model_tokens = []
log_path = 'logs/slurm-218958_inference.out'

with open(log_path, 'r') as f:
    for line in f:
        if line.startswith('generated_text'):
            matches = re.findall(r"'([^']*)'", line)
            for m in matches:
                model_tokens.extend(clean_tokens(m))

print(f"  Model tokens: {len(model_tokens)}")

# ── Collect reference captions from val datafile ──────────────────────────────
print("Reading reference captions...")
ref_tokens = []
datafile = 'datafiles/val_clotho.json'

try:
    # File is large — read first 50k chars to get a sample of captions
    with open(datafile, 'r') as f:
        chunk = f.read(200000)

    # Extract answer fields (reference captions)
    answers = re.findall(r'"answer"\s*:\s*"([^"]+)"', chunk)
    for ans in answers[:2000]:
        ref_tokens.extend(clean_tokens(ans))
    print(f"  Reference tokens: {len(ref_tokens)}")
except Exception as e:
    print(f"  Could not read datafile: {e}")
    # Fallback: use typical Clotho caption vocabulary
    ref_tokens = clean_tokens(
        "rain falling water flowing birds chirping wind blowing traffic passing "
        "footsteps walking engine humming door opening children playing crowd talking "
        "thunder rumbling waves crashing leaves rustling machinery running dog barking "
        "music playing keyboard typing hammer striking bell ringing fire crackling "
        "rain drops splashing stream bubbling owl hooting cat meowing train passing "
        "airplane flying car driving bicycle riding bus stopping crowd cheering "
        "rain on roof wind through trees water dripping pipes creek flowing "
        ) * 50

# ── Build frequency dicts ─────────────────────────────────────────────────────
model_freq = Counter(model_tokens)
ref_freq   = Counter(ref_tokens)

# Keep top N words
TOP_N = 80
model_top = dict(model_freq.most_common(TOP_N))
ref_top   = dict(ref_freq.most_common(TOP_N))

# ── Try wordcloud library, fall back to manual bar chart ─────────────────────
try:
    from wordcloud import WordCloud

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Vocabulary Comparison: Model Outputs vs. Reference Captions',
                 fontsize=14, fontweight='bold')

    wc_model = WordCloud(
        width=600, height=400, background_color='white',
        colormap='Blues', max_words=60,
        prefer_horizontal=0.9
    ).generate_from_frequencies(model_top)

    wc_ref = WordCloud(
        width=600, height=400, background_color='white',
        colormap='Greens', max_words=60,
        prefer_horizontal=0.9
    ).generate_from_frequencies(ref_top)

    axes[0].imshow(wc_model, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title('Model outputs\n(overuses DSP jargon)', fontsize=12)

    axes[1].imshow(wc_ref, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title('Reference captions\n(concrete audio events)', fontsize=12)

    plt.tight_layout()
    plt.savefig('paper/figures/fig9_wordcloud.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('paper/figures/fig9_wordcloud.png', bbox_inches='tight', dpi=150)
    print("Saved fig9_wordcloud (wordcloud version)")

except ImportError:
    print("wordcloud not installed — generating bar chart version instead")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Top-30 Words: Model Outputs vs. Reference Captions',
                 fontsize=14, fontweight='bold')

    # Model top 30
    model_30 = model_freq.most_common(30)
    words_m, counts_m = zip(*model_30)
    axes[0].barh(range(len(words_m)), counts_m, color='steelblue', alpha=0.8)
    axes[0].set_yticks(range(len(words_m)))
    axes[0].set_yticklabels(words_m, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Frequency', fontsize=11)
    axes[0].set_title('Model outputs\n(overuses DSP jargon)', fontsize=12)
    axes[0].grid(True, axis='x', alpha=0.3)

    # Reference top 30
    ref_30 = ref_freq.most_common(30)
    words_r, counts_r = zip(*ref_30)
    axes[1].barh(range(len(words_r)), counts_r, color='seagreen', alpha=0.8)
    axes[1].set_yticks(range(len(words_r)))
    axes[1].set_yticklabels(words_r, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Frequency', fontsize=11)
    axes[1].set_title('Reference captions\n(concrete audio events)', fontsize=12)
    axes[1].grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('paper/figures/fig9_wordcloud.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('paper/figures/fig9_wordcloud.png', bbox_inches='tight', dpi=150)
    print("Saved fig9_wordcloud (bar chart version)")

# ── Print top words for inspection ───────────────────────────────────────────
print("\nTop 20 model words:", [w for w,_ in model_freq.most_common(20)])
print("Top 20 reference words:", [w for w,_ in ref_freq.most_common(20)])
