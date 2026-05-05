"""
Evaluate MC/QA accuracy on val_mc.json.
For each task type:
  - Clotho-MCQ: correct if predicted first letter matches answer first letter (A/B/C/D)
  - CLE:        correct if predicted first letter matches answer first letter (a/b/c/d)
  - ClothoAQA:  correct if predicted first word matches answer (yes/no)

Usage:
    python scripts/eval_mc_accuracy.py --checkpoint path/to/model.ckpt
"""

import os, sys, json, re, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('HF_HOME', '/home/yandex/APDL2526a/sapirtalmi/hf_cache')
os.environ.setdefault('TRANSFORMERS_CACHE', '/home/yandex/APDL2526a/sapirtalmi/hf_cache')

import torch
import torchaudio
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from models.mellow import Mellow
from models.generate import generate_greedy_batch

# ── Config ────────────────────────────────────────────────────────────────────
TOKENIZER   = 'HuggingFaceTB/SmolLM2-135M'
HTSAT_PATH  = '/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/htsat'
DATA_ROOT   = '/home/yandex/APDL2526a/idantarshish/mellow_apdl_project/data'
VAL_MC      = '/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/datafiles/val_mc.json'
SR          = 32000
MAX_SAMPLES = SR * 10
IP_TEXT_LEN = 129

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--limit', type=int, default=None, help='Limit samples per subtype for speed')
args = parser.parse_args()

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = Mellow(
    audioenc_name='HTSAT', d_in=768, text_decoder=TOKENIZER,
    prefix_length=40, freeze_text_decoder_weights=False, d_out=576,
    use_pretrained_audioencoder=True, freeze_audio_encoder_weights=True,
    pretrained_audioencoder_path=HTSAT_PATH,
)
ckpt = torch.load(args.checkpoint, map_location=device)
state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
model.load_state_dict(state, strict=True)
model.eval().to(device)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
tokenizer.add_special_tokens({'pad_token': '!'})
print(f"Checkpoint: {args.checkpoint}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_audio(path):
    wav, sr = torchaudio.load(path, channels_first=True)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.shape[1] >= MAX_SAMPLES:
        wav = wav[:, :MAX_SAMPLES]
    else:
        pad = torch.zeros(1, MAX_SAMPLES)
        pad[:, :wav.shape[1]] = wav
        wav = pad
    return wav  # [1, T]

def tokenize(text):
    return tokenizer.encode_plus(
        text + ' <|endoftext|>', add_special_tokens=True, truncation=True,
        max_length=IP_TEXT_LEN, pad_to_max_length=True, return_tensors='pt'
    )

def extract_answer_letter(text, subtype):
    text = text.strip().lower()
    if subtype == 'ClothoAQA.json':
        if text.startswith('yes'): return 'yes'
        if text.startswith('no'):  return 'no'
        return text.split()[0] if text else ''
    # MCQ / CLE: look for first a/b/c/d letter
    m = re.match(r'^([a-d])\b', text)
    if m: return m.group(1)
    return text[0] if text else ''

def extract_correct_letter(answer, subtype):
    answer = answer.strip().lower()
    if subtype == 'ClothoAQA.json':
        if answer.startswith('yes'): return 'yes'
        if answer.startswith('no'):  return 'no'
        return answer.split()[0] if answer else ''
    m = re.match(r'^([a-d])\b', answer)
    if m: return m.group(1)
    return answer[0] if answer else ''

# ── Load data ─────────────────────────────────────────────────────────────────
with open(VAL_MC) as f:
    data = json.load(f)

random.seed(42)
by_subtype = {}
for x in data:
    s = x['subtype']
    by_subtype.setdefault(s, []).append(x)

if args.limit:
    for s in by_subtype:
        by_subtype[s] = random.sample(by_subtype[s], min(args.limit, len(by_subtype[s])))

# ── Eval loop ─────────────────────────────────────────────────────────────────
results = {}

for subtype, items in by_subtype.items():
    correct = 0
    total   = 0
    errors  = 0

    for item in tqdm(items, desc=subtype):
        fp1 = os.path.join(DATA_ROOT, item['filepath1'].replace('\\', '/'))
        fp2 = item.get('filepath2', '')
        fp2 = os.path.join(DATA_ROOT, fp2.replace('\\', '/')) if fp2 else fp1

        try:
            a1 = load_audio(fp1).to(device)
            a2 = load_audio(fp2).to(device)
        except Exception:
            errors += 1
            continue

        inp_text = item['input'].lower()
        tok = tokenize(inp_text)

        input_dict = {
            'audio1': a1,
            'audio2': a2,
            'input':  {k: v.to(device) for k, v in tok.items()},
            'answer': {k: v.to(device) for k, v in tok.items()},
        }

        with torch.no_grad():
            prefix, _, _ = model.generate_prefix_inference(input_dict)
            generated = generate_greedy_batch(
                model, tokenizer, embed=prefix, strip_option_prefix=False
            )

        pred = generated[0] if generated else ''
        pred_letter = extract_answer_letter(pred, subtype)
        true_letter = extract_correct_letter(item['answer'], subtype)

        if pred_letter == true_letter:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    random_baseline = 0.25 if subtype != 'ClothoAQA.json' else 0.5
    results[subtype] = {
        'accuracy': acc, 'correct': correct, 'total': total,
        'errors': errors, 'random_baseline': random_baseline
    }
    print(f"\n{subtype}: {acc:.3f} ({correct}/{total}) vs random={random_baseline:.2f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== MC Accuracy Summary ===")
print(f"Checkpoint: {args.checkpoint}")
for s, r in results.items():
    gain = r['accuracy'] - r['random_baseline']
    print(f"  {s:25s}  acc={r['accuracy']:.3f}  random={r['random_baseline']:.2f}  gain={gain:+.3f}  n={r['total']}")

out_path = args.checkpoint.replace('.ckpt', '_mc_accuracy.json')
with open(out_path, 'w') as f:
    json.dump({'checkpoint': args.checkpoint, 'results': results}, f, indent=2)
print(f"\nSaved: {out_path}")
