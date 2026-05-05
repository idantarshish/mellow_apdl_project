"""
Standalone debiased eval — bypasses trainer.py entirely.

Runs generate_greedy_batch with:
  - block_option_tokens=True  (masks a/b/c on first generated token)
  - strip_option_prefix=True  (strips leading "a) / b) / c) " from output)

Usage:
  python scripts/eval_debiased.py --checkpoint PATH [--debug N] [--no-block] [--no-strip]
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.model import get_model_class
from models.generate import generate_greedy_batch
from data.audiotext_eval_dataset import AudioTextEvalDataset, collate_fn
from metrics.get_metrics import Metric

# ── fixed config (mirrors eval_1gpu.yaml) ────────────────────────────────────
CFG = {
    "data_path":   "/home/yandex/APDL2526a/idantarshish/mellow_apdl_project/data",
    "datafile":    "/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/datafiles/val_clotho.json",
    "sampling_rate": 32000,
    "segment_seconds": 10,
    "tokenizer_type": "HuggingFaceTB/SmolLM2-135M",
    "op_text_len": 250,
    "ip_text_len": 129,
    "batch_size":  4,
    "num_workers": 0,
    "hf_cache":    "/home/yandex/APDL2526a/sapirtalmi/hf_cache",
    "model": {
        "audioenc_name": "HTSAT",
        "out_emb": 768,
        "d_proj": 576,
        "prefix_length": 40,
        "freeze_gpt_weights": False,
        "use_pretrained_audioencoder": True,
        "freeze_audio_encoder_weights": True,
        "pretrained_audioencoder_path": "/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/htsat",
        "text_decoder": "HuggingFaceTB/SmolLM2-135M",
    },
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument("--debug", type=int, default=None, help="Limit to N samples")
    p.add_argument("--no-block", action="store_true", help="Disable first-token masking")
    p.add_argument("--no-strip", action="store_true", help="Disable a)/b)/c) prefix stripping")
    return p.parse_args()

def main():
    args = parse_args()
    os.environ["HF_HOME"] = CFG["hf_cache"]
    os.environ["TRANSFORMERS_CACHE"] = CFG["hf_cache"]

    block = not args.no_block
    strip = not args.no_strip
    print(f"block_option_tokens={block}  strip_option_prefix={strip}")
    print(f"checkpoint: {args.checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ── pre-download HF models ─────────────────────────────────────────────
    AutoModelForCausalLM.from_pretrained(CFG["model"]["text_decoder"])
    AutoTokenizer.from_pretrained(CFG["tokenizer_type"])

    # ── build model ────────────────────────────────────────────────────────
    Model = get_model_class("Mellow")
    model = Model(
        audioenc_name=CFG["model"]["audioenc_name"],
        d_in=CFG["model"]["out_emb"],
        text_decoder=CFG["model"]["text_decoder"],
        prefix_length=CFG["model"]["prefix_length"],
        freeze_text_decoder_weights=CFG["model"]["freeze_gpt_weights"],
        d_out=CFG["model"]["d_proj"],
        use_pretrained_audioencoder=CFG["model"]["use_pretrained_audioencoder"],
        freeze_audio_encoder_weights=CFG["model"]["freeze_audio_encoder_weights"],
        pretrained_audioencoder_path=CFG["model"]["pretrained_audioencoder_path"],
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt, strict=True)
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    # ── dataset ────────────────────────────────────────────────────────────
    dataset = AudioTextEvalDataset(
        data_path=CFG["data_path"],
        datafiles=[CFG["datafile"]],
        sampling_rate=CFG["sampling_rate"],
        max_clip_len=CFG["segment_seconds"],
        tokenizer_type=CFG["tokenizer_type"],
        ip_text_len=CFG["ip_text_len"],
        op_text_len=CFG["op_text_len"],
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=CFG["batch_size"], num_workers=CFG["num_workers"],
        collate_fn=collate_fn, shuffle=False, drop_last=False,
    )
    print(f"Dataset: {len(dataset)} examples")

    # ── output dir ─────────────────────────────────────────────────────────
    tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = ("_block" if block else "") + ("_strip" if strip else "")
    out_dir = os.path.join(
        "/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/outputs",
        f"{tag}_debiased{suffix}",
        "val_clotho_outputs",
    )
    os.makedirs(out_dir, exist_ok=True)

    # ── inference loop ─────────────────────────────────────────────────────
    generations, answers, filepaths, inputs = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating"):
            input_dict = {
                "audio1":  batch["waveform1"].to(device),
                "audio2":  batch["waveform2"].to(device),
                "input":   {k: v.to(device) for k, v in batch["input"].items()},
                "answer":  {k: v.to(device) for k, v in batch["answer"].items()},
            }

            prefix, _, _ = model.generate_prefix_inference(input_dict)
            gen = generate_greedy_batch(
                model, dataset.tokenizer, embed=prefix,
                block_option_tokens=block,
                strip_option_prefix=strip,
            )

            generations += gen
            answers    += batch["answer_text"]
            inputs     += batch["input_text"]
            filepaths  += batch["file_path1"]

            if args.debug is not None and len(generations) >= args.debug:
                break

    # ── save per-sample JSON ───────────────────────────────────────────────
    for i, (fp, inp, gen, ans) in enumerate(zip(filepaths, inputs, generations, answers)):
        sample_name = os.path.splitext(os.path.basename(fp))[0]
        out_path = os.path.join(out_dir, f"{i:04d}_{sample_name}.json")
        with open(out_path, "w") as f:
            json.dump({"filepath": fp, "input": inp, "generated": gen, "answer": ans}, f, indent=2)

    print(f"Saved {len(generations)} samples to {out_dir}")

    # ── metrics ────────────────────────────────────────────────────────────
    metric = Metric(CFG["datafile"], CFG["sampling_rate"])
    metric.get_metrics(generations, answers, filepaths)
    print("\n=== Metrics ===")
    for k, v in metric.metrics.items():
        print(f"  {k}: {v['score']:.4f}" if isinstance(v, dict) else f"  {k}: {v}")

if __name__ == "__main__":
    main()
