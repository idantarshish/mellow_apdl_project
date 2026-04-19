import os
import sys
import json
import torch
import logging
from tqdm import tqdm
from datetime import datetime
from collections import Counter

import distributed
from utils.launch_utils import parse_args, multiprocessing_init, parse_mode
from training.log import configure_logging
from training.trainer import Trainer, TrainerMode
from models.generate import generate_greedy_batch
from metrics.get_metrics import Metric
from utils.utils import LazyConversionDict


def majority_vote(candidates):
    """Returns the most frequent string in a list of strings."""
    if not candidates:
        return ""
    # Strip whitespace and normalize to improve matching consistency
    counts = Counter([c.strip() for c in candidates])
    return counts.most_common(1)[0][0]


def run_self_consistency_eval(trainer: Trainer, args):
    trainer.logger.info("Starting Self-Consistency Evaluation")

    # Parameters for SC
    n_iterations = getattr(args, 'sc_n', 5)
    sc_temp = getattr(args, 'sc_temperature', 0.7)
    sc_top_p = getattr(args, 'sc_top_p', 0.9)

    trainer.config["model"]["decoder"]["prefix_dim"] = trainer.config["model"]["encoder"]["d_proj"]

    # Initialize Model
    model = trainer.get_model().to(trainer.device)
    checkpoint = torch.load(trainer.config["checkpoint_path"], map_location=trainer.device)
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint, strict=True)
    model.eval()

    tasks = trainer.config["data"]["datafiles"]
    for task in tasks:
        trainer.logger.info(f"Evaluating task {task} with SC (n={n_iterations})")
        trainer.config["data"]["datafiles"] = [task]

        metric = Metric(task, trainer.config["data"]["sampling_rate"])
        dataset, _, data_loader = trainer.get_data("datafiles")

        all_voted_generations = []
        all_answers = []
        all_filepaths = []
        all_inputs = []

        with torch.no_grad():
            for batch_data_dict in tqdm(data_loader, desc=f"SC Eval {task}"):
                # Prepare input dict similar to trainer.py
                input_dict = {
                    "audio1": batch_data_dict['waveform1'],
                    "audio2": batch_data_dict['waveform2'],
                    "input": batch_data_dict['input'],
                    "answer": batch_data_dict['answer'],
                }
                input_dict = LazyConversionDict(input_dict, lambda x: x.to(trainer.device))

                # Get prefix embedding from audio/input
                prefix, _, _ = model.generate_prefix_inference(input_dict)

                # Generate N candidates for the whole batch
                batch_candidates = []  # Will be a list of lists: [n_iterations][batch_size]
                for _ in range(n_iterations):
                    # Use provided temperature and top_p for sampling
                    gen_texts = generate_greedy_batch(
                        model,
                        data_loader.dataset.tokenizer,
                        embed=prefix,
                        temperature=sc_temp,
                        top_p=sc_top_p
                    )
                    batch_candidates.append(gen_texts)

                # Transpose and Vote
                # batch_candidates is [N, BatchSize], we want to vote per item in BatchSize
                batch_size = len(batch_candidates[0])
                for i in range(batch_size):
                    item_candidates = [batch_candidates[j][i] for j in range(n_iterations)]
                    voted_text = majority_vote(item_candidates)
                    all_voted_generations.append(voted_text)

                all_answers += batch_data_dict['answer_text']
                all_inputs += batch_data_dict['input_text']
                all_filepaths += batch_data_dict['file_path1']

        # Save results (Rank 0 only)
        if trainer.distributed.rank() == 0:
            taskname = task.split(os.path.sep)[-1].split(".json")[0]
            samples_dir = os.path.join(trainer.config["save_dir"], f"{taskname}_sc_outputs")
            os.makedirs(samples_dir, exist_ok=True)

            for i, (fp, inp, gen, ans) in enumerate(zip(all_filepaths, all_inputs, all_voted_generations, all_answers)):
                sample_name = os.path.splitext(os.path.basename(fp))[0]
                sample_path = os.path.join(samples_dir, f"{i:04d}_{sample_name}.json")
                with open(sample_path, "w") as f:
                    json.dump({"filepath": fp, "input": inp, "generated": gen, "answer": ans}, f, indent=2)

            # Calculate and log metrics
            metric.get_metrics(all_voted_generations, all_answers, all_filepaths)
            trainer.logger.info(f"Task {task} SC results:")
            for key in metric.metrics.keys():
                trainer.logger.info("%s: %f", key, metric.metrics[key]["score"])


def main():
    args, conf = parse_args()
    args = parse_mode(args)

    # Default values for Self-Consistency
    args.sc_n = getattr(args, 'sc_n', 5)
    args.sc_temperature = getattr(args, 'sc_temperature', 0.7)
    args.sc_top_p = getattr(args, 'sc_top_p', 0.9)

    args.job_id = f"SC_EVAL_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    args.save_dir = os.path.join(args.save_dir, args.job_id)
    os.makedirs(args.save_dir, exist_ok=True)

    configure_logging()
    multiprocessing_init()

    if args.distributed_backend is None:
        distributed_ctx = distributed.get_local_context()
    else:
        import distributed.torch as impl
        distributed_ctx = impl.TorchDistributedContext(args.distributed_backend)

    with distributed_ctx:
        if distributed_ctx.rank() > 0:
            logging.getLogger().setLevel(logging.ERROR)

        with Trainer(vars(args), distributed_ctx=distributed_ctx) as trainer:
            try:
                run_self_consistency_eval(trainer, args)
            except Exception:
                logging.error("Error during SC evaluation", exc_info=sys.exc_info())
                if args.reraise_exceptions:
                    raise
                return -1
    return 0


if __name__ == "__main__":
    exit(main())