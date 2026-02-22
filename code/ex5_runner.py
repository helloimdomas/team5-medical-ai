import argparse
import sys
import os
import datetime

sys.path.insert(0, os.path.dirname(__file__))

import torch
from data import load_captions, build_corpus
from model import get_configs, NanoGPT
from generate import generate
from experiment import ExperimentLogger


def main():
    parser = argparse.ArgumentParser(description="Exercise 5: sampling hyperparameters")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top_k", type=int, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint. If not set, uses latest ex1 endc checkpoint.")
    parser.add_argument("--label", type=str, default=None)
    args = parser.parse_args()

    if args.label is None:
        args.label = f"temp={args.temperature}, top_k={args.top_k}"

    logger = ExperimentLogger("outputs/logs/ex5_experiments.json")

    print(f"=== Ex5: {args.label} ===")

    captions = load_captions()
    corpus = build_corpus(captions, sep="\n<ENDC>\n")

    TrainConfig, ModelConfig = get_configs()
    cfg = TrainConfig()
    mcfg = ModelConfig(vocab_size=corpus["vocab_size"], block_size=cfg.block_size)

    model = NanoGPT(mcfg).to(cfg.device)

    checkpoint = args.checkpoint
    if checkpoint is None:
        candidates = [
            "outputs/checkpoints/ex1_sep=endc.pt",
            "outputs/checkpoints/baseline.pt",
        ]
        for c in candidates:
            if os.path.exists(c):
                checkpoint = c
                break
        if checkpoint is None:
            print("No checkpoint found. Train a model first (e.g. python code/ex1_runner.py --sep endc)")
            sys.exit(1)

    print(f"Loading checkpoint: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=cfg.device, weights_only=True))

    samples = []
    for i in range(args.num_samples):
        text = generate(model, corpus["encode"], corpus["decode"], cfg,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, top_k=args.top_k)
        samples.append(text)
        print(f"\n--- Sample {i + 1} ---")
        print(text)

    entry = {
        "label": args.label,
        "timestamp": datetime.datetime.now().isoformat(),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "checkpoint": checkpoint,
        "generated_samples": samples,
    }

    count = logger.append(entry)
    print(f"\nLogged: '{args.label}' ({args.num_samples} samples)")
    print(f"Total ex5 experiments: {count}")


if __name__ == "__main__":
    main()
