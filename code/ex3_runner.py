import argparse
import sys
import os
import datetime

sys.path.insert(0, os.path.dirname(__file__))

from data import load_captions, build_corpus, make_get_batch
from model import get_configs, NanoGPT
from train import train
from generate import generate
from experiment import ExperimentLogger


def main():
    parser = argparse.ArgumentParser(description="Exercise 3: case normalization experiment")
    parser.add_argument("--case", type=str, default="original",
                        choices=["lower", "upper", "original"])
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    case_transform = None if args.case == "original" else args.case
    if args.label is None:
        args.label = f"case={args.case}"

    logger = ExperimentLogger("outputs/logs/ex3_experiments.json")

    print(f"=== Ex3: {args.label} ===")

    captions = load_captions()
    corpus = build_corpus(captions, sep="\n<ENDC>\n", case_transform=case_transform)
    print(f"Vocab size: {corpus['vocab_size']}, "
          f"Train tokens: {corpus['train_data'].numel()}, "
          f"Val tokens: {corpus['val_data'].numel()}")

    TrainConfig, ModelConfig = get_configs()
    cfg = TrainConfig()
    cfg.max_iters = args.max_iters
    mcfg = ModelConfig(vocab_size=corpus["vocab_size"], block_size=cfg.block_size)

    model = NanoGPT(mcfg).to(cfg.device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    get_batch = make_get_batch(corpus["train_data"], corpus["val_data"], cfg)
    ckpt_name = args.label.replace(" ", "_").replace("/", "_")
    checkpoint_path = f"outputs/checkpoints/ex3_{ckpt_name}.pt"

    results = train(model, get_batch, cfg,
                    checkpoint_path=checkpoint_path, desc=f"Ex3: {args.label}")

    prompt = "h&e stained section showing" if args.case == "lower" else "H&E stained section showing"
    if args.case == "upper":
        prompt = "H&E STAINED SECTION SHOWING"

    samples = []
    for i in range(args.num_samples):
        text = generate(model, corpus["encode"], corpus["decode"], cfg,
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, top_k=args.top_k)
        samples.append(text)
        print(f"\n--- Sample {i + 1} ---")
        print(text)

    entry = {
        "label": args.label,
        "timestamp": datetime.datetime.now().isoformat(),
        "case_transform": args.case,
        "vocab_size": corpus["vocab_size"],
        "train_tokens": int(corpus["train_data"].numel()),
        "val_tokens": int(corpus["val_data"].numel()),
        "final_train_loss": results["final_train_loss"],
        "final_val_loss": results["final_val_loss"],
        "eval_steps": results["eval_steps"],
        "train_losses": results["train_losses"],
        "val_losses": results["val_losses"],
        "generated_samples": samples,
        "generation_params": {
            "prompt": prompt,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
        },
        "checkpoint": checkpoint_path,
    }

    count = logger.append(entry)
    print(f"\nLogged: '{args.label}' -> train={results['final_train_loss']:.4f}, "
          f"val={results['final_val_loss']:.4f}")
    print(f"Total ex3 experiments: {count}")


if __name__ == "__main__":
    main()
