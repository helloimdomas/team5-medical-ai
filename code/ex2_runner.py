import argparse
import sys
import os
import datetime

sys.path.insert(0, os.path.dirname(__file__))

from data import load_captions, build_corpus, make_get_batch
from model import get_configs, NanoGPT
from train import train
from experiment import ExperimentLogger


def main():
    parser = argparse.ArgumentParser(description="Exercise 2: hyperparameter experiment")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--label", type=str, default=None)
    args = parser.parse_args()

    logger = ExperimentLogger("outputs/logs/ex2_experiments.json")

    captions = load_captions()
    corpus = build_corpus(captions, sep="\n<ENDC>\n")

    TrainConfig, ModelConfig = get_configs()
    cfg = TrainConfig()

    if args.lr is not None:
        cfg.lr = args.lr
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.eval_interval is not None:
        cfg.eval_interval = args.eval_interval
    if args.max_iters is not None:
        cfg.max_iters = args.max_iters

    if args.label is None:
        parts = []
        if args.lr is not None:
            parts.append(f"lr={args.lr}")
        if args.batch_size is not None:
            parts.append(f"bs={args.batch_size}")
        if args.eval_interval is not None:
            parts.append(f"eval_int={args.eval_interval}")
        args.label = ", ".join(parts) if parts else "baseline"

    print(f"=== Ex2: {args.label} ===")
    print(f"  lr={cfg.lr}, batch_size={cfg.batch_size}, "
          f"eval_interval={cfg.eval_interval}, max_iters={cfg.max_iters}")

    mcfg = ModelConfig(vocab_size=corpus["vocab_size"], block_size=cfg.block_size)
    model = NanoGPT(mcfg).to(cfg.device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    get_batch = make_get_batch(corpus["train_data"], corpus["val_data"], cfg)
    ckpt_name = args.label.replace(" ", "_").replace(",", "").replace("/", "_")
    checkpoint_path = f"outputs/checkpoints/ex2_{ckpt_name}.pt"

    results = train(model, get_batch, cfg,
                    checkpoint_path=checkpoint_path, desc=f"Ex2: {args.label}")

    entry = {
        "label": args.label,
        "timestamp": datetime.datetime.now().isoformat(),
        "hyperparameters": {
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "eval_interval": cfg.eval_interval,
            "max_iters": cfg.max_iters,
            "eval_iters": cfg.eval_iters,
        },
        "eval_steps": results["eval_steps"],
        "train_losses": results["train_losses"],
        "val_losses": results["val_losses"],
        "final_train_loss": results["final_train_loss"],
        "final_val_loss": results["final_val_loss"],
        "checkpoint": checkpoint_path,
    }

    count = logger.append(entry)
    print(f"\nLogged: '{args.label}' -> train={results['final_train_loss']:.4f}, "
          f"val={results['final_val_loss']:.4f}")
    print(f"Total ex2 experiments: {count}")


if __name__ == "__main__":
    main()
