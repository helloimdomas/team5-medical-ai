import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment import ExperimentLogger


def report_ex1():
    logger = ExperimentLogger("outputs/logs/ex1_experiments.json")
    log = logger.load()
    if not log:
        print("No ex1 experiments found. Run: python code/ex1_runner.py --sep endc")
        return

    md = []
    md.append("#### Exercise 1\n")
    md.append("When we created the text corpus, we used the `<ENDC>` separator to mark the end of a caption. "
              "We trained models with different separators to observe the effect.\n")

    for exp in log:
        sep_display = repr(exp.get("sep", ""))
        md.append(f'Output with separator `{sep_display}` (`{exp["label"]}`, '
                  f'final train={exp["final_train_loss"]:.4f}, val={exp["final_val_loss"]:.4f}):')
        md.append("```")
        for sample in exp.get("generated_samples", []):
            md.append(sample)
        md.append("```\n")

    print("\n".join(md))


def report_ex2():
    logger = ExperimentLogger("outputs/logs/ex2_experiments.json")
    log = logger.load()
    if not log:
        print("No ex2 experiments found. Run: python code/ex2_runner.py")
        return

    print("=" * 80)
    print(f"{'Label':<25} {'LR':<10} {'BS':<6} {'Eval Int':<10} {'Train Loss':<12} {'Val Loss':<12}")
    print("-" * 80)
    for exp in log:
        hp = exp["hyperparameters"]
        print(f"{exp['label']:<25} {hp['lr']:<10} {hp['batch_size']:<6} "
              f"{hp['eval_interval']:<10} {exp['final_train_loss']:<12.4f} {exp['final_val_loss']:<12.4f}")
    print("=" * 80)

    def categorize(exp):
        hp = exp["hyperparameters"]
        cats = []
        if hp["lr"] != 1e-3 or exp["label"] == "baseline":
            cats.append("learning_rate")
        if hp["batch_size"] != 12 or exp["label"] == "baseline":
            cats.append("batch_size")
        if hp["eval_interval"] != 200 or exp["label"] == "baseline":
            cats.append("eval_interval")
        if not cats:
            cats = ["other"]
        return cats

    groups = {"learning_rate": [], "batch_size": [], "eval_interval": [], "other": []}
    for exp in log:
        for cat in categorize(exp):
            if cat in groups:
                groups[cat].append(exp)
    groups = {k: v for k, v in groups.items() if v}

    group_titles = {
        "learning_rate": "Effect of Learning Rate",
        "batch_size": "Effect of Batch Size",
        "eval_interval": "Effect of Eval Interval",
        "other": "Other Experiments",
    }

    n_groups = len(groups)
    if n_groups > 0:
        fig, axes = plt.subplots(1, n_groups, figsize=(7 * n_groups, 5), squeeze=False)
        colors = plt.cm.tab10.colors

        for ax_idx, (group_name, exps) in enumerate(groups.items()):
            ax = axes[0][ax_idx]
            for i, exp in enumerate(exps):
                c = colors[i % len(colors)]
                ax.plot(exp["eval_steps"], exp["train_losses"],
                        color=c, linestyle="-", label=f'{exp["label"]} (train)')
                ax.plot(exp["eval_steps"], exp["val_losses"],
                        color=c, linestyle="--", label=f'{exp["label"]} (val)')
            ax.set_xlabel("Training Iteration")
            ax.set_ylabel("Loss")
            ax.set_title(group_titles.get(group_name, group_name))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = "outputs/plots/ex2_loss_plots.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {plot_path}")

    md = []
    md.append("#### Exercise 2\n")
    md.append("**Experiment setup:** We trained the model with different hyperparameters while keeping "
              "all other settings at their defaults (CPU config: `lr=1e-3`, `batch_size=12`, `eval_interval=200`).\n")
    md.append("**Summary of results:**\n")
    md.append("| Experiment | LR | Batch Size | Eval Interval | Final Train Loss | Final Val Loss |")
    md.append("|---|---|---|---|---|---|")
    for exp in log:
        hp = exp["hyperparameters"]
        md.append(f"| {exp['label']} | {hp['lr']} | {hp['batch_size']} | {hp['eval_interval']} "
                  f"| {exp['final_train_loss']:.4f} | {exp['final_val_loss']:.4f} |")
    md.append("")
    md.append("**Loss curves:**\n")
    md.append("![Ex2 Loss Plots](../outputs/plots/ex2_loss_plots.png)\n")

    for group_name, exps in groups.items():
        title = group_titles.get(group_name, group_name)
        labels = ", ".join(f"`{e['label']}`" for e in exps)
        md.append(f"**{title}** ({labels}):\n")
        md.append("- *TODO: Explain the observed changes here*\n")

    md.append("**Underfitting indicators:** *TODO*\n")
    md.append("**Overfitting indicators:** *TODO*\n")
    md.append("**Unstable training indicators:** *TODO*\n")

    print("\n" + "=" * 80)
    print("MARKDOWN FOR REPORT:")
    print("=" * 80)
    print("\n".join(md))


def report_ex3():
    logger = ExperimentLogger("outputs/logs/ex3_experiments.json")
    log = logger.load()
    if not log:
        print("No ex3 experiments found. Run: python code/ex3_runner.py --case lower")
        return

    md = []
    md.append("#### Exercise 3\n")
    md.append("We reduced the number of tokens by converting all letters to lowercase/uppercase "
              "and compared against the original.\n")
    md.append("| Experiment | Case | Vocab Size | Train Tokens | Final Train Loss | Final Val Loss |")
    md.append("|---|---|---|---|---|---|")
    for exp in log:
        md.append(f"| {exp['label']} | {exp.get('case_transform', 'original')} | "
                  f"{exp['vocab_size']} | {exp['train_tokens']} | "
                  f"{exp['final_train_loss']:.4f} | {exp['final_val_loss']:.4f} |")
    md.append("")

    for exp in log:
        md.append(f'Generated output (`{exp["label"]}`):\n')
        md.append("```")
        for sample in exp.get("generated_samples", []):
            md.append(sample)
        md.append("```\n")

    print("\n".join(md))


def report_ex5():
    logger = ExperimentLogger("outputs/logs/ex5_experiments.json")
    log = logger.load()
    if not log:
        print("No ex5 experiments found. Run: python code/ex5_runner.py --temperature 0.5 --top_k 10")
        return

    md = []
    md.append("#### Exercise 5\n")
    md.append("We generated captions with different temperature and top_k values.\n")

    for exp in log:
        md.append(f'**Configuration: `temperature={exp["temperature"]}`, `top_k={exp["top_k"]}`** '
                  f'(`{exp["label"]}`):\n')
        for i, sample in enumerate(exp.get("generated_samples", []), 1):
            md.append(f"Sample {i}:")
            md.append("```")
            md.append(sample)
            md.append("```\n")
        md.append("- **Fluency and structure:** *TODO*")
        md.append("- **Repetition or degeneration:** *TODO*")
        md.append("- **Factual plausibility:** *TODO*\n")

    md.append("**Optimal configuration:** *TODO: justify your choice*\n")

    print("\n".join(md))


def list_all():
    for ex, path in [("Ex1", "outputs/logs/ex1_experiments.json"),
                     ("Ex2", "outputs/logs/ex2_experiments.json"),
                     ("Ex3", "outputs/logs/ex3_experiments.json"),
                     ("Ex5", "outputs/logs/ex5_experiments.json")]:
        logger = ExperimentLogger(path)
        log = logger.load()
        print(f"\n{ex} ({len(log)} experiments):")
        logger.list()


def main():
    parser = argparse.ArgumentParser(description="Generate exercise reports from logged experiments")
    parser.add_argument("--exercise", type=int, default=None, choices=[1, 2, 3, 5])
    parser.add_argument("--list", action="store_true", help="List all experiments across all exercises")
    parser.add_argument("--delete", type=int, default=None, metavar="INDEX",
                        help="Delete experiment at index (use with --exercise)")
    parser.add_argument("--clear", action="store_true", help="Clear all experiments (use with --exercise)")
    args = parser.parse_args()

    if args.list:
        list_all()
        return

    if args.exercise is None:
        parser.print_help()
        return

    log_paths = {
        1: "outputs/logs/ex1_experiments.json",
        2: "outputs/logs/ex2_experiments.json",
        3: "outputs/logs/ex3_experiments.json",
        5: "outputs/logs/ex5_experiments.json",
    }

    if args.clear:
        logger = ExperimentLogger(log_paths[args.exercise])
        logger.clear()
        print(f"Cleared all ex{args.exercise} experiments.")
        return

    if args.delete is not None:
        logger = ExperimentLogger(log_paths[args.exercise])
        removed = logger.delete(args.delete)
        print(f"Deleted: '{removed.get('label', 'unnamed')}'")
        logger.list()
        return

    {1: report_ex1, 2: report_ex2, 3: report_ex3, 5: report_ex5}[args.exercise]()


if __name__ == "__main__":
    main()
