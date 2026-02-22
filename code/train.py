import os
import torch
from tqdm.auto import tqdm


def train(model, get_batch, cfg, checkpoint_path=None, desc="training"):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    eval_steps = []
    train_losses = []
    val_losses = []

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    model.train()
    pbar = tqdm(range(cfg.max_iters), desc=desc)
    for it in pbar:
        if it % cfg.eval_interval == 0:
            losses = estimate_loss()
            eval_steps.append(it)
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            pbar.set_postfix(
                train=f"{losses['train']:.4f}", val=f"{losses['val']:.4f}"
            )

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final = estimate_loss()
    eval_steps.append(cfg.max_iters)
    train_losses.append(final["train"])
    val_losses.append(final["val"])

    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    return {
        "eval_steps": eval_steps,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_train_loss": final["train"],
        "final_val_loss": final["val"],
    }
