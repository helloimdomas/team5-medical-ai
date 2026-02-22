import os
import json
import torch
from datasets import load_dataset, concatenate_datasets


def load_captions(cache_path="assignments/captions_cache.json"):
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            captions = json.load(f)
    else:
        ds_dict = load_dataset("MartiHan/Open-MELON-VL-2.5K")
        ds_all = concatenate_datasets(list(ds_dict.values()))
        captions = [str(x) for x in ds_all["caption"]]
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(captions, f)
    return captions


def build_corpus(captions, sep="\n<ENDC>\n", case_transform=None):
    if case_transform == "lower":
        captions = [c.lower() for c in captions]
    elif case_transform == "upper":
        captions = [c.upper() for c in captions]

    text = sep.join(captions)

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return {
        "text": text,
        "chars": chars,
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": itos,
        "encode": encode,
        "decode": decode,
        "data": data,
        "train_data": train_data,
        "val_data": val_data,
    }


def make_get_batch(train_data, val_data, cfg):
    def get_batch(split):
        src = train_data if split == "train" else val_data
        ix = torch.randint(len(src) - cfg.block_size - 1, (cfg.batch_size,))
        x = torch.stack([src[i : i + cfg.block_size] for i in ix])
        y = torch.stack([src[i + 1 : i + cfg.block_size + 1] for i in ix])
        return x.to(cfg.device), y.to(cfg.device)

    return get_batch
