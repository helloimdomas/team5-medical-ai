import os
import json


class ExperimentLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def load(self):
        if os.path.exists(self.log_path):
            with open(self.log_path, "r") as f:
                return json.load(f)
        return []

    def save(self, log):
        with open(self.log_path, "w") as f:
            json.dump(log, f, indent=2)

    def append(self, entry):
        log = self.load()
        log.append(entry)
        self.save(log)
        return len(log)

    def delete(self, index):
        log = self.load()
        removed = log.pop(index)
        self.save(log)
        return removed

    def clear(self):
        self.save([])

    def list(self):
        log = self.load()
        if not log:
            print("  (no experiments logged)")
        for i, exp in enumerate(log):
            label = exp.get("label", "unnamed")
            train = exp.get("final_train_loss", "?")
            val = exp.get("final_val_loss", "?")
            if isinstance(train, float):
                print(f"  [{i}] {label:<30} train={train:.4f}  val={val:.4f}")
            else:
                print(f"  [{i}] {label}")
        return log
