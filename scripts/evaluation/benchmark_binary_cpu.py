import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "train"))

from model_architectures import BinaryClassifier


def main() -> int:
    model = BinaryClassifier()
    loss_fn = torch.nn.BCELoss()
    result = {
        "params": sum(p.numel() for p in model.parameters()),
        "threads": torch.get_num_threads(),
        "batches": {},
    }
    for batch_size in [1, 16, 32]:
        x = torch.randn(batch_size, 1, 224, 224)
        y = torch.rand(batch_size, 1).round()
        for _ in range(2):
            loss = loss_fn(model(x), y)
            loss.backward()
            model.zero_grad(set_to_none=True)
        started_at = time.time()
        steps = 5
        for _ in range(steps):
            loss = loss_fn(model(x), y)
            loss.backward()
            model.zero_grad(set_to_none=True)
        result["batches"][str(batch_size)] = {
            "sec_per_train_step": (time.time() - started_at) / steps
        }
    out = PROJECT_ROOT / "outputs" / "teacher" / "binary_cpu_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
