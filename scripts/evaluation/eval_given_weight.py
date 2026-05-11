import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix, f1_score
from torchvision import transforms


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / "train"))

    from utilities import AITEXPatched  # pylint: disable=import-error
    from model_architectures import BinaryClassifier  # pylint: disable=import-error

    ckpt = project_root / "models" / "bigger_binary_F1_0.98 (1).pth"
    aitex_dir = project_root / "data" / "aitex"

    transform = transforms.Compose([transforms.Resize((224, 224))])
    data = AITEXPatched(str(aitex_dir), transform=transform, greyscale=True)

    device = torch.device("cpu")
    model = BinaryClassifier()
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for img, label in data:
            res = model(img.reshape((1, 1, 224, 224)).to(device))
            y_true.append(int(label))
            y_pred.append(int(res.cpu().item() >= 0.5))

    out = {
        "checkpoint": str(ckpt),
        "dataset_size": len(data),
        "class_counts": {
            "normal": data.has_defect.count(0),
            "defect": data.has_defect.count(1),
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "f1": float(f1_score(y_true, y_pred)),
    }

    out_path = project_root / "outputs" / "teacher" / "given_weight_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
