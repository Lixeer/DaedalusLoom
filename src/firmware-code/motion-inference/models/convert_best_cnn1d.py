import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_torch


INPUT_CHANNELS = 114
INPUT_LENGTH = 32
NUM_CLASSES = 3
BATCH_SIZE = 4
CALIBRATION_STEPS = 8
DEVICE = "cpu"


class BestCnn1d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(INPUT_CHANNELS, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (INPUT_LENGTH // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x)


def load_state_dict(path: Path) -> dict:
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    cleaned = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def create_calibration_loader() -> DataLoader:
    generator = torch.Generator().manual_seed(20260604)
    samples = [
        torch.randn(INPUT_CHANNELS, INPUT_LENGTH, generator=generator)
        for _ in range(BATCH_SIZE * CALIBRATION_STEPS)
    ]
    return DataLoader(samples, batch_size=BATCH_SIZE, shuffle=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert best_cnn1d.pth to ESP-DL format.")
    parser.add_argument("--pth", default=r"D:\桌面\best_cnn1d.pth", help="Input PyTorch state_dict path.")
    parser.add_argument("--out", default="main/cnn1d_model.espdl", help="Output ESP-DL model path.")
    parser.add_argument("--bits", type=int, default=8, choices=(8, 16), help="Quantization bit width.")
    parser.add_argument("--target", default="esp32p4", help="ESP-DL target, for example esp32p4.")
    args = parser.parse_args()

    model = BestCnn1d().to(DEVICE).eval()
    state_dict = load_state_dict(Path(args.pth))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch. Missing={missing}, unexpected={unexpected}")

    with torch.no_grad():
        dummy = torch.zeros(1, INPUT_CHANNELS, INPUT_LENGTH, dtype=torch.float32)
        output = model(dummy)
    print(f"Loaded model: input={[1, INPUT_CHANNELS, INPUT_LENGTH]}, output={list(output.shape)}")

    quant_setting = QuantizationSettingFactory.espdl_setting()
    calib_loader = create_calibration_loader()

    def collate_fn(batch: torch.Tensor) -> torch.Tensor:
        return batch.to(DEVICE)

    espdl_quantize_torch(
        model=model,
        espdl_export_file=args.out,
        calib_dataloader=calib_loader,
        calib_steps=CALIBRATION_STEPS,
        input_shape=[1, INPUT_CHANNELS, INPUT_LENGTH],
        target=args.target,
        num_of_bits=args.bits,
        collate_fn=collate_fn,
        setting=quant_setting,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_config=True,
        export_test_values=False,
        verbose=1,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
