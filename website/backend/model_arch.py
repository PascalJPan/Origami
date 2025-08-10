# backend/model_arch.py
import torch
import torch.nn as nn

class ProteinClassifier2(nn.Module):
    """
    Input:  x shape [B, L, 24]  (B=batch, L=sequence length, 24 features/residue)
    Output: logits shape [B, L, num_classes]
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.convLayers = nn.Sequential(
            nn.Conv1d(in_channels=24, out_channels=16, kernel_size=3,  padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5,  padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=7,  padding=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=21, padding=10)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect [B, L, 24]
        if x.dim() != 3:
            raise ValueError(f"Expected x with 3 dims [B, L, 24], got {tuple(x.shape)}")
        if x.size(-1) != 24:
            raise ValueError(f"Expected last dim=24, got {x.size(-1)}")

        # Conv1d expects [B, C, L] → here C=24 features
        x = x.transpose(1, 2)            # [B, 24, L]
        x = self.convLayers(x)           # [B, 32, L]   (note: 32 out channels after final Conv1d)
        x = x.transpose(1, 2)            # [B, L, 32]
        logits = self.classifier(x)      # [B, L, num_classes]
        return logits


def build_model(num_classes: int) -> ProteinClassifier2:
    """Helper for loaders / TorchScript export."""
    return ProteinClassifier2(num_classes=num_classes)
