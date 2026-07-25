from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class ContrastiveLoss(nn.Module):
    """NT-Xent-style contrastive loss used by the original MONAI SSL example."""

    def __init__(self, batch_size: int, temperature: float = 0.5) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        negative_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)
        self.register_buffer("negative_mask", negative_mask.float())

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.shape[0] != self.batch_size or second.shape[0] != self.batch_size:
            raise ValueError(
                "Contrastive batch size changed. Keep drop_last=True and ensure "
                "batch_size * sw_batch_size matches the configured loss batch size."
            )

        first = F.normalize(first, dim=1)
        second = F.normalize(second, dim=1)
        representations = torch.cat([first, second], dim=0)
        similarity = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )

        positive_first_second = torch.diag(similarity, self.batch_size)
        positive_second_first = torch.diag(similarity, -self.batch_size)
        positives = torch.cat([positive_first_second, positive_second_first], dim=0)

        numerator = torch.exp(positives / self.temperature)
        denominator = self.negative_mask * torch.exp(similarity / self.temperature)
        return (-torch.log(numerator / denominator.sum(dim=1))).mean()


class SelfSupervisedLoss(nn.Module):
    def __init__(self, local_crop_batch_size: int, args) -> None:
        super().__init__()
        self.rotation_loss = nn.CrossEntropyLoss()
        self.reconstruction_loss = nn.L1Loss()
        self.contrastive_loss = ContrastiveLoss(
            local_crop_batch_size,
            temperature=args.contrastive_temperature,
        )
        self.rotation_weight = args.rotation_weight
        self.contrastive_weight = args.contrastive_weight
        self.reconstruction_weight = args.reconstruction_weight

    def forward(
        self,
        rotation_logits: torch.Tensor,
        rotation_targets: torch.Tensor,
        first_embedding: torch.Tensor,
        second_embedding: torch.Tensor,
        reconstructed: torch.Tensor,
        target_images: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        rotation = self.rotation_weight * self.rotation_loss(
            rotation_logits, rotation_targets
        )
        contrastive = self.contrastive_weight * self.contrastive_loss(
            first_embedding, second_embedding
        )
        reconstruction = self.reconstruction_weight * self.reconstruction_loss(
            reconstructed, target_images
        )
        total = rotation + contrastive + reconstruction
        return total, (rotation, contrastive, reconstruction)
