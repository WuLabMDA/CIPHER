from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep


class SSLHead(nn.Module):
    """Swin Transformer encoder with rotation, contrastive, and reconstruction heads."""

    def __init__(self, args) -> None:
        super().__init__()

        patch_size = ensure_tuple_rep(args.patch_size, args.spatial_dims)
        window_size = ensure_tuple_rep(args.window_size, args.spatial_dims)

        self.swin_vit = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=list(args.depths),
            num_heads=list(args.num_heads),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
            use_v2=args.use_v2,
        )

        # Four Swin stages produce feature_size * 16 channels in the final stage.
        encoder_dim = args.feature_size * 16
        self.rotation_head = nn.Linear(encoder_dim, 4)
        self.contrastive_head = nn.Linear(encoder_dim, 512)

        self.reconstruction_head = nn.Sequential(
            nn.Conv3d(encoder_dim, encoder_dim // 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(encoder_dim // 2),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(encoder_dim // 2, encoder_dim // 4, kernel_size=3, padding=1),
            nn.InstanceNorm3d(encoder_dim // 4),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(encoder_dim // 4, encoder_dim // 8, kernel_size=3, padding=1),
            nn.InstanceNorm3d(encoder_dim // 8),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(encoder_dim // 8, encoder_dim // 16, kernel_size=3, padding=1),
            nn.InstanceNorm3d(encoder_dim // 16),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(encoder_dim // 16, encoder_dim // 16, kernel_size=3, padding=1),
            nn.InstanceNorm3d(encoder_dim // 16),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(encoder_dim // 16, args.in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.swin_vit(x.contiguous())[4]
        tokens = encoded.flatten(start_dim=2).transpose(1, 2)

        if tokens.shape[1] < 2:
            raise RuntimeError(
                "The final Swin feature map must contain at least two spatial tokens."
            )

        rotation_logits = self.rotation_head(tokens[:, 0])
        contrastive_embedding = self.contrastive_head(tokens[:, 1])
        reconstruction = self.reconstruction_head(encoded)
        return rotation_logits, contrastive_embedding, reconstruction
