from __future__ import annotations

import numpy as np
import torch


def patch_random_drop(
    image: torch.Tensor,
    replacement: torch.Tensor | None = None,
    *,
    max_drop: float = 0.3,
    max_block_size: float = 0.25,
    tolerance: float = 0.05,
) -> torch.Tensor:
    """Mask random 3D blocks with noise or blocks from another sample."""
    channels, height, width, depth = image.shape
    target_voxels = np.random.uniform(0.0, max_drop) * height * width * depth

    max_sizes = (
        max(2, int(height * max_block_size)),
        max(2, int(width * max_block_size)),
        max(2, int(depth * max_block_size)),
    )
    min_sizes = (
        max(1, int(height * tolerance)),
        max(1, int(width * tolerance)),
        max(1, int(depth * tolerance)),
    )

    modified_voxels = 0
    while modified_voxels < target_voxels:
        starts = (
            np.random.randint(0, max(1, height - min_sizes[0] + 1)),
            np.random.randint(0, max(1, width - min_sizes[1] + 1)),
            np.random.randint(0, max(1, depth - min_sizes[2] + 1)),
        )
        block_sizes = tuple(
            np.random.randint(low, max(low + 1, high + 1))
            for low, high in zip(min_sizes, max_sizes)
        )
        ends = (
            min(starts[0] + block_sizes[0], height),
            min(starts[1] + block_sizes[1], width),
            min(starts[2] + block_sizes[2], depth),
        )

        slices = (
            slice(None),
            slice(starts[0], ends[0]),
            slice(starts[1], ends[1]),
            slice(starts[2], ends[2]),
        )

        if replacement is None:
            block_shape = (
                channels,
                ends[0] - starts[0],
                ends[1] - starts[1],
                ends[2] - starts[2],
            )
            noise = torch.randn(block_shape, dtype=image.dtype, device=image.device)
            value_range = noise.max() - noise.min()
            if value_range > 1e-8:
                noise = (noise - noise.min()) / value_range
            else:
                noise.zero_()
            image[slices] = noise
        else:
            image[slices] = replacement[slices]

        modified_voxels += (
            (ends[0] - starts[0])
            * (ends[1] - starts[1])
            * (ends[2] - starts[2])
        )

    return image


def random_axial_rotation(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate each sample by 0, 90, 180, or 270 degrees in the axial plane."""
    batch_size = samples.shape[0]
    augmented = samples.detach().clone()
    labels = torch.zeros(batch_size, dtype=torch.long, device=samples.device)

    for index in range(batch_size):
        orientation = int(np.random.randint(0, 4))
        if orientation:
            augmented[index] = samples[index].rot90(orientation, dims=(1, 2))
        labels[index] = orientation

    return augmented, labels


def random_block_augmentation(samples: torch.Tensor, args) -> torch.Tensor:
    """Apply random noise masking and optional cross-sample block replacement."""
    batch_size = samples.shape[0]
    augmented = samples.detach().clone()

    for index in range(batch_size):
        augmented[index] = patch_random_drop(
            augmented[index],
            max_drop=args.max_drop,
            max_block_size=args.max_block_size,
            tolerance=args.block_tolerance,
        )

        replacement_index = int(np.random.randint(0, batch_size))
        if replacement_index != index:
            augmented[index] = patch_random_drop(
                augmented[index],
                augmented[replacement_index],
                max_drop=args.max_drop,
                max_block_size=args.max_block_size,
                tolerance=args.block_tolerance,
            )

    return augmented
