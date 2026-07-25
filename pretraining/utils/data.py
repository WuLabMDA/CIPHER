from __future__ import annotations

from pathlib import Path

from monai.data import (
    CacheDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
    SmartCacheDataset,
    load_decathlon_datalist,
)
from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandSpatialCropSamplesd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)


def _build_transforms(args) -> Compose:
    transforms = [
        LoadImaged(keys=["image"], ensure_channel_first=True, image_only=False),
        Orientationd(keys=["image"], axcodes="RAS"),
    ]

    if args.apply_spacing:
        transforms.append(
            Spacingd(
                keys=["image"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode="bilinear",
            )
        )

    transforms.extend(
        [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(
                keys=["image"],
                source_key="image",
                allow_smaller=True,
            ),
            SpatialPadd(
                keys=["image"],
                spatial_size=[args.roi_x, args.roi_y, args.roi_z],
            ),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=True,
            ),
            ResizeWithPadOrCropd(
                keys=["image"],
                spatial_size=[args.roi_x, args.roi_y, args.roi_z],
                method="symmetric",
                mode="constant",
            ),
            ToTensord(keys=["image"]),
        ]
    )
    return Compose(transforms)


def _build_training_dataset(data, transform, args):
    if args.dataset_type == "cache":
        return CacheDataset(
            data=data,
            transform=transform,
            cache_rate=args.cache_rate,
            num_workers=args.num_workers,
        )

    if args.dataset_type == "smartcache":
        cache_num = args.smartcache_cache_num
        if cache_num is None:
            cache_num = 32 * args.batch_size * args.sw_batch_size
        return SmartCacheDataset(
            data=data,
            transform=transform,
            replace_rate=args.smartcache_replace_rate,
            cache_num=cache_num,
            num_init_workers=args.num_workers,
            num_replace_workers=args.num_workers,
        )

    return Dataset(data=data, transform=transform)


def get_loaders(args):
    datalist_path = Path(args.datalist)
    if not datalist_path.exists():
        raise FileNotFoundError(
            f"Data-list JSON was not found: {datalist_path}. "
            "Copy and edit the provided example JSON first."
        )

    base_dir = args.data_root if args.data_root else None
    training_data = load_decathlon_datalist(
        str(datalist_path),
        is_segmentation=False,
        data_list_key="training",
        base_dir=base_dir,
    )
    validation_data = load_decathlon_datalist(
        str(datalist_path),
        is_segmentation=False,
        data_list_key="validation",
        base_dir=base_dir,
    )

    if not training_data:
        raise ValueError("The training data list is empty.")
    if not validation_data:
        raise ValueError("The validation data list is empty.")

    print(f"Training volumes: {len(training_data)}")
    print(f"Validation volumes: {len(validation_data)}")

    train_transform = _build_transforms(args)
    validation_transform = _build_transforms(args)

    train_dataset = _build_training_dataset(training_data, train_transform, args)
    validation_dataset = Dataset(validation_data, validation_transform)

    train_sampler = None
    validation_sampler = None
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        validation_sampler = DistributedSampler(
            validation_dataset,
            shuffle=False,
            drop_last=True,
        )

    common_loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
        "drop_last": True,
    }

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        **common_loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        sampler=validation_sampler,
        shuffle=False,
        **common_loader_kwargs,
    )
    return train_loader, validation_loader
