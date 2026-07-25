# CIPHER Public-Release Checklist

Complete the following items before making the repository public.

## Required Files

- [ ] Copy `Fig_1.png` into `assets/Fig_1.png`.
- [ ] Copy `Fig_2.png` into `assets/Fig_2.png`.
- [ ] Add the final fine-tuning and inference scripts under `fine_tuning/`.
- [ ] Add lung-segmentation and patient-level reconstruction-error aggregation code.
- [ ] Add the locked-threshold selection and external inference code.
- [ ] Add model weights only if data-use agreements and institutional policies allow public release.
- [ ] Select and add a repository license.
- [ ] Replace the provisional citation after publication.
- [ ] Confirm the final GitHub organization and repository URL.

## Important Method Reconciliation

The manuscript and the supplied training files currently describe different preprocessing settings.

### Manuscript description

- Voxel spacing: `1.5 × 1.5 × 2.0 mm`
- CT window: `−330 to 150 HU`
- Input block: `96 × 96 × 96 voxels`

### Supplied configuration/code

- Configured spacing values: `0.8 × 0.8 × 2.5 mm`
- CT window: `−1000 to 400 HU`
- Input block: `96 × 96 × 32 voxels`
- The original loader imported `Spacingd` but did not apply it, so the configured spacing values had no effect inside the uploaded pipeline.

The cleaned loader provides an `apply_spacing` switch. It is set to `false` in the supplied reproduction configuration to preserve the behavior of the uploaded code. Confirm the settings actually used for the reported experiments, then update both the manuscript and public configuration so they match.

## Reproducibility Checks

- [ ] Confirm exact Python, PyTorch, CUDA, MONAI, and GPU versions.
- [ ] Confirm the number and model of GPUs used for each stage.
- [ ] Confirm all random seeds and data splits.
- [ ] Confirm whether validation crops should remain random or be deterministic.
- [ ] Confirm whether contrastive negatives are calculated per GPU or gathered across GPUs.
- [ ] Run a clean training test from a newly cloned repository.
- [ ] Verify checkpoint loading and resume behavior.
- [ ] Verify that no institutional paths, patient identifiers, credentials, or protected data remain.
